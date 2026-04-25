# sglang DeepSeek V3/R1 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` | [#11877](https://github.com/sgl-project/sglang/pull/11877) |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_1.mdx` | no direct PR-number commit |
| `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` | no direct PR-number commit |
| `docs_new/docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.mdx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-r1-basic-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-v3-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-v31-deployment.jsx` | no direct PR-number commit |
| `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` | no direct PR-number commit |
| `examples/chat_template/tool_chat_template_deepseekv3.jinja` | [#5908](https://github.com/sgl-project/sglang/pull/5908), [#9525](https://github.com/sgl-project/sglang/pull/9525), [#10209](https://github.com/sgl-project/sglang/pull/10209) |
| `examples/chat_template/tool_chat_template_deepseekv31.jinja` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#9895](https://github.com/sgl-project/sglang/pull/9895), [#14837](https://github.com/sgl-project/sglang/pull/14837) |
| `examples/chat_template/tool_chat_template_deepseekv32.jinja` | [#11063](https://github.com/sgl-project/sglang/pull/11063) |
| `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` | [#19148](https://github.com/sgl-project/sglang/pull/19148) |
| `python/sglang/srt/function_call/deepseekv31_detector.py` | [#9446](https://github.com/sgl-project/sglang/pull/9446), [#11589](https://github.com/sgl-project/sglang/pull/11589), [#13394](https://github.com/sgl-project/sglang/pull/13394) |
| `python/sglang/srt/function_call/deepseekv32_detector.py` | [#14573](https://github.com/sgl-project/sglang/pull/14573), [#14750](https://github.com/sgl-project/sglang/pull/14750), [#15278](https://github.com/sgl-project/sglang/pull/15278), [#16091](https://github.com/sgl-project/sglang/pull/16091), [#18174](https://github.com/sgl-project/sglang/pull/18174) |
| `python/sglang/srt/function_call/deepseekv3_detector.py` | [#6655](https://github.com/sgl-project/sglang/pull/6655), [#7562](https://github.com/sgl-project/sglang/pull/7562), [#9525](https://github.com/sgl-project/sglang/pull/9525) |
| `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541), [#14572](https://github.com/sgl-project/sglang/pull/14572), [#15381](https://github.com/sgl-project/sglang/pull/15381), [#17007](https://github.com/sgl-project/sglang/pull/17007), [#19428](https://github.com/sgl-project/sglang/pull/19428) |
| `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#15086](https://github.com/sgl-project/sglang/pull/15086), [#15938](https://github.com/sgl-project/sglang/pull/15938) |
| `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#12520](https://github.com/sgl-project/sglang/pull/12520), [#18280](https://github.com/sgl-project/sglang/pull/18280), [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` | no direct PR-number commit |
| `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11565](https://github.com/sgl-project/sglang/pull/11565), [#11652](https://github.com/sgl-project/sglang/pull/11652), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12094](https://github.com/sgl-project/sglang/pull/12094), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12816](https://github.com/sgl-project/sglang/pull/12816), [#13236](https://github.com/sgl-project/sglang/pull/13236), [#13459](https://github.com/sgl-project/sglang/pull/13459), [#13646](https://github.com/sgl-project/sglang/pull/13646), ... (33 total) |
| `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` | no direct PR-number commit |
| `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#15938](https://github.com/sgl-project/sglang/pull/15938) |
| `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#18488](https://github.com/sgl-project/sglang/pull/18488), [#19945](https://github.com/sgl-project/sglang/pull/19945) |
| `python/sglang/srt/layers/attention/nsa/transform_index.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#12300](https://github.com/sgl-project/sglang/pull/12300) |
| `python/sglang/srt/layers/attention/nsa/triton_kernel.py` | no direct PR-number commit |
| `python/sglang/srt/layers/attention/nsa/utils.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541), [#15938](https://github.com/sgl-project/sglang/pull/15938), [#17076](https://github.com/sgl-project/sglang/pull/17076) |
| `python/sglang/srt/layers/attention/nsa_backend.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11652](https://github.com/sgl-project/sglang/pull/11652), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12215](https://github.com/sgl-project/sglang/pull/12215), [#12294](https://github.com/sgl-project/sglang/pull/12294), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12788](https://github.com/sgl-project/sglang/pull/12788), [#12964](https://github.com/sgl-project/sglang/pull/12964), [#13022](https://github.com/sgl-project/sglang/pull/13022), [#13236](https://github.com/sgl-project/sglang/pull/13236), ... (28 total) |
| `python/sglang/srt/layers/communicator_nsa_cp.py` | [#12065](https://github.com/sgl-project/sglang/pull/12065), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541) |
| `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` | [#22774](https://github.com/sgl-project/sglang/pull/22774) |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | [#21405](https://github.com/sgl-project/sglang/pull/21405), [#22774](https://github.com/sgl-project/sglang/pull/22774) |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | [#18461](https://github.com/sgl-project/sglang/pull/18461), [#22774](https://github.com/sgl-project/sglang/pull/22774) |
| `python/sglang/srt/models/deepseek_common/utils.py` | [#18461](https://github.com/sgl-project/sglang/pull/18461), [#22774](https://github.com/sgl-project/sglang/pull/22774) |
| `python/sglang/srt/models/deepseek_v2.py` | [#2667](https://github.com/sgl-project/sglang/pull/2667), [#3237](https://github.com/sgl-project/sglang/pull/3237), [#3550](https://github.com/sgl-project/sglang/pull/3550), [#3582](https://github.com/sgl-project/sglang/pull/3582), [#3730](https://github.com/sgl-project/sglang/pull/3730), [#3785](https://github.com/sgl-project/sglang/pull/3785), [#3888](https://github.com/sgl-project/sglang/pull/3888), [#4836](https://github.com/sgl-project/sglang/pull/4836), [#4918](https://github.com/sgl-project/sglang/pull/4918), [#5571](https://github.com/sgl-project/sglang/pull/5571), [#5662](https://github.com/sgl-project/sglang/pull/5662), [#5707](https://github.com/sgl-project/sglang/pull/5707), ... (73 total) |
| `test/manual/layers/attention/nsa/test_act_quant_triton.py` | no direct PR-number commit |
| `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` | [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `test/manual/layers/attention/nsa/test_index_buf_accessor.py` | [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `test/registered/amd/accuracy/mi30x/test_deepseek_r1_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py` | [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` | [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` | [#17179](https://github.com/sgl-project/sglang/pull/17179) |
| `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py` | no direct PR-number commit |
| `test/registered/backends/test_deepseek_r1_fp8_trtllm_backend.py` | no direct PR-number commit |
| `test/registered/kernels/test_nsa_indexer.py` | [#17076](https://github.com/sgl-project/sglang/pull/17076), [#18280](https://github.com/sgl-project/sglang/pull/18280), [#18389](https://github.com/sgl-project/sglang/pull/18389), [#19041](https://github.com/sgl-project/sglang/pull/19041) |
| `test/registered/unit/mem_cache/test_nsa_pool_host_unit.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 128
- Extra PRs preserved from existing docs: 111
- Total PRs in this document: 238
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2024-01-15 | [#7](https://github.com/sgl-project/sglang/pull/7) | merged | fix radix cache match | `python/sglang/srt/managers/router/radix_cache.py` |
| 2024-12-29 | [#2637](https://github.com/sgl-project/sglang/pull/2637) | merged | AMD: set weights and scaling numbers properly for block FP8 | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD DeepSeek_V3 FP8 Numerical fix | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-01-03 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | [Feature, Hardware] Enable DeepseekV3 on AMD GPUs | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| 2025-02-07 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | Feature/docs deepseek usage and add multi-node |  |
| 2025-02-10 | [#3466](https://github.com/sgl-project/sglang/pull/3466) | merged | Support Eagle2 for Triton backend | `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | refine deepseek_v3 launch server doc | `benchmark/deepseek_v3/README.md` |
| 2025-02-14 | [#3550](https://github.com/sgl-project/sglang/pull/3550) | merged | feat: support flashinfer mla attention for deepseek v3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-24 | [#3785](https://github.com/sgl-project/sglang/pull/3785) | merged | Refactor flashinfer logic for deepseek v3 and fix accuracy bug | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-24 | [#3730](https://github.com/sgl-project/sglang/pull/3730) | merged | Feature DeepSeek V3/R1 INT8 Quantization (block-wise) | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-25 | [#3237](https://github.com/sgl-project/sglang/pull/3237) | merged | [ROCm] Enable Fused MLA Triton kernel for DeepSeekV3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-03-02 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | add deepgemm and sglang fp8 block-wise gemm benchmark | `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md` |
| 2025-03-07 | [#3888](https://github.com/sgl-project/sglang/pull/3888) | merged | [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise) | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | Support nextn for flashinfer mla attention backend | `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/srt/test_mla_flashinfer.py` |
| 2025-03-10 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGemm integrate to sgl-kernel | `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh` |
| 2025-03-11 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | linear support deepgemm | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py` |
| 2025-03-13 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | add INT8 example into dsv3 README | `benchmark/deepseek_v3/README.md` |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | Support FlashMLA backend | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-03-19 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | Support FlashMLA backend cuda graph | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/server_args.py` |
| 2025-03-21 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | Set deepgemm to the default value in the hopper architecture. | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` |
| 2025-03-24 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | [ROCm] Enable MTP (NextN) on AMD GPU | `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py` |
| 2025-03-29 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | [Feature] Support FA3 backend for MLA | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-03-29 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | Add deepseek style fused moe group gate selection kernel | `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py` |
| 2025-04-03 | [#4817](https://github.com/sgl-project/sglang/pull/4817) | merged | [sgl-kernel] per token group quant support COLUMN MAJOR | `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py` |
| 2025-04-04 | [#4918](https://github.com/sgl-project/sglang/pull/4918) | merged | Add DeepSeek V3/R1 shared experts fusion | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-10 | [#5190](https://github.com/sgl-project/sglang/pull/5190) | merged | Fix DeepSeek error when using DeepEP mode | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-10 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | reduce moe_align_block_size_kernel small batch mode overhead | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py` |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | fix: use deepgemm only on hopper | `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2025-04-12 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | feat: use fa3 mla by default on hopper | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py` |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | Apply deepseek cuda rope | `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | apply fused moe gate in ds v3/r1 | `python/sglang/srt/layers/moe/topk.py` |
| 2025-04-15 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | [Fix] Turn off DeepGEMM by default | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `docs/references/deepseek.md` |
| 2025-04-15 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | kernel: support slightly faster merge_state_v2 cuda kernel | `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py` |
| 2025-04-16 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | Support MHA with chunked prefix cache for DeepSeek chunked prefill | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2025-04-17 | [#5205](https://github.com/sgl-project/sglang/pull/5205) | merged | Refactor DeepSeek decoder layer branches | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-18 | [#4836](https://github.com/sgl-project/sglang/pull/4836) | merged | Introduce moe_dense_tp_size to fix dense layer errors in DeepSeek V3 + 4x8xH100 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-18 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | Remove one kernel in per_tensor_quant_mla_fp8 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-04-20 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | [perf] introduce deep gemm group_gemm_masked as bmm | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/test/test_block_fp8.py` |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | enable DeepSeek V3 shared_experts_fusion in sm90 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-22 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | Remove extra copy in deepseek forward absorb | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/rotary_embedding.py`, `.github/workflows/pr-test-amd.yml` |
| 2025-04-22 | [#5641](https://github.com/sgl-project/sglang/pull/5641) | merged | [feature] Add H20 fp8_w8a8 FusedMoE config for --n-share-experts-fusion=16 | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | Turn on DeepGemm By Default and Update Doc | `python/sglang/srt/layers/quantization/deep_gemm.py`, `docs/references/deepseek.md` |
| 2025-04-23 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | Fuse q_a_proj and kv_a_proj for DeepSeek models | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions. | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| 2025-04-27 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-04-27 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | Fuse MLA set kv cache kernel | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-28 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Add Cutlass MLA attention backend | `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2025-04-29 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | Auto set draft model path for MTP | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | Update ci test and doc for MTP api change | `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md` |
| 2025-05-02 | [#5908](https://github.com/sgl-project/sglang/pull/5908) | merged | feat: Refactor DeepSeekV3 function call | `examples/chat_template/tool_chat_template_deepseekv3.jinja` |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | Overlap qk norm with two streams | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-04 | [#6011](https://github.com/sgl-project/sglang/pull/6011) | open | feat: flashinfer_mla speculative decoding with custom mask | `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-05-08 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | Update doc for MLA attention backends | `docs/references/deepseek.md`, `docs/backend/server_arguments.md` |
| 2025-05-08 | [#6079](https://github.com/sgl-project/sglang/pull/6079) | merged | Clean logs for DeepSeek-V3 launching | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-08 | [#5662](https://github.com/sgl-project/sglang/pull/5662) | merged | [perf] dsv3 bmm fallback to bf16 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-15 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | [Feat] Support FlashMLA backend with MTP and FP8 KV cache | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-05-18 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | [Feat] optimize Qwen3 on H20 by hybrid Attention Backend | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py` |
| 2025-05-28 | [#6655](https://github.com/sgl-project/sglang/pull/6655) | merged | fix(deepseekv3): Fix DeepSeekV3Detector tool_index assignment and multi-tool call streaming support | `python/sglang/srt/function_call/deepseekv3_detector.py` |
| 2025-05-29 | [#6738](https://github.com/sgl-project/sglang/pull/6738) | open | Partially supports using MHA kernels in MLA forward when page-size > 1. | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-29 | [#6265](https://github.com/sgl-project/sglang/pull/6265) | merged | [fix][RL] Fix DeepSeekV3ForCausalLM.post_load_weights for multiple update weight | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-06-07 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | Fuse routed scaling factor in topk_reduce kernel | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-06-08 | [#6853](https://github.com/sgl-project/sglang/pull/6853) | merged | [DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | Fuse routed scaling factor in deepseek | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-06-09 | [#7005](https://github.com/sgl-project/sglang/pull/7005) | open | FlowMLA: zero-overhead DP MLA throughput boost via memory optimization | `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | Support new DeepGEMM format in per token group quant | `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h` |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | Support new DeepGEMM format in per token group quant (part 2: srt) | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/entrypoints/engine.py`, `python/pyproject.toml` |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | Re-quantize DeepSeek model weights to support DeepGEMM new input format | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py` |
| 2025-06-14 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | Refactor DeepGEMM integration | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | Support new DeepGEMM | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-14 | [#7181](https://github.com/sgl-project/sglang/pull/7181) | merged | [fix] fix dsv3 weight loader tqdm and simplify shared experts fusion | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-15 | [#7180](https://github.com/sgl-project/sglang/pull/7180) | merged | [fix] fix determine_num_fused_shared_experts | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-17 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | feat: mtp support dp-attention | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/layers/attention/triton_backend.py` |
| 2025-06-17 | [#7164](https://github.com/sgl-project/sglang/pull/7164) | merged | Fix Deepseek R1 0528 FP4 tensor name mismatch issue during weights loading. | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-17 | [#7286](https://github.com/sgl-project/sglang/pull/7286) | merged | fix: resolve b200 dsv3 mtp issue | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-20 | [#7371](https://github.com/sgl-project/sglang/pull/7371) | merged | Fix judgment condition for enabling Deepseek V3/R1 shared expert fusion optimization | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-23 | [#7481](https://github.com/sgl-project/sglang/pull/7481) | merged | [perf] slightly imporve DeepSeek-R1-FP4 TP8 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-24 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | Fix MTP with Deepseek R1 Fp4 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-02 | [#7635](https://github.com/sgl-project/sglang/pull/7635) | merged | Apply dsv3_fused_a_gemm kernel | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-02 | [#7677](https://github.com/sgl-project/sglang/pull/7677) | merged | Apply dsv3 router gemm kernel for deepseek-r1 fp4 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-03 | [#7735](https://github.com/sgl-project/sglang/pull/7735) | merged | fix awq and dsv3 fused gemm compatible | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-03 | [#7738](https://github.com/sgl-project/sglang/pull/7738) | merged | fix dsv3 fused proj check | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-03 | [#7750](https://github.com/sgl-project/sglang/pull/7750) | merged | [fix] fix dsv3_router_gemm filter | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-06 | [#7793](https://github.com/sgl-project/sglang/pull/7793) | merged | fix: disable dsv3_router_gemm in dsv3_nextn | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-07 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-24 | [#7562](https://github.com/sgl-project/sglang/pull/7562) | merged | Fix incomplete tool call capture issue in streaming response of DeepSeek-V3 when enable MTP | `python/sglang/srt/function_call/deepseekv3_detector.py` |
| 2025-07-28 | [#8311](https://github.com/sgl-project/sglang/pull/8311) | closed | Support DeepSeek-R1 w4a8 low latency deepep | `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-08-01 | [#8051](https://github.com/sgl-project/sglang/pull/8051) | merged | Update batch size limitation of dsv3_router_gemm kernel to 16 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-08-23 | [#9525](https://github.com/sgl-project/sglang/pull/9525) | merged | tool-call(dsv3): Improve deepseek-v3 chat template and tool_choice = `required` | `python/sglang/srt/function_call/deepseekv3_detector.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja` |
| 2025-08-27 | [#9446](https://github.com/sgl-project/sglang/pull/9446) | merged | Support DeepSeek-V3.1 tool call | `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-08-30 | [#9815](https://github.com/sgl-project/sglang/pull/9815) | merged | fix: dsv3 lite q_lora_rank none | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-02 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-03 | [#9895](https://github.com/sgl-project/sglang/pull/9895) | merged | Update tool_chat_template_deepseekv31.jinja | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-09-03 | [#9671](https://github.com/sgl-project/sglang/pull/9671) | merged | Optimized deepseek-v3/r1 model performance on mxfp4 run | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-03 | [#9959](https://github.com/sgl-project/sglang/pull/9959) | merged | Revert "Optimized deepseek-v3/r1 model performance on mxfp4 run (#9671)" | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-04 | [#10008](https://github.com/sgl-project/sglang/pull/10008) | merged | Optimized deepseek-v3/r1 model performance on mxfp4 run | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-04 | [#8677](https://github.com/sgl-project/sglang/pull/8677) | merged | Fix accuracy drop of dsv3 run in dp enablement | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-06 | [#9834](https://github.com/sgl-project/sglang/pull/9834) | merged | perf: Avoid unnecessary data type conversions for DeepSeek-V3 on Blackwell | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-08 | [#10104](https://github.com/sgl-project/sglang/pull/10104) | merged | Fix run time error in dsv3-fp8 model on mi35x | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-09-09 | [#10209](https://github.com/sgl-project/sglang/pull/10209) | merged | tool-call(dsv3): Fixed a parse problem when there are multiple function definitions in tool_calls | `examples/chat_template/tool_chat_template_deepseekv3.jinja` |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py` |
| 2025-10-05 | [#11063](https://github.com/sgl-project/sglang/pull/11063) | merged | Add DeepSeek-V3.2 Tool Call Template | `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2025-10-06 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | Support DeepSeek V3.2 Exp | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-13 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | Update DeepSeek-R1-FP4 default config on blackwell | `python/sglang/srt/server_args.py` |
| 2025-10-14 | [#11565](https://github.com/sgl-project/sglang/pull/11565) | merged | [DSv32] Use torch.compile for _get_logits_head_gate | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-15 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | [1/N]Support DeepSeek-R1 w4a8 normal deepep | `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-10-16 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-17 | [#11109](https://github.com/sgl-project/sglang/pull/11109) | closed | [Draft] Support MTP for DeepSeek-V3.2 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-17 | [#11682](https://github.com/sgl-project/sglang/pull/11682) | merged | Cleaning indexer for DeepSeek V3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-10-19 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | [Spec Decoding] Support MTP for dsv3.2 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-24 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py` |
| 2025-10-25 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | [2/N]Support DeepSeek-R1 w4a8 low latency deepep | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-10-25 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | [Doc] Add documentation for DeepSeek V3.2 | `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` |
| 2025-10-27 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | [doc] add example of using w4fp8 for Deepseek | `benchmark/deepseek_v3/README.md` |
| 2025-10-28 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | Support running FP4 Deepseek on SM120. | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-28 | [#11655](https://github.com/sgl-project/sglang/pull/11655) | merged | [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` |
| 2025-10-29 | [#12294](https://github.com/sgl-project/sglang/pull/12294) | merged | [Deepseek V3.2] Enable flashmla_auto with MTP | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-10-29 | [#12095](https://github.com/sgl-project/sglang/pull/12095) | merged | [2/2] Deepseek deterministic: support deepseek v3 deterministic inference on 8 x H200 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-30 | [#12094](https://github.com/sgl-project/sglang/pull/12094) | merged | Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-31 | [#12300](https://github.com/sgl-project/sglang/pull/12300) | merged | [DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type | `python/sglang/srt/layers/attention/nsa/transform_index.py` |
| 2025-11-06 | [#11892](https://github.com/sgl-project/sglang/pull/11892) | merged | DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | Update dsv3 quantization auto setting for sm100 | `python/sglang/srt/server_args.py` |
| 2025-11-07 | [#12788](https://github.com/sgl-project/sglang/pull/12788) | merged | [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-07 | [#12520](https://github.com/sgl-project/sglang/pull/12520) | merged | [Test] Add DeepSeekV3.2 NSA Indexer Test Suite | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2025-11-07 | [#12816](https://github.com/sgl-project/sglang/pull/12816) | merged | [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-08 | [#12582](https://github.com/sgl-project/sglang/pull/12582) | merged | [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel | `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-11-11 | [#12689](https://github.com/sgl-project/sglang/pull/12689) | merged | [ROCM] Optimized deepseek-r1 model with rmsnorm + fp8 quant fusion | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-12 | [#12583](https://github.com/sgl-project/sglang/pull/12583) | merged | [Deepseek V3.2] Fix accuracy bug in the Indexer | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-12 | [#12215](https://github.com/sgl-project/sglang/pull/12215) | merged | [DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat` | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-14 | [#11589](https://github.com/sgl-project/sglang/pull/11589) | merged | [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2025-11-14 | [#13236](https://github.com/sgl-project/sglang/pull/13236) | merged | [Deepseek V3.2] Clean up MTP | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-17 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | (1/n)support context parallel with deepseekv3.2-DSA | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-17 | [#13022](https://github.com/sgl-project/sglang/pull/13022) | merged | [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-18 | [#13264](https://github.com/sgl-project/sglang/pull/13264) | merged | [NVIDIA] Fix broken fp8 MoE of deepseek v3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | [Fix] Fix DeepSeek V3 MTP on B200 | `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-11-20 | [#13617](https://github.com/sgl-project/sglang/pull/13617) | merged | [ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-20 | [#12964](https://github.com/sgl-project/sglang/pull/12964) | merged | [DeepseekV3.2] Deepseek fp8 support for MHA path | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-20 | [#13459](https://github.com/sgl-project/sglang/pull/13459) | merged | [Deepseek V3.2] Change indexer weights_proj to fp32 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-21 | [#13705](https://github.com/sgl-project/sglang/pull/13705) | merged | [AMD] Enable fused shared expert append and flatten quant for fp8 deepseekR1 model | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-24 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-11-25 | [#13544](https://github.com/sgl-project/sglang/pull/13544) | merged | [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-26 | [#13954](https://github.com/sgl-project/sglang/pull/13954) | merged | Fix Deepseek v3.1 loading issue | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-30 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | [DeepSeekV3.2] Enable pure TP & Partial DP Attention | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-01 | [#14194](https://github.com/sgl-project/sglang/pull/14194) | open | [feature] implement dcp for deepseek_v2 | `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2025-12-04 | [#14325](https://github.com/sgl-project/sglang/pull/14325) | merged | [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-06 | [#13115](https://github.com/sgl-project/sglang/pull/13115) | merged | support mtp with deepseek r1 nvfp4 model | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-08 | [#14573](https://github.com/sgl-project/sglang/pull/14573) | merged | [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-10 | [#14837](https://github.com/sgl-project/sglang/pull/14837) | merged | [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210) | `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-12-11 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | [NPU]dsv3.2 cp for npu | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py` |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | Fix dsv3 dp accuracy issue when using bf16-kv | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2025-12-12 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | [NPU] optimization for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2025-12-15 | [#15086](https://github.com/sgl-project/sglang/pull/15086) | merged | [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` |
| 2025-12-16 | [#14975](https://github.com/sgl-project/sglang/pull/14975) | merged | [AMD] Support fused_rms_mxfp4_quant in the prefill stage for DeepSeek-R1-MXFP4 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-17 | [#15315](https://github.com/sgl-project/sglang/pull/15315) | open | [Performance] Optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe | `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | Fix the accuracy issue when running mxfp4 dsv3 model and enable ep | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py` |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | [Deepseek V3.2] Support Overlap Spec + NSA | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-18 | [#15278](https://github.com/sgl-project/sglang/pull/15278) | merged | feat: DeepSeek-V3.2 Streaming tool call output | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-18 | [#15380](https://github.com/sgl-project/sglang/pull/15380) | open | [Performance] sgl-kernel, optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe | `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu`, `sgl-kernel/python/sgl_kernel/cutlass_moe.py` |
| 2025-12-18 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | [perf]optimize w4afp8 kernel on deepseek-v3-0324 | `python/sglang/srt/layers/quantization/w4afp8.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` |
| 2025-12-19 | [#15429](https://github.com/sgl-project/sglang/pull/15429) | merged | [Deepseek V3.2] Fix Deepseek MTP in V1 mode | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-19 | [#15040](https://github.com/sgl-project/sglang/pull/15040) | merged | [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-21 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | Support piecewise cuda graph for dsv3 fp4 | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-26 | [#14750](https://github.com/sgl-project/sglang/pull/14750) | merged | [Tool Call][DSV32] Streamline function call parameters | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-29 | [#14280](https://github.com/sgl-project/sglang/pull/14280) | merged | feat PD: add eagle3 support for DeepSeek V3 in EP mode | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-31 | [#13394](https://github.com/sgl-project/sglang/pull/13394) | merged | Fix DeepSeekV31's structural tag trigger | `python/sglang/srt/function_call/deepseekv31_detector.py` |
| 2026-01-02 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-07 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | Clean Some Environment Variables for DeepSeek V32 | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2026-01-10 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | [DSv32] Overlap indexer weights_proj during dual_stream decode | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | `python/sglang/test/run_eval.py` |
| 2026-01-16 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-01-18 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | [Refactor] Split out deepseek v2 weight loader function into mixin | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py` |
| 2026-01-19 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2026-01-19 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-20 | [#17179](https://github.com/sgl-project/sglang/pull/17179) | merged | [AMD] Add DeepSeek-V3.2 and VLMs model in nightly tests | `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` |
| 2026-01-20 | [#17205](https://github.com/sgl-project/sglang/pull/17205) | merged | [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-23 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | [NPU]bugfix: fix for dsv3.2 and dsvl2 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-01-23 | [#16758](https://github.com/sgl-project/sglang/pull/16758) | merged | [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-25 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-26 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | [NPU]DeepSeek-V3.2 support npu mlaprolog | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-01-28 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | [DSv32] Overlap indexer qk projection and activation quant | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-28 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` |
| 2026-01-31 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-02-02 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-02 | [#17964](https://github.com/sgl-project/sglang/pull/17964) | merged | [NPU] support dsv32 radixcache on ascend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-10 | [#18297](https://github.com/sgl-project/sglang/pull/18297) | merged | Deepseekv32 compatibility with transformers v5 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-10 | [#18488](https://github.com/sgl-project/sglang/pull/18488) | merged | Tilelang sparse decode fwd for dsv32 mi355 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-02-11 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | Fix Bug on dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-12 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp | `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile` |
| 2026-02-15 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | Fix model loading for DeepSeek-V3.2-AWQ | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-16 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-02-16 | [#18892](https://github.com/sgl-project/sglang/pull/18892) | open | [Kernel] Add JIT support for DeepSeek V3 GEMM | `python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh`, `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` |
| 2026-02-19 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | [AMD] Fix mi35x dsv32 mtp nightly | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-22 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-02-24 | [#18624](https://github.com/sgl-project/sglang/pull/18624) | merged | [AMD] DSR1/V3 use fp8 bmm in MLA for MI300X | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-25 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | [ROCm] Optimize Deepseek R1 on MI300X | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-26 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` |
| 2026-02-27 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4 | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-02-27 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-03-02 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | [Feature] add feature mla_ag_after_qlora for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-03-03 | [#18174](https://github.com/sgl-project/sglang/pull/18174) | merged | [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-03-03 | [#16091](https://github.com/sgl-project/sglang/pull/16091) | merged | [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format. | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-03-05 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek) | `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-06 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-11 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | [deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug | `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py` |
| 2026-03-17 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2026-03-19 | [#9744](https://github.com/sgl-project/sglang/pull/9744) | merged | [CPU] Add FP8 Bmm support | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-03-19 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256. | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-19 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | Fix gpu-fault issue when run deepseek-r1 and enable dp | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-03-24 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-24 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-03-27 | [#21526](https://github.com/sgl-project/sglang/pull/21526) | open | Fix aiter router GEMM regression for non-DSR1 MoE models on ROCm gfx95 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-27 | [#21529](https://github.com/sgl-project/sglang/pull/21529) | open | Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-27 | [#21530](https://github.com/sgl-project/sglang/pull/21530) | open | [ROCm] Fix fused MLA decode rope path for Kimi K2.5 and DeepSeek-variant models | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`, `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` |
| 2026-03-27 | [#21531](https://github.com/sgl-project/sglang/pull/21531) | open | [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` |
| 2026-03-30 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | [Intel GPU] Enable DeepSeek R1 inference on XPU | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication" | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-03-31 | [#21657](https://github.com/sgl-project/sglang/pull/21657) | merged | [AMD] Use tgemm.mm for MoEGate router gemm in deepseek_v2.py | `python/sglang/srt/layers/rocm_linear_utils.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-04 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | [RL] Support mxfp8 DeepSeek V3 | `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2026-04-04 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | Add dsv3 router gemm benchmark on blackwell | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-05 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | Enable IndexCache for DeepSeek V3.2 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-04-07 | [#22268](https://github.com/sgl-project/sglang/pull/22268) | open | [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-09 | [#20089](https://github.com/sgl-project/sglang/pull/20089) | merged | feat: [1/2] [DeepEP] Fuse shared expert into MoE dispatch under EP | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-04-09 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | [Lora] Lora quat info re-factor and support deepseekv3 mla lora | `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/moe_wna16.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py` |
| 2026-04-10 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-16 | [#22938](https://github.com/sgl-project/sglang/pull/22938) | open | [AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122) | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-17 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors | `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` |
| 2026-04-20 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-04-21 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | [CPU] expand the interface of shared_expert without scaling factor | `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp` |
| 2026-04-24 | [#22774](https://github.com/sgl-project/sglang/pull/22774) | merged | [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) | `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |

## Per-PR Diff Audit Cards

### PR #7 - fix radix cache match

- Link: https://github.com/sgl-project/sglang/pull/7
- Status/date: merged / 2024-01-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix radix cache match"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/managers/router/radix_cache.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/managers/router/radix_cache.py` modified +2/-2 (4 lines); hunks: -116,12 +116,12 @@ def _match_prefix_helper(self, node, key, value, last_node):; symbols: _match_prefix_helper, touching `_match_prefix_helper`.
- Code diff details:
  - `python/sglang/srt/managers/router/radix_cache.py` modified +2/-2 (4 lines); hunks: -116,12 +116,12 @@ def _match_prefix_helper(self, node, key, value, last_node):; symbols: _match_prefix_helper
- Key code excerpts:

```diff
diff -- python/sglang/srt/managers/router/radix_cache.py
@@ -116,12 +116,12 @@ def _match_prefix_helper(self, node, key, value, last_node):
-                if prefix_len == len(key) and prefix_len != len(c_key):
+                if prefix_len < len(c_key):
-                    value.append(child.value[:prefix_len])
+                    value.append(child.value)
```

- Reviewed files:
  - runtime: `python/sglang/srt/managers/router/radix_cache.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/router/radix_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2637 - AMD: set weights and scaling numbers properly for block FP8

- Link: https://github.com/sgl-project/sglang/pull/2637
- Status/date: merged / 2024-12-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +56/-6, 120 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "AMD: set weights and scaling numbers properly for block FP8"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; PR body summary: As it is.
- Key implementation: `python/sglang/srt/layers/quantization/fp8.py` modified +38/-1 (39 lines); hunks: -272,6 +272,19 @@ def create_weights(; -369,7 +382,7 @@ def apply(; symbols: create_weights, process_weights_after_loading, apply, touching `create_weights, process_weights_after_loading, apply`; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +10/-3 (13 lines); hunks: -22,7 +22,10; -73,7 +76,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8, touching `per_token_group_quant_fp8`; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +8/-2 (10 lines); hunks: -7,6 +7,9; -63,8 +66,11 @@ def input_to_float8(; symbols: normalize_e4m3fn_to_e4m3fnuz, input_to_float8, touching `normalize_e4m3fn_to_e4m3fnuz, input_to_float8`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +38/-1 (39 lines); hunks: -272,6 +272,19 @@ def create_weights(; -369,7 +382,7 @@ def apply(; symbols: create_weights, process_weights_after_loading, apply
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +10/-3 (13 lines); hunks: -22,7 +22,10; -73,7 +76,7 @@ def per_token_group_quant_fp8(; symbols: per_token_group_quant_fp8
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +8/-2 (10 lines); hunks: -7,6 +7,9; -63,8 +66,11 @@ def input_to_float8(; symbols: normalize_e4m3fn_to_e4m3fnuz, input_to_float8
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -272,6 +272,19 @@ def create_weights(
+            # If ROCm, normalize the weights and scales to e4m3fnuz
+            if is_hip():
+                # activation_scheme: dynamic
+                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
+                    weight=layer.weight,
+                    weight_scale=layer.weight_scale_inv,
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -22,7 +22,10 @@
-from sglang.srt.utils import get_device_name
+from sglang.srt.utils import get_device_name, is_hip
+is_hip_ = is_hip()
+fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn
@@ -73,7 +76,7 @@ def per_token_group_quant_fp8(
-    dtype: torch.dtype = torch.float8_e4m3fn,
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -7,6 +7,9 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +38/-1; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +10/-3; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2667 - AMD DeepSeek_V3 FP8 Numerical fix

- Link: https://github.com/sgl-project/sglang/pull/2667
- Status/date: merged / 2024-12-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `c5210dfa3802`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-7, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "AMD DeepSeek_V3 FP8 Numerical fix"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Together with core changes from: https://github.com/sgl-project/sglang/pull/2637 Several places in deepseek_v2.py needs extra normalization to weights (to e4m3fnuz), scaling num....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +34/-7 (41 lines); hunks: -46,6 +46,7; -55,7 +56,9; symbols: forward_absorb, load_weights, DeepseekV3ForCausalLM, touching `forward_absorb, load_weights, DeepseekV3ForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-7 (41 lines); hunks: -46,6 +46,7; -55,7 +56,9; symbols: forward_absorb, load_weights, DeepseekV3ForCausalLM
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -46,6 +46,7 @@
+    normalize_e4m3fn_to_e4m3fnuz,
@@ -55,7 +56,9 @@
-from sglang.srt.utils import is_flashinfer_available
+from sglang.srt.utils import is_flashinfer_available, is_hip
+is_hip_ = is_hip()
@@ -573,7 +576,13 @@ def forward_absorb(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +34/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #2601 - [Feature, Hardware] Enable DeepseekV3 on AMD GPUs

- Link: https://github.com/sgl-project/sglang/pull/2601
- Status/date: merged / 2025-01-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-5, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature, Hardware] Enable DeepseekV3 on AMD GPUs"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`; PR body summary: - Support DeepseekV3 on AMD Instinct MI300X GPU - Add proper fix for AMD FP8 to support DeepseekV3 FP8 model - Bypass to cast FP8 to BF16 in MLA - Add AMD config.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +5/-5 (10 lines); hunks: -477,9 +477,9 @@ def invoke_fused_moe_kernel(; -614,7 +614,7 @@ def get_default_config(; symbols: invoke_fused_moe_kernel, get_default_config, fused_experts_impl, touching `invoke_fused_moe_kernel, get_default_config, fused_experts_impl`; `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +4/-0 (4 lines); hunks: -406,6 +406,10 @@ def _decode_grouped_att_m_fwd(; symbols: _decode_grouped_att_m_fwd, touching `_decode_grouped_att_m_fwd`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +5/-5 (10 lines); hunks: -477,9 +477,9 @@ def invoke_fused_moe_kernel(; -614,7 +614,7 @@ def get_default_config(; symbols: invoke_fused_moe_kernel, get_default_config, fused_experts_impl
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +4/-0 (4 lines); hunks: -406,6 +406,10 @@ def _decode_grouped_att_m_fwd(; symbols: _decode_grouped_att_m_fwd
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -477,9 +477,9 @@ def invoke_fused_moe_kernel(
-        padded_size = padding_size
+            padded_size = padding_size
@@ -614,7 +614,7 @@ def get_default_config(
-                "num_stages": 4,
+                "num_stages": 2 if is_hip_flag else 4,
@@ -623,7 +623,7 @@ def get_default_config(
diff -- python/sglang/srt/layers/attention/triton_ops/decode_attention.py
@@ -406,6 +406,10 @@ def _decode_grouped_att_m_fwd(
+    # [TODO] work around shmem limit on MI3xx
+    if is_hip_ and Lk >= 576:
+        BLOCK = 16
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +5/-5; `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3314 - Feature/docs deepseek usage and add multi-node

- Link: https://github.com/sgl-project/sglang/pull/3314
- Status/date: merged / 2025-02-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 0 files, +0/-0, 0 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Feature/docs deepseek usage and add multi-node"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: the file-level diff; PR body summary: People are using the Deepseek V3 model, our guidance are not centralised for the ease of usage. Update: docs\references\deepseek.md Update: docs\references\multi_node.md Remove:....
- Key implementation: GitHub returned no file-level patch; this card keeps the PR metadata and should be re-opened manually before code changes depend on it.
- Code diff details:
  - No patch file list returned by GitHub.
- Key code excerpts:

```diff
No textual patch was returned by GitHub for the selected changed files.
```

- Reviewed files:
  - No patch file list returned by GitHub.
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #3466 - Support Eagle2 for Triton backend

- Link: https://github.com/sgl-project/sglang/pull/3466
- Status/date: merged / 2025-02-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +286/-42, 513 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Eagle2 for Triton backend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/triton_backend.py`, `python/sglang/srt/layers/attention/triton_ops/extend_attention.py`, `python/sglang/srt/speculative/eagle_worker.py`; PR body summary: Support Eagle2 for Triton backend and achieve **2.6x** speedup on batch size 1 for no cuda graph setting. (Based on https://github.com/sgl-project/sglang/pull/3317, https://gith....
- Key implementation: `python/sglang/srt/layers/attention/triton_backend.py` modified +223/-24 (247 lines); hunks: -3,6 +3,7; -18,7 +19,12; symbols: TritonAttnBackend, __init__, touching `TritonAttnBackend, __init__`; `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +4/-4 (8 lines); hunks: -50,7 +50,7 @@ def _fwd_kernel(; -87,7 +87,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd, touching `_fwd_kernel, extend_attention_fwd`; `python/sglang/srt/speculative/eagle_worker.py` modified +24/-8 (32 lines); hunks: -65,15 +65,31 @@ def __init__(; symbols: __init__, touching `__init__`; `test/srt/test_eagle_infer.py` modified +29/-0 (29 lines); hunks: -193,5 +193,34 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestEAGLEServerTriton, setUpClass, touching `test_gsm8k, TestEAGLEServerTriton, setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +223/-24 (247 lines); hunks: -3,6 +3,7; -18,7 +19,12; symbols: TritonAttnBackend, __init__
  - `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +4/-4 (8 lines); hunks: -50,7 +50,7 @@ def _fwd_kernel(; -87,7 +87,7 @@ def _fwd_kernel(; symbols: _fwd_kernel, extend_attention_fwd
  - `python/sglang/srt/speculative/eagle_worker.py` modified +24/-8 (32 lines); hunks: -65,15 +65,31 @@ def __init__(; symbols: __init__
  - `test/srt/test_eagle_infer.py` modified +29/-0 (29 lines); hunks: -193,5 +193,34 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestEAGLEServerTriton, setUpClass
  - `test/srt/test_triton_attention_kernels.py` modified +6/-6 (12 lines); hunks: -102,7 +102,7 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV, D):; -115,16 +115,16 @@ def _test_extend_attention_once(self, B, N_CTX, H_Q, H_KV,...; symbols: _test_extend_attention_once
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -3,6 +3,7 @@
+import triton
@@ -18,7 +19,12 @@
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
+        model_runner: ModelRunner,
diff -- python/sglang/srt/layers/attention/triton_ops/extend_attention.py
@@ -50,7 +50,7 @@ def _fwd_kernel(
-    mask_offsets,
+    mask_indptr,
@@ -87,7 +87,7 @@ def _fwd_kernel(
-        cur_seq_mask_start_idx = tl.load(mask_offsets + cur_seq)
+        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)
@@ -288,7 +288,7 @@ def extend_attention_fwd(
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -65,15 +65,31 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/triton_backend.py` modified +223/-24; `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` modified +4/-4; `python/sglang/srt/speculative/eagle_worker.py` modified +24/-8
  - tests: `test/srt/test_eagle_infer.py` modified +29/-0; `test/srt/test_triton_attention_kernels.py` modified +6/-6
- Risk and verification: The diff ships test coverage in `test/srt/test_eagle_infer.py`, `test/srt/test_triton_attention_kernels.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #3522 - refine deepseek_v3 launch server doc

- Link: https://github.com/sgl-project/sglang/pull/3522
- Status/date: merged / 2025-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "refine deepseek_v3 launch server doc"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `benchmark/deepseek_v3/README.md`; PR body summary: RT.
- Key implementation: `benchmark/deepseek_v3/README.md` modified +7/-0 (7 lines); hunks: -41,6 +41,7 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V...; -82,6 +83,8 @@ python3 -m sglang.launch_server --model-path deepseek-ai/DeepS....
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +7/-0 (7 lines); hunks: -41,6 +41,7 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V...; -82,6 +83,8 @@ python3 -m sglang.launch_server --model-path deepseek-ai/DeepS...
- Key code excerpts:

```diff
diff -- benchmark/deepseek_v3/README.md
@@ -41,6 +41,7 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-r
+<a id="option_args"></a>
@@ -82,6 +83,8 @@ python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 --d
+> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in
@@ -131,6 +134,8 @@ docker run --gpus all \
+> **Note that the launch command here does not enable Data Parallelism Attention or `torch.compile` Optimization**. For optimal performance, please refer to the command options in
@@ -150,6 +155,8 @@ python3 -m sglang.launch_server --model-path /path/to/DeepSeek-V3-BF16 --tp 32 -
```

- Reviewed files:
  - other: `benchmark/deepseek_v3/README.md` modified +7/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #3550 - feat: support flashinfer mla attention for deepseek v3

- Link: https://github.com/sgl-project/sglang/pull/3550
- Status/date: merged / 2025-02-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `70f894b810c0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +296/-132, 737 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support flashinfer mla attention for deepseek v3"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Kudos to @yzh119 Throughout the integration process, we have identified and resolved numerous issues with the exceptional support from the FlashInfer team. Currently, **SGLang i....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-7 (20 lines); hunks: -510,14 +510,20 @@ def forward(; symbols: forward, forward_normal, touching `forward, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-7 (20 lines); hunks: -510,14 +510,20 @@ def forward(; symbols: forward, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -510,14 +510,20 @@ def forward(
-        # Use normal computation for prefill and use weight absorption for extend/decode
-        if (
-            forward_batch.forward_mode.is_extend()
-            and forward_batch.extend_prefix_lens.sum() == 0
-        ):
-            return self.forward_normal(positions, hidden_states, forward_batch)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-7
- Risk and verification: The diff ships test coverage in `test/srt/test_eagle_infer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #3582 - Support NextN (MTP) speculative decoding for DeepSeek-V3/R1

- Link: https://github.com/sgl-project/sglang/pull/3582
- Status/date: merged / 2025-02-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `862dd76c7619`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +437/-7, 529 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support NextN (MTP) speculative decoding for DeepSeek-V3/R1"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: We implemented NextN (MTP) speculative decoding for DeepSeek-V3/R1 based on EAGLE 2 on Triton backend (https://github.com/sgl-project/sglang/pull/3466) and achieved **1.76x** sp....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +4/-1 (5 lines); hunks: -519,6 +519,8 @@ def forward(; -680,6 +682,7 @@ def __init__(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-1 (5 lines); hunks: -519,6 +519,8 @@ def forward(; -680,6 +682,7 @@ def __init__(; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -519,6 +519,8 @@ def forward(
+                and not forward_batch.forward_mode.is_target_verify()
+                and not forward_batch.forward_mode.is_draft_extend()
@@ -680,6 +682,7 @@ def __init__(
+        is_nextn: bool = False,
@@ -731,7 +734,7 @@ def __init__(
-        if (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3785 - Refactor flashinfer logic for deepseek v3 and fix accuracy bug

- Link: https://github.com/sgl-project/sglang/pull/3785
- Status/date: merged / 2025-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `b110084654a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +565/-19, 635 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor flashinfer logic for deepseek v3 and fix accuracy bug"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: `flashinfer_backend.py` for attention is too complex, this PR extract the logic of MLA and creates a new `flashinfer_mla_backend.py` Also, #3716 #3751 reports an accuracy bug wh....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +19/-17 (36 lines); hunks: -510,25 +510,27 @@ def forward(; symbols: forward, no_absorb, forward_normal, touching `forward, no_absorb, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +19/-17 (36 lines); hunks: -510,25 +510,27 @@ def forward(; symbols: forward, no_absorb, forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -510,25 +510,27 @@ def forward(
-        if global_server_args_dict["enable_flashinfer_mla"]:
-            if global_server_args_dict["disable_radix_cache"]:
-                if forward_batch.forward_mode.is_extend():
-                    return self.forward_normal(positions, hidden_states, forward_batch)
-                else:
-                    return self.forward_absorb(positions, hidden_states, forward_batch)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +19/-17
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #3730 - Feature DeepSeek V3/R1 INT8 Quantization (block-wise)

- Link: https://github.com/sgl-project/sglang/pull/3730
- Status/date: merged / 2025-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `1a6e97577acb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1092/-5, 1339 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Feature DeepSeek V3/R1 INT8 Quantization (block-wise)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Support block-wise INT8 quantization for DeepSeek V3/R1. INT8 is a friendly type for most hardware platforms. - Fork FP8 implementation and support the block-wise quantization o....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -47,6 +47,9; -994,6 +997,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-0 (15 lines); hunks: -47,6 +47,9; -994,6 +997,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -47,6 +47,9 @@
+from sglang.srt.layers.quantization.int8_utils import (
+    block_dequant as int8_block_dequant,
+)
@@ -994,6 +997,18 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+                if (
+                    hasattr(self.quant_config, "weight_block_size")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +15/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_block_int8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #3237 - [ROCm] Enable Fused MLA Triton kernel for DeepSeekV3

- Link: https://github.com/sgl-project/sglang/pull/3237
- Status/date: merged / 2025-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `6ce9dbe82882`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +863/-1, 894 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Enable Fused MLA Triton kernel for DeepSeekV3"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This PR introduces the concept of fusing MLA rope into the grouped attention on ROCm. To use this feature use the env variable : SGLANG_ROCM_FUSED_DECODE_MLA=1. Triton Kernel au....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +159/-1 (160 lines); hunks: -16,6 +16,7; -31,6 +32,9; symbols: no_absorb, forward_normal, forward_absorb, forward_absorb_fused_mla_rope, touching `no_absorb, forward_normal, forward_absorb`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +159/-1 (160 lines); hunks: -16,6 +16,7; -31,6 +32,9; symbols: no_absorb, forward_normal, forward_absorb, forward_absorb_fused_mla_rope
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -16,6 +16,7 @@
+import os
@@ -31,6 +32,9 @@
+from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
+    decode_attention_fwd_grouped_rope,
+)
@@ -533,7 +537,18 @@ def no_absorb() -> bool:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +159/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_triton_attention_rocm_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #3893 - add deepgemm and sglang fp8 block-wise gemm benchmark

- Link: https://github.com/sgl-project/sglang/pull/3893
- Status/date: merged / 2025-03-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +320/-0, 322 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add deepgemm and sglang fp8 block-wise gemm benchmark"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md`; PR body summary: The script benchmark all FP8 w8a8 block-wise matrix multiplications involved in DeepSeek V3/R1 under the current tensor parallelism (TP) setting with DeepGemm and vLLM/SGLang. *....
- Key implementation: `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, fp8_gemm_deepgemm, fp8_gemm_sglang, touching `per_token_cast_to_fp8, per_block_cast_to_fp8, fp8_gemm_deepgemm`; `benchmark/kernels/deepseek/README.md` added +6/-0 (6 lines); hunks: -0,0 +1,6.
- Code diff details:
  - `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py` added +314/-0 (314 lines); hunks: -0,0 +1,314; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, fp8_gemm_deepgemm, fp8_gemm_sglang
  - `benchmark/kernels/deepseek/README.md` added +6/-0 (6 lines); hunks: -0,0 +1,6
- Key code excerpts:

```diff
diff -- benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py
@@ -0,0 +1,314 @@
+import itertools
+from typing import Tuple
+import deep_gemm
+import numpy as np
+import torch
+import triton
diff -- benchmark/kernels/deepseek/README.md
@@ -0,0 +1,6 @@
+## DeepSeek kernels benchmark
+- `benchmark_deepgemm_fp8_gemm.py`
+    - You should install [DeepGemm](https://github.com/deepseek-ai/DeepGEMM) from source before run `benchmark_deepgemm_fp8_gemm.py`.
+    - You can use the `--run_correctness` parameter to verify all kernels results's correctness.
+    - You can use the `--tp_size` parameter to benchmark all FP8 w8a8 block-wise matrix multiplications involved in DeepSeek V3/R1 under the current tensor parallelism (TP) settin
```

- Reviewed files:
  - other: `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py` added +314/-0; `benchmark/kernels/deepseek/README.md` added +6/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #3888 - [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)

- Link: https://github.com/sgl-project/sglang/pull/3888
- Status/date: merged / 2025-03-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `c7f254468fca`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +369/-21, 503 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Support channel-wise INT8 quantization for DeepSeek V3/R1. INT8 is a friendly type for most hardware platforms. Co-author: @yych0745 @sleepcoo @b0urnee - Moe: Fused moe triton k....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +16/-12 (28 lines); hunks: -1202,18 +1202,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +16/-12 (28 lines); hunks: -1202,18 +1202,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1202,18 +1202,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-                if (
-                    hasattr(self.quant_config, "weight_block_size")
-                    and w.dtype == torch.int8
-                ):
-                    weight_block_size = self.quant_config.weight_block_size
-                    if weight_block_size is not None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +16/-12
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_int8_kernel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4218 - Support nextn for flashinfer mla attention backend

- Link: https://github.com/sgl-project/sglang/pull/4218
- Status/date: merged / 2025-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +393/-58, 648 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support nextn for flashinfer mla attention backend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/srt/test_mla_flashinfer.py`; PR body summary: Support the compatibility of nextn and flashinfer mla attention backend. Currently topk can only be set to 1 due to lack of custom mask support for flashinfer MLA wrapper. - Imp....
- Key implementation: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +317/-57 (374 lines); hunks: -11,9 +11,10; -23,6 +24,7; symbols: FlashInferMLAAttnBackend, __init__, init_forward_metadata, touching `FlashInferMLAAttnBackend, __init__, init_forward_metadata`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -555,6 +555,8 @@ def no_absorb() -> bool:; symbols: no_absorb, touching `no_absorb`; `test/srt/test_mla_flashinfer.py` modified +63/-0 (63 lines); hunks: -1,6 +1,7; -100,5 +101,67 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestFlashinferMLAMTP, setUpClass, tearDownClass, touching `test_gsm8k, TestFlashinferMLAMTP, setUpClass`; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -123,6 +123,16 @@ def init_attention_backend(self):; symbols: init_attention_backend, touching `init_attention_backend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +317/-57 (374 lines); hunks: -11,9 +11,10; -23,6 +24,7; symbols: FlashInferMLAAttnBackend, __init__, init_forward_metadata
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -555,6 +555,8 @@ def no_absorb() -> bool:; symbols: no_absorb
  - `test/srt/test_mla_flashinfer.py` modified +63/-0 (63 lines); hunks: -1,6 +1,7; -100,5 +101,67 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestFlashinferMLAMTP, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: -123,6 +123,16 @@ def init_attention_backend(self):; symbols: init_attention_backend
  - `docs/references/deepseek.md` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ Please refer to [the example](https://github.com/sgl-project/s...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -11,9 +11,10 @@
-from typing import TYPE_CHECKING, Optional, Union
+from typing import TYPE_CHECKING, Callable, Optional, Union
+import triton
@@ -23,6 +24,7 @@
+from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
@@ -58,12 +60,16 @@ class FlashInferMLAAttnBackend(AttentionBackend):
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -555,6 +555,8 @@ def no_absorb() -> bool:
+                    and not forward_batch.forward_mode.is_target_verify()
+                    and not forward_batch.forward_mode.is_draft_extend()
diff -- test/srt/test_mla_flashinfer.py
@@ -1,6 +1,7 @@
+import requests
@@ -100,5 +101,67 @@ def test_gsm8k(self):
+class TestFlashinferMLAMTP(unittest.TestCase):
+    @classmethod
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +317/-57; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0; `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0
  - tests: `test/srt/test_mla_flashinfer.py` modified +63/-0
  - docs: `docs/references/deepseek.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_mla_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4165 - DeepGemm integrate to sgl-kernel

- Link: https://github.com/sgl-project/sglang/pull/4165
- Status/date: merged / 2025-03-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +324/-5, 388 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepGemm integrate to sgl-kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh`; PR body summary: Integrate DeepGemm in setup. Linear usage: #4199 ..
- Key implementation: `sgl-kernel/tests/test_deep_gemm.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, construct, construct_grouped, touching `per_token_cast_to_fp8, per_block_cast_to_fp8, construct`; `sgl-kernel/setup.py` modified +52/-1 (53 lines); hunks: -14,11 +14,13; -52,6 +54,7 @@ def _get_version():; symbols: _get_version, CustomBuildPy, run, make_jit_include_symlinks, touching `_get_version, CustomBuildPy, run`; `sgl-kernel/build.sh` modified +4/-3 (7 lines); hunks: -11,11 +11,11 @@ else; -24,5 +24,6 @@ docker run --rm \; `.gitmodules` modified +3/-0 (3 lines); hunks: -7,3 +7,6.
- Code diff details:
  - `sgl-kernel/tests/test_deep_gemm.py` added +263/-0 (263 lines); hunks: -0,0 +1,263; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, construct, construct_grouped
  - `sgl-kernel/setup.py` modified +52/-1 (53 lines); hunks: -14,11 +14,13; -52,6 +54,7 @@ def _get_version():; symbols: _get_version, CustomBuildPy, run, make_jit_include_symlinks
  - `sgl-kernel/build.sh` modified +4/-3 (7 lines); hunks: -11,11 +11,11 @@ else; -24,5 +24,6 @@ docker run --rm \
  - `.gitmodules` modified +3/-0 (3 lines); hunks: -7,3 +7,6
  - `sgl-kernel/pyproject.toml` modified +1/-1 (2 lines); hunks: -1,6 +1,6
- Key code excerpts:

```diff
diff -- sgl-kernel/tests/test_deep_gemm.py
@@ -0,0 +1,263 @@
+import os
+import random
+import unittest
+from typing import Any, Tuple
+import deep_gemm
+import torch
diff -- sgl-kernel/setup.py
@@ -14,11 +14,13 @@
+import shutil
+from setuptools.command.build_py import build_py
@@ -52,6 +54,7 @@ def _get_version():
+deepgemm = root / "3rdparty" / "deepgemm"
@@ -63,6 +66,51 @@ def _get_version():
+class CustomBuildPy(build_py):
diff -- sgl-kernel/build.sh
@@ -11,11 +11,11 @@ else
```

- Reviewed files:
  - tests: `sgl-kernel/tests/test_deep_gemm.py` added +263/-0
  - other: `sgl-kernel/setup.py` modified +52/-1; `sgl-kernel/build.sh` modified +4/-3; `.gitmodules` modified +3/-0; `sgl-kernel/pyproject.toml` modified +1/-1; `sgl-kernel/3rdparty/deepgemm` added +1/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_deep_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4199 - linear support deepgemm

- Link: https://github.com/sgl-project/sglang/pull/4199
- Status/date: merged / 2025-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +76/-44, 184 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "linear support deepgemm"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py`; PR body summary: Use deepgemm linear gemm, dependent on https://github.com/sgl-project/sglang/pull/4165 Test case **MMLU Average accuracy: 0.878** performence test size | origin | deepgemm linea....
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +36/-28 (64 lines); hunks: -29,10 +29,13; -722,34 +725,39 @@ def grid(META):; symbols: _per_token_group_quant_fp8, grid, touching `_per_token_group_quant_fp8, grid`; `python/sglang/test/test_block_fp8.py` modified +39/-15 (54 lines); hunks: -1,4 +1,5; -11,6 +12,8; symbols: native_per_token_group_quant_fp8, native_w8a8_block_fp8_matmul, TestW8A8BlockFP8Matmul, setUpClass, touching `native_per_token_group_quant_fp8, native_w8a8_block_fp8_matmul, TestW8A8BlockFP8Matmul`; `test/srt/test_fp8_kernel.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ def setUpClass(cls):; symbols: setUpClass, _make_A, touching `setUpClass, _make_A`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +36/-28 (64 lines); hunks: -29,10 +29,13; -722,34 +725,39 @@ def grid(META):; symbols: _per_token_group_quant_fp8, grid
  - `python/sglang/test/test_block_fp8.py` modified +39/-15 (54 lines); hunks: -1,4 +1,5; -11,6 +12,8; symbols: native_per_token_group_quant_fp8, native_w8a8_block_fp8_matmul, TestW8A8BlockFP8Matmul, setUpClass
  - `test/srt/test_fp8_kernel.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ def setUpClass(cls):; symbols: setUpClass, _make_A
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -29,10 +29,13 @@
+    import deep_gemm
+_enable_jit_deepgemm = int(os.getenv("SGL_ENABLE_JIT_DEEPGEMM", "0"))
@@ -722,34 +725,39 @@ def grid(META):
-    kernel = (
-        _w8a8_block_fp8_matmul_unrolledx4
-        if (is_hip_ == True and num_workgroups <= get_device_core_count())
diff -- python/sglang/test/test_block_fp8.py
@@ -1,4 +1,5 @@
+import os
@@ -11,6 +12,8 @@
+_is_cuda = torch.cuda.is_available() and torch.version.cuda
@@ -208,21 +211,44 @@ def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.fl
-    OUT_DTYPES = [torch.float32, torch.half, torch.bfloat16]
-    M = [1, 7, 83, 512, 2048]
diff -- test/srt/test_fp8_kernel.py
@@ -17,7 +17,7 @@ def setUpClass(cls):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +36/-28
  - tests: `python/sglang/test/test_block_fp8.py` modified +39/-15; `test/srt/test_fp8_kernel.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4079 - add INT8 example into dsv3 README

- Link: https://github.com/sgl-project/sglang/pull/4079
- Status/date: merged / 2025-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-2, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add INT8 example into dsv3 README"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `benchmark/deepseek_v3/README.md`; PR body summary: add INT8 example into benchmark/deepseek_v3/README.md.
- Key implementation: `benchmark/deepseek_v3/README.md` modified +16/-2 (18 lines); hunks: -184,25 +184,39 @@ AWQ does not support BF16, so add the `--dtype half` flag....
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +16/-2 (18 lines); hunks: -184,25 +184,39 @@ AWQ does not support BF16, so add the `--dtype half` flag...
- Key code excerpts:

```diff
diff -- benchmark/deepseek_v3/README.md
@@ -184,25 +184,39 @@ AWQ does not support BF16, so add the `--dtype half` flag if AWQ is used for qua
+Assuming that master node IP is `MASTER_IP`, checkpoint path is `/path/to/DeepSeek-R1-INT8` and port=5000, we can have following commands to launch the server:
-	HEAD_IP:5000 --nnodes 2 --node-rank 0 --trust-remote --enable-torch-compile --torch-compile-max-bs 8
+	MASTER_IP:5000 --nnodes 2 --node-rank 0 --trust-remote --enable-torch-compile --torch-compile-max-bs 8
-	HEAD_IP:5000 --nnodes 2 --node-rank 1 --trust-remote --enable-torch-compile --torch-compile-max-bs 8
+	MASTER_IP:5000 --nnodes 2 --node-rank 1 --trust-remote --enable-torch-compile --torch-compile-max-bs 8
+> **Note that the launch command here enables `torch.compile` Optimization**. For optimal performance, please refer to the command options in [Performance Optimization Options](#o
```

- Reviewed files:
  - other: `benchmark/deepseek_v3/README.md` modified +16/-2
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #4472 - Support FlashMLA backend

- Link: https://github.com/sgl-project/sglang/pull/4472
- Status/date: merged / 2025-03-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +209/-1, 285 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support FlashMLA backend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py`; PR body summary: Integrate flashmla for decoding, and the accuracy test is currently okay. The current implementation is quite simple, directly integrating flashmla as the backend. Later, we nee....
- Key implementation: `python/sglang/srt/layers/attention/flashmla_backend.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: FlashMLABackend, __init__, forward_decode, touching `FlashMLABackend, __init__, forward_decode`; `python/sglang/srt/layers/attention/utils.py` modified +54/-0 (54 lines); hunks: -15,6 +15,7 @@ def create_flashinfer_kv_indices_triton(; -37,3 +38,56 @@ def create_flashinfer_kv_indices_triton(; symbols: create_flashinfer_kv_indices_triton, create_flashmla_kv_indices_triton, touching `create_flashinfer_kv_indices_triton, create_flashmla_kv_indices_triton`; `python/sglang/srt/model_executor/model_runner.py` modified +8/-0 (8 lines); hunks: -149,6 +149,7 @@ def __init__(; -223,6 +224,9 @@ def model_specific_adjustment(self):; symbols: __init__, model_specific_adjustment, init_attention_backend, touching `__init__, model_specific_adjustment, init_attention_backend`; `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -173,6 +173,7 @@ class ServerArgs:; -227,6 +228,8 @@ def __post_init__(self):; symbols: ServerArgs, __post_init__, add_cli_args, touching `ServerArgs, __post_init__, add_cli_args`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` added +128/-0 (128 lines); hunks: -0,0 +1,128; symbols: FlashMLABackend, __init__, forward_decode
  - `python/sglang/srt/layers/attention/utils.py` modified +54/-0 (54 lines); hunks: -15,6 +15,7 @@ def create_flashinfer_kv_indices_triton(; -37,3 +38,56 @@ def create_flashinfer_kv_indices_triton(; symbols: create_flashinfer_kv_indices_triton, create_flashmla_kv_indices_triton
  - `python/sglang/srt/model_executor/model_runner.py` modified +8/-0 (8 lines); hunks: -149,6 +149,7 @@ def __init__(; -223,6 +224,9 @@ def model_specific_adjustment(self):; symbols: __init__, model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -173,6 +173,7 @@ class ServerArgs:; -227,6 +228,8 @@ def __post_init__(self):; symbols: ServerArgs, __post_init__, add_cli_args
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: -71,6 +71,7; -1273,7 +1274,10 @@ def merge_batch(self, other: "ScheduleBatch"):; symbols: merge_batch, get_model_worker_batch
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashmla_backend.py
@@ -0,0 +1,128 @@
+from __future__ import annotations
+"""
+Support attention backend for flashMLA.
+Current initial integration of FlashMLA shows normal accuracy, but performance is slightly lacking.
+#TODO
+Support FlashMLA decode with cudagraph
diff -- python/sglang/srt/layers/attention/utils.py
@@ -15,6 +15,7 @@ def create_flashinfer_kv_indices_triton(
+    # find the req pool idx, this is for batch to token
@@ -37,3 +38,56 @@ def create_flashinfer_kv_indices_triton(
+@triton.jit
+def create_flashmla_kv_indices_triton(
+    req_to_token_ptr,  # [max_batch, max_context_len]
+    req_pool_indices_ptr,
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -149,6 +149,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashmla_backend.py` added +128/-0; `python/sglang/srt/layers/attention/utils.py` modified +54/-0; `python/sglang/srt/model_executor/model_runner.py` modified +8/-0; `python/sglang/srt/server_args.py` modified +8/-0; `python/sglang/srt/managers/schedule_batch.py` modified +5/-1
  - other: `scripts/playground/bench_speculative.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/managers/schedule_batch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4514 - Support FlashMLA backend cuda graph

- Link: https://github.com/sgl-project/sglang/pull/4514
- Status/date: merged / 2025-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +188/-32, 286 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support FlashMLA backend cuda graph"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/server_args.py`; PR body summary: Support FlashMLA backend cuda graph. Optimize index calculation, complete the calculation in init_forward * Optimize the FlashInfer block table calculation logic to compute only....
- Key implementation: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +184/-30 (214 lines); hunks: -1,16 +1,13; -28,10 +25,30; symbols: FlashMLADecodeMetadata, __init__, FlashMLABackend, touching `FlashMLADecodeMetadata, __init__, FlashMLABackend`; `python/sglang/srt/layers/attention/utils.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton, touching `create_flashmla_kv_indices_triton`; `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunks: -232,7 +232,10 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +184/-30 (214 lines); hunks: -1,16 +1,13; -28,10 +25,30; symbols: FlashMLADecodeMetadata, __init__, FlashMLABackend
  - `python/sglang/srt/layers/attention/utils.py` modified +0/-1 (1 lines); hunks: -49,7 +49,6 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton
  - `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunks: -232,7 +232,10 @@ def __post_init__(self):; symbols: __post_init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashmla_backend.py
@@ -1,16 +1,13 @@
-Support attention backend for flashMLA.
+Support attention backend for FlashMLA.
-Current initial integration of FlashMLA shows normal accuracy, but performance is slightly lacking.
-Support FlashMLA decode with cudagraph
-Integrate FA3 prefill
+from dataclasses import dataclass
diff -- python/sglang/srt/layers/attention/utils.py
@@ -49,7 +49,6 @@ def create_flashmla_kv_indices_triton(
-    max_pagesize: tl.constexpr,
diff -- python/sglang/srt/server_args.py
@@ -232,7 +232,10 @@ def __post_init__(self):
-            assert self.page_size == 64, "FlashMLA only support page_size=64"
+            logger.warning(
+                "FlashMLA only supports a page_size of 64, change page_size to 64."
+            )
+            self.page_size = 64
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +184/-30; `python/sglang/srt/layers/attention/utils.py` modified +0/-1; `python/sglang/srt/server_args.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4613 - Set deepgemm to the default value in the hopper architecture.

- Link: https://github.com/sgl-project/sglang/pull/4613
- Status/date: merged / 2025-03-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-3, 53 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Set deepgemm to the default value in the hopper architecture."; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; PR body summary: Set deepgemm to the default value in the hopper architecture..
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-3 (12 lines); hunks: -26,11 +26,14; -39,9 +42,12; symbols: grid, touching `grid`; `python/sglang/srt/utils.py` modified +7/-0 (7 lines); hunks: -1006,6 +1006,13 @@ def get_amdgpu_memory_capacity():; symbols: get_amdgpu_memory_capacity, get_device_sm, get_nvgpu_memory_capacity, touching `get_amdgpu_memory_capacity, get_device_sm, get_nvgpu_memory_capacity`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-3 (12 lines); hunks: -26,11 +26,14; -39,9 +42,12; symbols: grid
  - `python/sglang/srt/utils.py` modified +7/-0 (7 lines); hunks: -1006,6 +1006,13 @@ def get_amdgpu_memory_capacity():; symbols: get_amdgpu_memory_capacity, get_device_sm, get_nvgpu_memory_capacity
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -26,11 +26,14 @@
+    get_device_sm,
+_enable_jit_deepgemm = False
@@ -39,9 +42,12 @@
-logger = logging.getLogger(__name__)
+    sm_version = get_device_sm()
+    if sm_version >= 90 and int(os.getenv("SGL_ENABLE_JIT_DEEPGEMM", "1")):
diff -- python/sglang/srt/utils.py
@@ -1006,6 +1006,13 @@ def get_amdgpu_memory_capacity():
+def get_device_sm():
+    if torch.cuda.is_available():
+        major, minor = torch.cuda.get_device_capability()
+        return major * 10 + minor
+    return 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-3; `python/sglang/srt/utils.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4631 - [ROCm] Enable MTP (NextN) on AMD GPU

- Link: https://github.com/sgl-project/sglang/pull/4631
- Status/date: merged / 2025-03-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +43/-4, 99 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Enable MTP (NextN) on AMD GPU"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py`; PR body summary: To enable MTP (NextN) on AMD GPU. Added a few kernels to `torch_extension_rocm.cc`. Enabled MTP test for AMD GPU. Benchmark result Correctness check.
- Key implementation: `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h` added +20/-0 (20 lines); hunks: -0,0 +1,20; `sgl-kernel/csrc/torch_extension_rocm.cc` modified +12/-0 (12 lines); hunks: -65,6 +65,18 @@ TORCH_LIBRARY_EXPAND(sgl_kernel, m) {; `python/sglang/srt/speculative/build_eagle_tree.py` modified +2/-2 (4 lines); hunks: -4,9 +4,9; `python/sglang/srt/speculative/eagle_utils.py` modified +3/-1 (4 lines); hunks: -14,7 +14,7; -23,6 +23,8.
- Code diff details:
  - `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h` added +20/-0 (20 lines); hunks: -0,0 +1,20
  - `sgl-kernel/csrc/torch_extension_rocm.cc` modified +12/-0 (12 lines); hunks: -65,6 +65,18 @@ TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  - `python/sglang/srt/speculative/build_eagle_tree.py` modified +2/-2 (4 lines); hunks: -4,9 +4,9
  - `python/sglang/srt/speculative/eagle_utils.py` modified +3/-1 (4 lines); hunks: -14,7 +14,7; -23,6 +23,8
  - `sgl-kernel/csrc/speculative/eagle_utils.cu` modified +4/-0 (4 lines); hunks: -17,7 +17,11
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h
@@ -0,0 +1,20 @@
+#include <torch/library.h>
+#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
+#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
+#define CHECK_LAST_DIM_CONTIGUOUS(x) \
+  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")
+#define CHECK_INPUT(x) \
diff -- sgl-kernel/csrc/torch_extension_rocm.cc
@@ -65,6 +65,18 @@ TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
+  m.def(
+      "verify_tree_greedy(Tensor! predicts, Tensor! accept_index, Tensor! accept_token_num, "
+      "Tensor candidates, Tensor retrive_index, Tensor retrive_next_token, Tensor retrive_next_sibling, "
+      "Tensor target_predict, int cuda_stream) -> ()");
+  m.impl("verify_tree_greedy", torch::kCUDA, &verify_tree_greedy);
+  m.def(
diff -- python/sglang/srt/speculative/build_eagle_tree.py
@@ -4,9 +4,9 @@
```

- Reviewed files:
  - other: `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h` added +20/-0; `sgl-kernel/csrc/torch_extension_rocm.cc` modified +12/-0; `sgl-kernel/csrc/speculative/eagle_utils.cu` modified +4/-0; `sgl-kernel/setup_rocm.py` modified +1/-0
  - runtime: `python/sglang/srt/speculative/build_eagle_tree.py` modified +2/-2; `python/sglang/srt/speculative/eagle_utils.py` modified +3/-1
  - tests: `test/srt/test_mla_deepseek_v3.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_mla_deepseek_v3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4831 - [Feature] Support FA3 backend for MLA

- Link: https://github.com/sgl-project/sglang/pull/4831
- Status/date: merged / 2025-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +180/-74, 352 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Support FA3 backend for MLA"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Support FlashAttention 3 backend for MLA. FA3 official example of MLA can be found in this file..
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +171/-73 (244 lines); hunks: -13,7 +13,9; -58,6 +60,9 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_extend, forward_decode, touching `__init__, init_forward_metadata, forward_extend`; `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunks: -230,6 +230,10 @@ def model_specific_adjustment(self):; -879,7 +883,7 @@ def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend, touching `model_specific_adjustment, init_attention_backend`; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -655,6 +655,7 @@ def __init__(; -667,6 +668,9 @@ def no_absorb(self, forward_batch: ForwardBatch) -> bool:; symbols: __init__, no_absorb, touching `__init__, no_absorb`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +171/-73 (244 lines); hunks: -13,7 +13,9; -58,6 +60,9 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_extend, forward_decode
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunks: -230,6 +230,10 @@ def model_specific_adjustment(self):; -879,7 +883,7 @@ def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: -655,6 +655,7 @@ def __init__(; -667,6 +668,9 @@ def no_absorb(self, forward_batch: ForwardBatch) -> bool:; symbols: __init__, no_absorb
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -13,7 +13,9 @@
+from sglang.srt.configs.model_config import AttentionArch
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -58,6 +60,9 @@ def __init__(
+        self.use_mla = (
+            model_runner.model_config.attention_arch == AttentionArch.MLA
+        ) and (not global_server_args_dict["disable_mla"])
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -230,6 +230,10 @@ def model_specific_adjustment(self):
+                elif server_args.attention_backend == "fa3":
+                    logger.info(
+                        f"MLA optimization is turned on. Use flash attention 3 backend."
+                    )
@@ -879,7 +883,7 @@ def init_attention_backend(self):
-                "FlashAttention v3 Backend is in Beta. Multimodal, Page > 1, FP8, MLA and Speculative Decoding are not supported."
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -655,6 +655,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +171/-73; `python/sglang/srt/model_executor/model_runner.py` modified +5/-1; `python/sglang/srt/models/deepseek_v2.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4530 - Add deepseek style fused moe group gate selection kernel

- Link: https://github.com/sgl-project/sglang/pull/4530
- Status/date: merged / 2025-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +616/-1, 659 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add deepseek style fused moe group gate selection kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py`; PR body summary: PR adapted and improved from #3191 Rewrite Macro. Extended to support all power of 2 `# expert` & `# expert group`, also all `# topk_group` & `# topk` use cases + dtype support....
- Key implementation: `sgl-kernel/csrc/moe/moe_fused_gate.cu` added +447/-0 (447 lines); hunks: -0,0 +1,447; `sgl-kernel/benchmark/bench_moe_fused_gate.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: biased_grouped_topk_org, biased_grouped_topk_org_kernel, benchmark, touching `biased_grouped_topk_org, biased_grouped_topk_org_kernel, benchmark`; `sgl-kernel/tests/test_moe_fused_gate.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: test_moe_fused_gate_combined, touching `test_moe_fused_gate_combined`; `sgl-kernel/python/sgl_kernel/moe.py` modified +12/-0 (12 lines); hunks: -32,3 +32,15 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate, touching `topk_softmax, moe_fused_gate`.
- Code diff details:
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` added +447/-0 (447 lines); hunks: -0,0 +1,447
  - `sgl-kernel/benchmark/bench_moe_fused_gate.py` added +74/-0 (74 lines); hunks: -0,0 +1,74; symbols: biased_grouped_topk_org, biased_grouped_topk_org_kernel, benchmark
  - `sgl-kernel/tests/test_moe_fused_gate.py` added +72/-0 (72 lines); hunks: -0,0 +1,72; symbols: test_moe_fused_gate_combined
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +12/-0 (12 lines); hunks: -32,3 +32,15 @@ def topk_softmax(; symbols: topk_softmax, moe_fused_gate
  - `sgl-kernel/csrc/torch_extension.cc` modified +5/-0 (5 lines); hunks: -138,6 +138,11 @@ TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/moe_fused_gate.cu
@@ -0,0 +1,447 @@
+#include <ATen/cuda/CUDAContext.h>
+#include <cuda_runtime.h>
+#include <cutlass/array.h>
+#include <cutlass/cutlass.h>
+#include <cutlass/numeric_types.h>
+#include <stdio.h>
diff -- sgl-kernel/benchmark/bench_moe_fused_gate.py
@@ -0,0 +1,74 @@
+import itertools
+import math
+import torch
+import triton
+import triton.language as tl
+from sgl_kernel import moe_fused_gate
diff -- sgl-kernel/tests/test_moe_fused_gate.py
@@ -0,0 +1,72 @@
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/moe_fused_gate.cu` added +447/-0; `sgl-kernel/benchmark/bench_moe_fused_gate.py` added +74/-0; `sgl-kernel/python/sgl_kernel/moe.py` modified +12/-0; `sgl-kernel/csrc/torch_extension.cc` modified +5/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +3/-0; `sgl-kernel/python/sgl_kernel/__init__.py` modified +1/-1
  - tests: `sgl-kernel/tests/test_moe_fused_gate.py` added +72/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_moe_fused_gate.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4817 - [sgl-kernel] per token group quant support COLUMN MAJOR

- Link: https://github.com/sgl-project/sglang/pull/4817
- Status/date: merged / 2025-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +252/-80, 489 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[sgl-kernel] per token group quant support COLUMN MAJOR"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py`; no usable PR-body summary.
- Key implementation: `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +196/-58 (254 lines); hunks: -9,12 +9,12; -25,15 +25,16 @@ def _per_token_group_quant_8bit(; symbols: _per_token_group_quant_8bit, _per_token_group_quant_fp8, _per_token_group_quant_fp8_colmajor, touching `_per_token_group_quant_8bit, _per_token_group_quant_fp8, _per_token_group_quant_fp8_colmajor`; `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +49/-19 (68 lines); hunks: -16,7 +16,7 @@ __device__ __forceinline__ float GroupReduceMax(float val, con...; -26,19 +26,30 @@ __global__ void per_token_group_quant_8bit_kernel(; `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py` modified +7/-3 (10 lines); hunks: -148,9 +148,11 @@ def sglang_per_token_group_quant_8bit(; -196,7 +198,9 @@ def benchmark(batch_size, seq_len, group_size, dst_dtype, pr...; symbols: sglang_per_token_group_quant_8bit, calculate_diff, benchmark, touching `sglang_per_token_group_quant_8bit, calculate_diff, benchmark`.
- Code diff details:
  - `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +196/-58 (254 lines); hunks: -9,12 +9,12; -25,15 +25,16 @@ def _per_token_group_quant_8bit(; symbols: _per_token_group_quant_8bit, _per_token_group_quant_fp8, _per_token_group_quant_fp8_colmajor
  - `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +49/-19 (68 lines); hunks: -16,7 +16,7 @@ __device__ __forceinline__ float GroupReduceMax(float val, con...; -26,19 +26,30 @@ __global__ void per_token_group_quant_8bit_kernel(
  - `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py` modified +7/-3 (10 lines); hunks: -148,9 +148,11 @@ def sglang_per_token_group_quant_8bit(; -196,7 +198,9 @@ def benchmark(batch_size, seq_len, group_size, dst_dtype, pr...; symbols: sglang_per_token_group_quant_8bit, calculate_diff, benchmark
- Key code excerpts:

```diff
diff -- sgl-kernel/tests/test_per_token_group_quant_8bit.py
@@ -9,12 +9,12 @@
-is_hip_ = is_hip()
-fp8_type_ = torch.float8_e4m3fnuz if is_hip_ else torch.float8_e4m3fn
+_is_hip = is_hip()
+fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
-def _per_token_group_quant_8bit(
+def _per_token_group_quant_fp8(
diff -- sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu
@@ -16,7 +16,7 @@ __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
-template <typename T, typename DST_DTYPE>
+template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false>
@@ -26,19 +26,30 @@ __global__ void per_token_group_quant_8bit_kernel(
-    const float max_8bit) {
+    const float max_8bit,
+    const int scale_num_rows = 0,
diff -- sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py
@@ -148,9 +148,11 @@ def sglang_per_token_group_quant_8bit(
```

- Reviewed files:
  - tests: `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +196/-58
  - other: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +49/-19; `sgl-kernel/benchmark/bench_per_token_group_quant_8bit.py` modified +7/-3
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_per_token_group_quant_8bit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #4918 - Add DeepSeek V3/R1 shared experts fusion

- Link: https://github.com/sgl-project/sglang/pull/4918
- Status/date: merged / 2025-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `924ca7c92c86`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +536/-36, 939 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek V3/R1 shared experts fusion"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: I gain idea mainly from https://github.com/vllm-project/vllm/pull/15502 , thanks for the author's work.I will add references in the modifications for `grouped_topk` function and....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +81/-5 (86 lines); hunks: -16,12 +16,14; -87,6 +89,8; symbols: DeepseekV2MLP, __init__, touching `DeepseekV2MLP, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +81/-5 (86 lines); hunks: -16,12 +16,14; -87,6 +89,8; symbols: DeepseekV2MLP, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -16,12 +16,14 @@
+import logging
+from tqdm import tqdm
@@ -87,6 +89,8 @@
+logger = logging.getLogger(__name__)
@@ -168,6 +172,12 @@ def __init__(
+        self.n_share_experts_fusion = (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +81/-5
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_moe_align.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5190 - Fix DeepSeek error when using DeepEP mode

- Link: https://github.com/sgl-project/sglang/pull/5190
- Status/date: merged / 2025-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-6, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeek error when using DeepEP mode"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: command output: it does not error, and gsm8k is 0.936.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +8/-6 (14 lines); hunks: -280,10 +280,7 @@ def forward(; -313,8 +310,7 @@ def forward_deepep(; symbols: forward, forward_normal, forward_deepep, _forward_shared_experts, touching `forward, forward_normal, forward_deepep`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-6 (14 lines); hunks: -280,10 +280,7 @@ def forward(; -313,8 +310,7 @@ def forward_deepep(; symbols: forward, forward_normal, forward_deepep, _forward_shared_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -280,10 +280,7 @@ def forward(
-        if self.n_shared_experts is not None and self.n_share_experts_fusion == 0:
-            shared_output = self.shared_experts(hidden_states)
-        else:
-            shared_output = None
+        shared_output = self._forward_shared_experts(hidden_states)
@@ -313,8 +310,7 @@ def forward_deepep(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +8/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5086 - reduce moe_align_block_size_kernel small batch mode overhead

- Link: https://github.com/sgl-project/sglang/pull/5086
- Status/date: merged / 2025-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +143/-56, 263 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "reduce moe_align_block_size_kernel small batch mode overhead"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`; PR body summary: Acc test I set `token_cnts_buffer` and `cumsum_buffer` to `torch.empty` in `fused_moe.py`: Acc result: Kernel unit-test !图片 Benchmark In H200 main branch: pr:.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-1 (2 lines); hunks: -702,7 +702,7 @@ def moe_align_block_size(; symbols: moe_align_block_size, touching `moe_align_block_size`; `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +111/-44 (155 lines); hunks: -64,10 +64,10 @@ __global__ void moe_align_block_size_kernel(; -98,6 +98,65 @@ __global__ void moe_align_block_size_kernel(; `sgl-kernel/benchmark/bench_moe_align_block_size.py` modified +31/-10 (41 lines); hunks: -241,9 +241,9 @@ def calculate_diff(num_tokens, num_experts=256, block_size=1...; -294,17 +294,28 @@ def benchmark(num_tokens, num_experts, topk, provider):; symbols: calculate_diff, benchmark, sgl_moe_align_block_size_with_empty, touching `calculate_diff, benchmark, sgl_moe_align_block_size_with_empty`; `sgl-kernel/tests/test_moe_align.py` modified +0/-1 (1 lines); hunks: -151,7 +151,6 @@ def moe_align_block_size_triton(; symbols: moe_align_block_size_triton, test_moe_align_block_size_compare_implementations, touching `moe_align_block_size_triton, test_moe_align_block_size_compare_implementations`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-1 (2 lines); hunks: -702,7 +702,7 @@ def moe_align_block_size(; symbols: moe_align_block_size
  - `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +111/-44 (155 lines); hunks: -64,10 +64,10 @@ __global__ void moe_align_block_size_kernel(; -98,6 +98,65 @@ __global__ void moe_align_block_size_kernel(
  - `sgl-kernel/benchmark/bench_moe_align_block_size.py` modified +31/-10 (41 lines); hunks: -241,9 +241,9 @@ def calculate_diff(num_tokens, num_experts=256, block_size=1...; -294,17 +294,28 @@ def benchmark(num_tokens, num_experts, topk, provider):; symbols: calculate_diff, benchmark, sgl_moe_align_block_size_with_empty
  - `sgl-kernel/tests/test_moe_align.py` modified +0/-1 (1 lines); hunks: -151,7 +151,6 @@ def moe_align_block_size_triton(; symbols: moe_align_block_size_triton, test_moe_align_block_size_compare_implementations
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -702,7 +702,7 @@ def moe_align_block_size(
-        token_cnts_buffer = torch.zeros(
+        token_cnts_buffer = torch.empty(
diff -- sgl-kernel/csrc/moe/moe_align_kernel.cu
@@ -64,10 +64,10 @@ __global__ void moe_align_block_size_kernel(
-  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
-  const size_t start_idx = threadIdx.x * tokens_per_thread;
+  const size_t tid = threadIdx.x;
+  const size_t stride = blockDim.x;
-  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
+  for (size_t i = tid; i < numel; i += stride) {
diff -- sgl-kernel/benchmark/bench_moe_align_block_size.py
@@ -241,9 +241,9 @@ def calculate_diff(num_tokens, num_experts=256, block_size=128, topk=8):
-num_tokens_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
+num_tokens_range = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
-topk_range = [2, 4, 8]
+topk_range = [1, 2, 4, 8]
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-1
  - other: `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +111/-44; `sgl-kernel/benchmark/bench_moe_align_block_size.py` modified +31/-10
  - tests: `sgl-kernel/tests/test_moe_align.py` modified +0/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_moe_align.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5310 - fix: use deepgemm only on hopper

- Link: https://github.com/sgl-project/sglang/pull/5310
- Status/date: merged / 2025-04-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: use deepgemm only on hopper"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +1/-1 (2 lines); hunks: -45,7 +45,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -45,7 +45,7 @@
-    if sm_version >= 90 and get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true"):
+    if sm_version == 90 and get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true"):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5210 - feat: use fa3 mla by default on hopper

- Link: https://github.com/sgl-project/sglang/pull/5210
- Status/date: merged / 2025-04-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +42/-11, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: use fa3 mla by default on hopper"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py`; PR body summary: - Support DP MLA for FA3.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunks: -80,6 +80,7; -245,7 +246,16 @@ def model_specific_adjustment(self):; symbols: model_specific_adjustment, init_attention_backend, touching `model_specific_adjustment, init_attention_backend`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +12/-7 (19 lines); hunks: -325,7 +325,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -527,7 +527,9 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, forward_decode, init_forward_metadata_capture_cuda_graph, touching `init_forward_metadata, forward_extend, forward_decode`; `python/sglang/srt/utils.py` modified +9/-0 (9 lines); hunks: -1828,3 +1828,12 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, is_hopper_with_cuda_12_3, touching `fast_topk, is_hopper_with_cuda_12_3`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunks: -80,6 +80,7; -245,7 +246,16 @@ def model_specific_adjustment(self):; symbols: model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +12/-7 (19 lines); hunks: -325,7 +325,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -527,7 +527,9 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, forward_decode, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/utils.py` modified +9/-0 (9 lines); hunks: -1828,3 +1828,12 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, is_hopper_with_cuda_12_3
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -80,6 +80,7 @@
+    is_hopper_with_cuda_12_3,
@@ -245,7 +246,16 @@ def model_specific_adjustment(self):
-                server_args.attention_backend = "triton"
+                if is_hopper_with_cuda_12_3():
+                    if server_args.speculative_eagle_topk is None or (
+                        server_args.speculative_eagle_topk is not None
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -325,7 +325,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-        if forward_batch.forward_mode.is_decode():
+        if forward_batch.forward_mode.is_decode_or_idle():
@@ -527,7 +527,9 @@ def forward_extend(
-        if self.kv_cache_dtype_str != "auto":
+        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
+        # has corresponding quantization method so that layer.k_scale is not None
diff -- python/sglang/srt/utils.py
@@ -1828,3 +1828,12 @@ def fast_topk(values, topk, dim):
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +21/-4; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +12/-7; `python/sglang/srt/utils.py` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5385 - Apply deepseek cuda rope

- Link: https://github.com/sgl-project/sglang/pull/5385
- Status/date: merged / 2025-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-1, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply deepseek cuda rope"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/rotary_embedding.py`; PR body summary: The torch.compile issue described in #4959 arises from overriding the `forward` method of `CustomOp`. When torch.compile is applied, the `_forward_method` of `CustomOp` is repla....
- Key implementation: `python/sglang/srt/layers/rotary_embedding.py` modified +12/-1 (13 lines); hunks: -645,7 +645,18 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; symbols: _compute_cos_sin_cache, forward, forward_hip, forward_native, touching `_compute_cos_sin_cache, forward, forward_hip`.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +12/-1 (13 lines); hunks: -645,7 +645,18 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; symbols: _compute_cos_sin_cache, forward, forward_hip, forward_native
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -645,7 +645,18 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
-    def forward(
+    def forward_hip(self, *args, **kwargs):
+        return self.forward_native(*args, **kwargs)
+    def forward(self, *args, **kwargs):
+        if torch._dynamo.is_compiling:
+            return self.forward_native(*args, **kwargs)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rotary_embedding.py` modified +12/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5371 - apply fused moe gate in ds v3/r1

- Link: https://github.com/sgl-project/sglang/pull/5371
- Status/date: merged / 2025-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +37/-16, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "apply fused moe gate in ds v3/r1"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`; PR body summary: torch profile main branch !图片 pr !图片 Only one kernel now. 36us->8us. fused_moe_gate gsm8k acc test fused moe gate performance(in H200) |qps|Input token throughput (tok/s)|Output....
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +37/-16 (53 lines); hunks: -12,6 +12,7; -25,6 +26,8; symbols: biased_grouped_topk_impl, is_power_of_two, biased_grouped_topk, select_experts, touching `biased_grouped_topk_impl, is_power_of_two, biased_grouped_topk`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +37/-16 (53 lines); hunks: -12,6 +12,7; -25,6 +26,8; symbols: biased_grouped_topk_impl, is_power_of_two, biased_grouped_topk, select_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -12,6 +12,7 @@
+import math
@@ -25,6 +26,8 @@
+if _is_cuda:
+    from sgl_kernel import moe_fused_gate
@@ -209,6 +212,10 @@ def biased_grouped_topk_impl(
+def is_power_of_two(n):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +37/-16
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5263 - [Fix] Turn off DeepGEMM by default

- Link: https://github.com/sgl-project/sglang/pull/5263
- Status/date: merged / 2025-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-2, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Turn off DeepGEMM by default"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `docs/references/deepseek.md`; PR body summary: Currently DeepGEMM is turned on by default for deepseek models, which caused confusions (#5223) on its performance due to warmup time. This PR makes DeepGEMM turned off by defau....
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +3/-1 (4 lines); hunks: -45,7 +45,9; `docs/references/deepseek.md` modified +3/-1 (4 lines); hunks: -136,7 +136,9 @@ With data parallelism attention enabled, we have achieved up....
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +3/-1 (4 lines); hunks: -45,7 +45,9
  - `docs/references/deepseek.md` modified +3/-1 (4 lines); hunks: -136,7 +136,9 @@ With data parallelism attention enabled, we have achieved up...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -45,7 +45,9 @@
-    if sm_version == 90 and get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true"):
+    if sm_version == 90 and get_bool_env_var(
+        "SGL_ENABLE_JIT_DEEPGEMM", default="false"
+    ):
diff -- docs/references/deepseek.md
@@ -136,7 +136,9 @@ With data parallelism attention enabled, we have achieved up to **1.9x** decodin
-**Usage**: Turn on by default for DeepSeek V3 models.
+- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library deisgned for FP8 matrix multiplications. Note that enabling DeepGEMM will cause large compil
+**Usage**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is turned off by default, and can be enabled with environment var
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +3/-1
  - docs: `docs/references/deepseek.md` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5381 - kernel: support slightly faster merge_state_v2 cuda kernel

- Link: https://github.com/sgl-project/sglang/pull/5381
- Status/date: merged / 2025-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +638/-4, 690 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "kernel: support slightly faster merge_state_v2 cuda kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py`; PR body summary: support slightly faster merge_state_v2 cuda kernel Performance 0.0000 means headsize is not supported. | tokens | heads | headsize | dtype | device | torch | triton | v1 | v2 |....
- Key implementation: `sgl-kernel/tests/test_merge_state_v2.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: merge_state_kernel, merge_state_triton, merge_state_torch, generate_markdown_table, touching `merge_state_kernel, merge_state_triton, merge_state_torch`; `sgl-kernel/csrc/attention/merge_attn_states.cu` added +201/-0 (201 lines); hunks: -0,0 +1,201; `sgl-kernel/python/sgl_kernel/attention.py` modified +35/-4 (39 lines); hunks: -1,4 +1,4; -10,16 +10,47 @@ def lightning_attention_decode(q, k, v, past_kv, slope, outp...; symbols: lightning_attention_decode, merge_state, merge_state_v2, cutlass_mla_decode, touching `lightning_attention_decode, merge_state, merge_state_v2`; `sgl-kernel/csrc/common_extension.cc` modified +2/-0 (2 lines); hunks: -47,6 +47,8 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {.
- Code diff details:
  - `sgl-kernel/tests/test_merge_state_v2.py` added +396/-0 (396 lines); hunks: -0,0 +1,396; symbols: merge_state_kernel, merge_state_triton, merge_state_torch, generate_markdown_table
  - `sgl-kernel/csrc/attention/merge_attn_states.cu` added +201/-0 (201 lines); hunks: -0,0 +1,201
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +35/-4 (39 lines); hunks: -1,4 +1,4; -10,16 +10,47 @@ def lightning_attention_decode(q, k, v, past_kv, slope, outp...; symbols: lightning_attention_decode, merge_state, merge_state_v2, cutlass_mla_decode
  - `sgl-kernel/csrc/common_extension.cc` modified +2/-0 (2 lines); hunks: -47,6 +47,8 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-0 (2 lines); hunks: -89,6 +89,8 @@ void lightning_attention_decode(
- Key code excerpts:

```diff
diff -- sgl-kernel/tests/test_merge_state_v2.py
@@ -0,0 +1,396 @@
+from typing import Optional
+import pytest
+import torch
+import triton
+import triton.language as tl
+from sgl_kernel import merge_state, merge_state_v2
diff -- sgl-kernel/csrc/attention/merge_attn_states.cu
@@ -0,0 +1,201 @@
+#include <ATen/cuda/CUDAContext.h>
+#include <c10/cuda/CUDAGuard.h>
+#include <algorithm>
+#include <optional>
+#include "pytorch_extension_utils.h"
+// Helper functions to convert between different data types
diff -- sgl-kernel/python/sgl_kernel/attention.py
@@ -1,4 +1,4 @@
```

- Reviewed files:
  - tests: `sgl-kernel/tests/test_merge_state_v2.py` added +396/-0
  - other: `sgl-kernel/csrc/attention/merge_attn_states.cu` added +201/-0; `sgl-kernel/python/sgl_kernel/attention.py` modified +35/-4; `sgl-kernel/csrc/common_extension.cc` modified +2/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-0; `sgl-kernel/CMakeLists.txt` modified +1/-0; `sgl-kernel/python/sgl_kernel/__init__.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_merge_state_v2.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5113 - Support MHA with chunked prefix cache for DeepSeek chunked prefill

- Link: https://github.com/sgl-project/sglang/pull/5113
- Status/date: merged / 2025-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +734/-46, 947 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support MHA with chunked prefix cache for DeepSeek chunked prefill"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`; PR body summary: The current implementation of MLA is slow when when handling long prefix lengths, such as sequences with 32k input length, as highlighted in issues like #5031. Profiling reveale....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +174/-9 (183 lines); hunks: -18,6 +18,7; -78,7 +79,7; symbols: AttnForwardMethod, DeepseekV2MLP, __init__, no_absorb, touching `AttnForwardMethod, DeepseekV2MLP, __init__`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +181/-0 (181 lines); hunks: -181,6 +181,28 @@ class ForwardBatch:; -484,6 +506,128 @@ def _compute_mrope_positions(; symbols: ForwardBatch, _compute_mrope_positions, get_max_chunk_capacity, set_prefix_chunk_idx, touching `ForwardBatch, _compute_mrope_positions, get_max_chunk_capacity`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +80/-34 (114 lines); hunks: -16,7 +16,7; -593,41 +593,87 @@ def forward_extend(; symbols: forward_extend, forward_decode, touching `forward_extend, forward_decode`; `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunks: -167,6 +167,7 @@ def __init__(; -318,6 +319,16 @@ def model_specific_adjustment(self):; symbols: __init__, model_specific_adjustment, init_torch_distributed, touching `__init__, model_specific_adjustment, init_torch_distributed`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +174/-9 (183 lines); hunks: -18,6 +18,7; -78,7 +79,7; symbols: AttnForwardMethod, DeepseekV2MLP, __init__, no_absorb
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +181/-0 (181 lines); hunks: -181,6 +181,28 @@ class ForwardBatch:; -484,6 +506,128 @@ def _compute_mrope_positions(; symbols: ForwardBatch, _compute_mrope_positions, get_max_chunk_capacity, set_prefix_chunk_idx
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +80/-34 (114 lines); hunks: -16,7 +16,7; -593,41 +593,87 @@ def forward_extend(; symbols: forward_extend, forward_decode
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunks: -167,6 +167,7 @@ def __init__(; -318,6 +319,16 @@ def model_specific_adjustment(self):; symbols: __init__, model_specific_adjustment, init_torch_distributed
  - `python/sglang/test/attention/test_prefix_chunk_info.py` added +224/-0 (224 lines); hunks: -0,0 +1,224; symbols: MockForwardBatch, __init__, get_max_chunk_capacity, MockReqToTokenPool
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -18,6 +18,7 @@
+from enum import IntEnum, auto
@@ -78,7 +79,7 @@
-    from sgl_kernel import awq_dequantize, bmm_fp8
+    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2
@@ -94,6 +95,19 @@
+class AttnForwardMethod(IntEnum):
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -181,6 +181,28 @@ class ForwardBatch:
+    # For MLA chunked prefix cache used in chunked prefill
+    # Tell attention backend whether the kv cache needs to be attended in current pass
+    attn_attend_prefix_cache: Optional[bool] = None
+    # Number of prefix cache chunks
+    num_prefix_chunks: Optional[int] = None
+    # Index of current chunk, used by attention backend
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -16,7 +16,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +174/-9; `python/sglang/srt/model_executor/forward_batch_info.py` modified +181/-0; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +80/-34; `python/sglang/srt/model_executor/model_runner.py` modified +11/-0; `python/sglang/srt/server_args.py` modified +6/-0
  - tests: `python/sglang/test/attention/test_prefix_chunk_info.py` added +224/-0; `test/srt/test_fa3.py` modified +53/-2
  - docs: `docs/references/deepseek.md` modified +3/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/test_prefix_chunk_info.py`, `test/srt/test_fa3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5205 - Refactor DeepSeek decoder layer branches

- Link: https://github.com/sgl-project/sglang/pull/5205
- Status/date: merged / 2025-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +47/-21, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor DeepSeek decoder layer branches"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This code depends on https://github.com/sgl-project/sglang/pull/5190 and please subtract diff from there.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +47/-21 (68 lines); hunks: -18,7 +18,8; -28,6 +29,7; symbols: __init__, forward, forward_normal_chunked_kv, _FFNInputMode, touching `__init__, forward, forward_normal_chunked_kv`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +47/-21 (68 lines); hunks: -18,7 +18,8; -28,6 +29,7; symbols: __init__, forward, forward_normal_chunked_kv, _FFNInputMode
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -18,7 +18,8 @@
-from enum import IntEnum, auto
+from dataclasses import dataclass
+from enum import Enum, IntEnum, auto
@@ -28,6 +29,7 @@
+    get_tensor_model_parallel_rank,
@@ -147,7 +149,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +47/-21
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #4836 - Introduce moe_dense_tp_size to fix dense layer errors in DeepSeek V3 + 4x8xH100

- Link: https://github.com/sgl-project/sglang/pull/4836
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `53dcf3887639`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +31/-1, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Introduce moe_dense_tp_size to fix dense layer errors in DeepSeek V3 + 4x8xH100"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This code depends on https://github.com/sgl-project/sglang/pull/5190 and https://github.com/sgl-project/sglang/pull/5205 and please subtract diff from there 4x8xH100 cannot run....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +17/-1 (18 lines); hunks: -1066,12 +1066,18 @@ def __init__(; -1084,6 +1090,10 @@ def __init__(; symbols: __init__, _enable_moe_dense_fully_dp, _compute_info, touching `__init__, _enable_moe_dense_fully_dp, _compute_info`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +17/-1 (18 lines); hunks: -1066,12 +1066,18 @@ def __init__(; -1084,6 +1090,10 @@ def __init__(; symbols: __init__, _enable_moe_dense_fully_dp, _compute_info
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1066,12 +1066,18 @@ def __init__(
+            if self._enable_moe_dense_fully_dp():
+                mlp_tp_rank, mlp_tp_size = 0, 1
+            else:
+                mlp_tp_rank, mlp_tp_size = None, None
+                tp_rank=mlp_tp_rank,
+                tp_size=mlp_tp_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +17/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5473 - use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel

- Link: https://github.com/sgl-project/sglang/pull/5473
- Status/date: merged / 2025-04-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +25/-6, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; PR body summary: This pr implemented a faster kernel for per_token_group_quant_fp8, which can replace triton kernel. https://github.com/sgl-project/sglang/pull/4817 I tested the kernel with 2048....
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +24/-5 (29 lines); hunks: -275,18 +275,37 @@ def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8, touching `sglang_per_token_group_quant_fp8`; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-1 (2 lines); hunks: -141,7 +141,7 @@ def apply_w8a8_block_fp8_linear(; symbols: apply_w8a8_block_fp8_linear, touching `apply_w8a8_block_fp8_linear`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +24/-5 (29 lines); hunks: -275,18 +275,37 @@ def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-1 (2 lines); hunks: -141,7 +141,7 @@ def apply_w8a8_block_fp8_linear(; symbols: apply_w8a8_block_fp8_linear
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -275,18 +275,37 @@ def sglang_per_token_group_quant_fp8(
+    column_major_scales: bool = False,
+    scale_tma_aligned: bool = False,
-    x_s = torch.empty(
-        x.shape[:-1] + (x.shape[-1] // group_size,),
-        device=x.device,
-        dtype=torch.float32,
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -141,7 +141,7 @@ def apply_w8a8_block_fp8_linear(
-            q_input, x_scale = per_token_group_quant_fp8(
+            q_input, x_scale = sglang_per_token_group_quant_fp8(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +24/-5; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5549 - Remove one kernel in per_tensor_quant_mla_fp8

- Link: https://github.com/sgl-project/sglang/pull/5549
- Status/date: merged / 2025-04-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +62/-18, 276 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove one kernel in per_tensor_quant_mla_fp8"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_nextn.py`; PR body summary: Thanks @Alcanderian for discussing it is acceptable to change APIs in the caller site Accuracy PR: 93.3 (baseline: roughly 93 when I tested in https://github.com/sgl-project/sgl....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +32/-9 (41 lines); hunks: -76,7 +76,7; -97,7 +97,6; symbols: AttnForwardMethod, forward, forward_normal, forward_absorb, touching `AttnForwardMethod, forward, forward_normal`; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-7 (16 lines); hunks: -58,10 +58,8; -897,24 +895,28 @@ def _per_tensor_quant_mla_fp8_stage2(; symbols: deep_gemm_fp8_fp8_bf16_nt, _per_tensor_quant_mla_fp8_stage2, per_tensor_quant_mla_fp8, touching `deep_gemm_fp8_fp8_bf16_nt, _per_tensor_quant_mla_fp8_stage2, per_tensor_quant_mla_fp8`; `python/sglang/srt/models/deepseek_nextn.py` modified +8/-2 (10 lines); hunks: -40,7 +40,7; -91,6 +91,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -1932,3 +1932,16 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture, BumpAllocator, __init__, allocate, touching `is_fa3_default_architecture, BumpAllocator, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +32/-9 (41 lines); hunks: -76,7 +76,7; -97,7 +97,6; symbols: AttnForwardMethod, forward, forward_normal, forward_absorb
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-7 (16 lines); hunks: -58,10 +58,8; -897,24 +895,28 @@ def _per_tensor_quant_mla_fp8_stage2(; symbols: deep_gemm_fp8_fp8_bf16_nt, _per_tensor_quant_mla_fp8_stage2, per_tensor_quant_mla_fp8
  - `python/sglang/srt/models/deepseek_nextn.py` modified +8/-2 (10 lines); hunks: -40,7 +40,7; -91,6 +91,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -1932,3 +1932,16 @@ def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture, BumpAllocator, __init__, allocate
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -76,7 +76,7 @@
-from sglang.srt.utils import DeepEPMode, add_prefix, is_cuda, is_hip
+from sglang.srt.utils import BumpAllocator, DeepEPMode, add_prefix, is_cuda, is_hip
@@ -97,7 +97,6 @@
@@ -590,6 +589,7 @@ def forward(
+        zero_allocator: BumpAllocator,
@@ -615,9 +615,13 @@ def forward(
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -58,10 +58,8 @@
@@ -897,24 +895,28 @@ def _per_tensor_quant_mla_fp8_stage2(
-    x: torch.Tensor, eps: float = 1e-12
+    x: torch.Tensor, x_s_out: torch.Tensor, eps: float = 1e-12
+    assert (
+        x_s_out.shape == (1,)
+        and x_s_out.dtype == torch.float32
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -40,7 +40,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +32/-9; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-7; `python/sglang/srt/models/deepseek_nextn.py` modified +8/-2; `python/sglang/srt/utils.py` modified +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5432 - [perf] introduce deep gemm group_gemm_masked as bmm

- Link: https://github.com/sgl-project/sglang/pull/5432
- Status/date: merged / 2025-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +361/-20, 481 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] introduce deep gemm group_gemm_masked as bmm"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/test/test_block_fp8.py`; PR body summary: per-token-group quant+deep_gemm's grouped_gemm_masked is generally faster than per-tensor quant+bmm_fp8 NOTE: Multi-batch throughput is hardly comparable because expert loads di....
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +108/-4 (112 lines); hunks: -44,6 +44,7; -53,10 +54,11; symbols: per_tensor_quant_mla_fp8, _per_token_group_quant_mla_deep_gemm_masked_fp8, per_tensor_quant_mla_deep_gemm_masked_fp8, scaled_fp8_quant, touching `per_tensor_quant_mla_fp8, _per_token_group_quant_mla_deep_gemm_masked_fp8, per_tensor_quant_mla_deep_gemm_masked_fp8`; `python/sglang/srt/models/deepseek_v2.py` modified +86/-16 (102 lines); hunks: -57,7 +57,11; -82,6 +86,7; symbols: __init__, forward_absorb, post_load_weights, touching `__init__, forward_absorb, post_load_weights`; `python/sglang/test/test_block_fp8.py` modified +167/-0 (167 lines); hunks: -7,6 +7,7; -212,6 +213,62 @@ def test_per_tensor_quant_mla_fp8(self):; symbols: test_per_tensor_quant_mla_fp8, TestPerTokenGroupQuantMlaDeepGemmMaskedFP8, setUpClass, _per_token_group_quant_mla_deep_gemm_masked_fp8, touching `test_per_tensor_quant_mla_fp8, TestPerTokenGroupQuantMlaDeepGemmMaskedFP8, setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +108/-4 (112 lines); hunks: -44,6 +44,7; -53,10 +54,11; symbols: per_tensor_quant_mla_fp8, _per_token_group_quant_mla_deep_gemm_masked_fp8, per_tensor_quant_mla_deep_gemm_masked_fp8, scaled_fp8_quant
  - `python/sglang/srt/models/deepseek_v2.py` modified +86/-16 (102 lines); hunks: -57,7 +57,11; -82,6 +86,7; symbols: __init__, forward_absorb, post_load_weights
  - `python/sglang/test/test_block_fp8.py` modified +167/-0 (167 lines); hunks: -7,6 +7,7; -212,6 +213,62 @@ def test_per_tensor_quant_mla_fp8(self):; symbols: test_per_tensor_quant_mla_fp8, TestPerTokenGroupQuantMlaDeepGemmMaskedFP8, setUpClass, _per_token_group_quant_mla_deep_gemm_masked_fp8
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -44,6 +44,7 @@
+_enable_jit_deepgemm_bmm = False
@@ -53,10 +54,11 @@
-    if sm_version == 90 and get_bool_env_var(
-        "SGL_ENABLE_JIT_DEEPGEMM", default="false"
-    ):
-        _enable_jit_deepgemm = True
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -57,7 +57,11 @@
-from sglang.srt.layers.quantization.fp8_kernel import per_tensor_quant_mla_fp8
+from sglang.srt.layers.quantization.fp8_kernel import (
+    _enable_jit_deepgemm_bmm,
+    per_tensor_quant_mla_deep_gemm_masked_fp8,
+    per_tensor_quant_mla_fp8,
+)
diff -- python/sglang/test/test_block_fp8.py
@@ -7,6 +7,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +108/-4; `python/sglang/srt/models/deepseek_v2.py` modified +86/-16
  - tests: `python/sglang/test/test_block_fp8.py` modified +167/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5571 - enable DeepSeek V3 shared_experts_fusion in sm90

- Link: https://github.com/sgl-project/sglang/pull/5571
- Status/date: merged / 2025-04-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `d9dd529854f7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-0, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "enable DeepSeek V3 shared_experts_fusion in sm90"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: H200 Benchmark I tested the benchmark using the command provided by https://github.com/sgl-project/sglang/issues/5514. To avoid warmup, I turned off deepgemm. Below are the resu....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +12/-0 (12 lines); hunks: -1427,6 +1427,18 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-0 (12 lines); hunks: -1427,6 +1427,18 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1427,6 +1427,18 @@ def __init__(
+        elif self.n_share_experts_fusion == 0:
+            if (
+                torch.cuda.get_device_capability("cuda") >= (9, 0)
+                and self.config.architectures[0] == "DeepseekV3ForCausalLM"
+                and self.config.n_routed_experts == 256
+                and (not global_server_args_dict["enable_deepep_moe"])
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +12/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5578 - Remove extra copy in deepseek forward absorb

- Link: https://github.com/sgl-project/sglang/pull/5578
- Status/date: merged / 2025-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +18/-21, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove extra copy in deepseek forward absorb"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/rotary_embedding.py`, `.github/workflows/pr-test-amd.yml`; PR body summary: Performance main branch: this PR: 0.5% improvement for bs 1 and 1.5% improvement for bs 32. main branch: this PR: removed 3 copy with 2 cat added (can overlap one later), durati....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -682,10 +682,6 @@ def forward_absorb(; -729,20 +725,20 @@ def forward_absorb(; symbols: forward_absorb, touching `forward_absorb`; `python/sglang/srt/layers/rotary_embedding.py` modified +2/-1 (3 lines); hunks: -665,6 +665,7 @@ def forward_native(; -695,7 +696,7 @@ def forward_native(; symbols: forward_native, Llama3RotaryEmbedding, touching `forward_native, Llama3RotaryEmbedding`; `.github/workflows/pr-test-amd.yml` modified +7/-7 (14 lines); hunks: -38,12 +38,12 @@ jobs:; -82,12 +82,12 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: -682,10 +682,6 @@ def forward_absorb(; -729,20 +725,20 @@ def forward_absorb(; symbols: forward_absorb
  - `python/sglang/srt/layers/rotary_embedding.py` modified +2/-1 (3 lines); hunks: -665,6 +665,7 @@ def forward_native(; -695,7 +696,7 @@ def forward_native(; symbols: forward_native, Llama3RotaryEmbedding
  - `.github/workflows/pr-test-amd.yml` modified +7/-7 (14 lines); hunks: -38,12 +38,12 @@ jobs:; -82,12 +82,12 @@ jobs:
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -682,10 +682,6 @@ def forward_absorb(
-        q_len = hidden_states.shape[0]
-        q_input = hidden_states.new_empty(
-            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
-        )
@@ -729,20 +725,20 @@ def forward_absorb(
-        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -665,6 +665,7 @@ def forward_native(
+        dtype = query.dtype
@@ -695,7 +696,7 @@ def forward_native(
-        return query, key
+        return query.to(dtype), key.to(dtype)
diff -- .github/workflows/pr-test-amd.yml
@@ -38,12 +38,12 @@ jobs:
-          docker pull lmsysorg/sglang:v0.4.5-rocm630
+          docker pull lmsysorg/sglang:v0.4.5.post2-rocm630
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +9/-13; `python/sglang/srt/layers/rotary_embedding.py` modified +2/-1
  - ci: `.github/workflows/pr-test-amd.yml` modified +7/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5641 - [feature] Add H20 fp8_w8a8 FusedMoE config for --n-share-experts-fusion=16

- Link: https://github.com/sgl-project/sglang/pull/5641
- Status/date: merged / 2025-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +146/-0, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] Add H20 fp8_w8a8 FusedMoE config for --n-share-experts-fusion=16"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; PR body summary: The original fusedMoE config on H20 will become invalid after enabling the new feature --n-share-experts-fusion=16(here tp=16). So we added relevant config to improve the perfor....
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: -0,0 +1,146
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5628 - Turn on DeepGemm By Default and Update Doc

- Link: https://github.com/sgl-project/sglang/pull/5628
- Status/date: merged / 2025-04-22
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Turn on DeepGemm By Default and Update Doc"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/deep_gemm.py`, `docs/references/deepseek.md`; PR body summary: Followup of #5580.
- Key implementation: `python/sglang/srt/layers/quantization/deep_gemm.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7; `docs/references/deepseek.md` modified +8/-2 (10 lines); hunks: -138,9 +138,15 @@ With data parallelism attention enabled, we have achieved u....
- Code diff details:
  - `python/sglang/srt/layers/quantization/deep_gemm.py` modified +1/-1 (2 lines); hunks: -25,7 +25,7
  - `docs/references/deepseek.md` modified +8/-2 (10 lines); hunks: -138,9 +138,15 @@ With data parallelism attention enabled, we have achieved u...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/deep_gemm.py
@@ -25,7 +25,7 @@
-        if get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="false"):
+        if get_bool_env_var("SGL_ENABLE_JIT_DEEPGEMM", default="true"):
diff -- docs/references/deepseek.md
@@ -138,9 +138,15 @@ With data parallelism attention enabled, we have achieved up to **1.9x** decodin
-- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library deisgned for FP8 matrix multiplications. Note that enabling DeepGEMM will cause large compil
+- **DeepGEMM**: The [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernel library optimized for FP8 matrix multiplications.
-**Usage**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is turned off by default, and can be enabled with environment var
+**Usage**: The activation and weight optimization above are turned on by default for DeepSeek V3 models. DeepGEMM is enabled by default on NVIDIA Hopper GPUs and disabled by defau
+Before serving the DeepSeek model, precompile the DeepGEMM kernels using:
+'''bash
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/deep_gemm.py` modified +1/-1
  - docs: `docs/references/deepseek.md` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/deep_gemm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5619 - Fuse q_a_proj and kv_a_proj for DeepSeek models

- Link: https://github.com/sgl-project/sglang/pull/5619
- Status/date: merged / 2025-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +78/-25, 209 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse q_a_proj and kv_a_proj for DeepSeek models"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: In self attention module of DeepSeek v3, the `q_a_proj` and `kv_a_proj` both takes hidden state as input, so they can be fused into one module, and one launching of DeepGemm is....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +78/-25 (103 lines); hunks: -443,12 +443,12 @@ def __init__(; -470,6 +470,14 @@ def __init__(; symbols: __init__, forward_normal, forward_absorb, touching `__init__, forward_normal, forward_absorb`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +78/-25 (103 lines); hunks: -443,12 +443,12 @@ def __init__(; -470,6 +470,14 @@ def __init__(; symbols: __init__, forward_normal, forward_absorb
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -443,12 +443,12 @@ def __init__(
-            self.q_a_proj = ReplicatedLinear(
+            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
-                self.q_lora_rank,
+                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
-                prefix=add_prefix("q_a_proj", prefix),
+                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +78/-25
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5707 - [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1

- Link: https://github.com/sgl-project/sglang/pull/5707
- Status/date: merged / 2025-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `5d93a950eeeb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-15, 150 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: We found bugs while using MTP and `-n-share-experts-fusion` together. 1. While using `--n-share-experts-fusion ${TP_SIZE}` with MTP, there will be weight loading error: !image 2....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +18/-14 (32 lines); hunks: -1431,11 +1431,27 @@ def __init__(; -1450,7 +1466,7 @@ def __init__(; symbols: __init__, determine_n_share_experts_fusion, get_input_embeddings, touching `__init__, determine_n_share_experts_fusion, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-14 (32 lines); hunks: -1431,11 +1431,27 @@ def __init__(; -1450,7 +1466,7 @@ def __init__(; symbols: __init__, determine_n_share_experts_fusion, get_input_embeddings
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1431,11 +1431,27 @@ def __init__(
+        self.determine_n_share_experts_fusion()
+        self.model = DeepseekV2Model(
+            config, quant_config, prefix=add_prefix("model", prefix)
+        )
+        self.lm_head = ParallelLMHead(
+            config.vocab_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +18/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5740 - update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions.

- Link: https://github.com/sgl-project/sglang/pull/5740
- Status/date: merged / 2025-04-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +45/-42, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions."; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; PR body summary: Reproduction commands :.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +41/-41 (82 lines); hunks: -1,102 +1,102; -107,40 +107,40; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +4/-1 (5 lines); hunks: -940,7 +940,10 @@ def get_moe_configs(; symbols: get_moe_configs, touching `get_moe_configs`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +41/-41 (82 lines); hunks: -1,102 +1,102; -107,40 +107,40
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +4/-1 (5 lines); hunks: -940,7 +940,10 @@ def get_moe_configs(; symbols: get_moe_configs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -1,102 +1,102 @@
-        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_M": 16,
-        "GROUP_SIZE_M": 16,
+        "GROUP_SIZE_M": 1,
-        "BLOCK_SIZE_M": 64,
-        "BLOCK_SIZE_N": 32,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -940,7 +940,10 @@ def get_moe_configs(
-            logger.info("Using configuration from %s for MoE layer.", config_file_path)
+            logger.info(
+                "Using configuration from %s for MoE layer. Please note that due to the large number of configs under fused_moe_triton/configs potentially not being tuned with the
+                config_file_path,
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +41/-41; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5716 - perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling

- Link: https://github.com/sgl-project/sglang/pull/5716
- Status/date: merged / 2025-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +27/-27, 132 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; PR body summary: We have provided fused_moe_triton config in PR(https://github.com/sgl-project/sglang/pull/5641) to improve model performance. However, we found that after using this config, the....
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +27/-27 (54 lines); hunks: -5,19 +5,19; -33,25 +33,25.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +27/-27 (54 lines); hunks: -5,19 +5,19; -33,25 +33,25
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json
@@ -5,19 +5,19 @@
-        "num_stages": 3
+        "num_stages": 4
-        "BLOCK_SIZE_K": 64,
-        "GROUP_SIZE_M": 1,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +27/-27
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5748 - Fuse MLA set kv cache kernel

- Link: https://github.com/sgl-project/sglang/pull/5748
- Status/date: merged / 2025-04-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +100/-9, 183 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse MLA set kv cache kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Fuse MLA set kv cache kernel and remove k concat operation. Currently only support FA3 backend. Can be applied to other backend with subsequent verification. Accuracy Performanc....
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +6/-4 (10 lines); hunks: -625,6 +625,7 @@ def forward_extend(; -639,11 +640,11 @@ def forward_extend(; symbols: forward_extend, forward_decode, touching `forward_extend, forward_decode`; `python/sglang/srt/layers/radix_attention.py` modified +5/-2 (7 lines); hunks: -92,8 +92,11 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-3 (5 lines); hunks: -757,14 +757,13 @@ def forward_absorb(; symbols: forward_absorb, touching `forward_absorb`; `python/sglang/srt/mem_cache/memory_pool.py` modified +87/-0 (87 lines); hunks: -34,6 +34,8; -405,6 +407,72 @@ def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype,...; symbols: copy_two_array, set_mla_kv_buffer_kernel, set_mla_kv_buffer_triton, MLATokenToKVPool, touching `copy_two_array, set_mla_kv_buffer_kernel, set_mla_kv_buffer_triton`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +6/-4 (10 lines); hunks: -625,6 +625,7 @@ def forward_extend(; -639,11 +640,11 @@ def forward_extend(; symbols: forward_extend, forward_decode
  - `python/sglang/srt/layers/radix_attention.py` modified +5/-2 (7 lines); hunks: -92,8 +92,11 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-3 (5 lines); hunks: -757,14 +757,13 @@ def forward_absorb(; symbols: forward_absorb
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +87/-0 (87 lines); hunks: -34,6 +34,8; -405,6 +407,72 @@ def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype,...; symbols: copy_two_array, set_mla_kv_buffer_kernel, set_mla_kv_buffer_triton, MLATokenToKVPool
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -625,6 +625,7 @@ def forward_extend(
+        k_rope: Optional[torch.Tensor] = None,
@@ -639,11 +640,11 @@ def forward_extend(
-                    forward_batch.token_to_kv_pool.set_kv_buffer(
+                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
-                        v,
+                        k_rope,
diff -- python/sglang/srt/layers/radix_attention.py
@@ -92,8 +92,11 @@ def forward(
-            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
-            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
+            if "k_rope" not in kwargs:
+                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
+                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
+            else:
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -757,14 +757,13 @@ def forward_absorb(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +6/-4; `python/sglang/srt/layers/radix_attention.py` modified +5/-2; `python/sglang/srt/models/deepseek_v2.py` modified +2/-3; `python/sglang/srt/mem_cache/memory_pool.py` modified +87/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5390 - Add Cutlass MLA attention backend

- Link: https://github.com/sgl-project/sglang/pull/5390
- Status/date: merged / 2025-04-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +305/-3, 373 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add Cutlass MLA attention backend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/utils.py`; PR body summary: Enables use of the blackwell cutlass MLA decode kernel with deepseek models. Adds "cutlass_mla" option for attention backend. Deepseek-R1 Benchmarks Using `--attention-backend c....
- Key implementation: `python/sglang/srt/layers/attention/cutlass_mla_backend.py` added +278/-0 (278 lines); hunks: -0,0 +1,278; symbols: CutlassMLADecodeMetadata, __init__, CutlassMLABackend, init_forward_metadata, touching `CutlassMLADecodeMetadata, __init__, CutlassMLABackend`; `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -271,6 +271,7 @@ def model_specific_adjustment(self):; -926,6 +927,12 @@ def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend, touching `model_specific_adjustment, init_attention_backend`; `python/sglang/srt/layers/attention/utils.py` modified +1/-1 (2 lines); hunks: -49,8 +49,8 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton, touching `create_flashmla_kv_indices_triton`; `python/sglang/srt/server_args.py` modified +14/-1 (15 lines); hunks: -256,6 +256,12 @@ def __post_init__(self):; -823,7 +829,14 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: __post_init__, add_cli_args, touching `__post_init__, add_cli_args`.
- Code diff details:
  - `python/sglang/srt/layers/attention/cutlass_mla_backend.py` added +278/-0 (278 lines); hunks: -0,0 +1,278; symbols: CutlassMLADecodeMetadata, __init__, CutlassMLABackend, init_forward_metadata
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -271,6 +271,7 @@ def model_specific_adjustment(self):; -926,6 +927,12 @@ def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/layers/attention/utils.py` modified +1/-1 (2 lines); hunks: -49,8 +49,8 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton
  - `python/sglang/srt/server_args.py` modified +14/-1 (15 lines); hunks: -256,6 +256,12 @@ def __post_init__(self):; -823,7 +829,14 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: __post_init__, add_cli_args
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +3/-0 (3 lines); hunks: -78,6 +78,7 @@ def cutlass_mla_decode(; -109,6 +110,8 @@ def cutlass_mla_decode(; symbols: cutlass_mla_decode, cutlass_mla_get_workspace_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/cutlass_mla_backend.py
@@ -0,0 +1,278 @@
+from __future__ import annotations
+"""
+Support attention backend for Cutlass MLA.
+"""
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, Optional, Union
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -271,6 +271,7 @@ def model_specific_adjustment(self):
+                    "cutlass_mla",
@@ -926,6 +927,12 @@ def init_attention_backend(self):
+        elif self.server_args.attention_backend == "cutlass_mla":
+            from sglang.srt.layers.attention.cutlass_mla_backend import (
+                CutlassMLABackend,
+            )
diff -- python/sglang/srt/layers/attention/utils.py
@@ -49,8 +49,8 @@ def create_flashmla_kv_indices_triton(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/cutlass_mla_backend.py` added +278/-0; `python/sglang/srt/model_executor/model_runner.py` modified +7/-0; `python/sglang/srt/layers/attention/utils.py` modified +1/-1; `python/sglang/srt/server_args.py` modified +14/-1; `python/sglang/srt/managers/schedule_batch.py` modified +1/-0
  - other: `sgl-kernel/python/sgl_kernel/attention.py` modified +3/-0
  - docs: `docs/backend/server_arguments.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/managers/schedule_batch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5793 - Auto set draft model path for MTP

- Link: https://github.com/sgl-project/sglang/pull/5793
- Status/date: merged / 2025-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +115/-287, 522 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Auto set draft model path for MTP"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/model_runner.py`; PR body summary: - load nextn layer weights directly from target model - refactor to reuse common weight loading logic for draft model and target model - auto set draft model path if not given W....
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +1/-257 (258 lines); hunks: -177,263 +177,7 @@ def forward(; symbols: forward, load_weights, touching `forward, load_weights`; `python/sglang/srt/models/deepseek_v2.py` modified +74/-17 (91 lines); hunks: -1502,11 +1502,20 @@ def forward(; -1612,7 +1621,20 @@ def post_load_weights(self):; symbols: forward, post_load_weights, load_weights, touching `forward, post_load_weights, load_weights`; `python/sglang/srt/model_executor/model_runner.py` modified +11/-2 (13 lines); hunks: -692,9 +692,14 @@ def profile_max_num_token(self, total_gpu_memory: int):; -809,7 +814,11 @@ def init_memory_pool(; symbols: profile_max_num_token, init_memory_pool, touching `profile_max_num_token, init_memory_pool`; `python/sglang/srt/configs/model_config.py` modified +7/-0 (7 lines); hunks: -47,6 +47,7 @@ def __init__(; -85,6 +86,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-257 (258 lines); hunks: -177,263 +177,7 @@ def forward(; symbols: forward, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-17 (91 lines); hunks: -1502,11 +1502,20 @@ def forward(; -1612,7 +1621,20 @@ def post_load_weights(self):; symbols: forward, post_load_weights, load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-2 (13 lines); hunks: -692,9 +692,14 @@ def profile_max_num_token(self, total_gpu_memory: int):; -809,7 +814,11 @@ def init_memory_pool(; symbols: profile_max_num_token, init_memory_pool
  - `python/sglang/srt/configs/model_config.py` modified +7/-0 (7 lines); hunks: -47,6 +47,7 @@ def __init__(; -85,6 +86,12 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/server_args.py` modified +21/-11 (32 lines); hunks: -22,7 +22,7; -333,6 +333,14 @@ def __post_init__(self):; symbols: __post_init__, __call__, auto_choose_speculative_params, get_model_arch
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -177,263 +177,7 @@ def forward(
-        if hasattr(self.config, "num_nextn_predict_layers"):
-            num_nextn_layers = self.config.num_nextn_predict_layers
-            assert num_nextn_layers == 1, "Only 1 nextn layer is supportted"
-            assert num_nextn_layers == self.config.num_hidden_layers
-        else:
-            raise ValueError("num_nextn_predict_layers is not in the config")
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1502,11 +1502,20 @@ def forward(
-    def post_load_weights(self):
+    def post_load_weights(self, is_nextn=False):
-        for layer_id in range(self.config.num_hidden_layers):
-            self_attn = self.model.layers[layer_id].self_attn
+        layer_ids = (
+            range(self.config.num_hidden_layers)
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -692,9 +692,14 @@ def profile_max_num_token(self, total_gpu_memory: int):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +1/-257; `python/sglang/srt/models/deepseek_v2.py` modified +74/-17; `python/sglang/srt/model_executor/model_runner.py` modified +11/-2; `python/sglang/srt/configs/model_config.py` modified +7/-0; `python/sglang/srt/server_args.py` modified +21/-11; `python/sglang/srt/managers/tp_worker.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5952 - Update ci test and doc for MTP api change

- Link: https://github.com/sgl-project/sglang/pull/5952
- Status/date: merged / 2025-05-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +66/-14, 124 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update ci test and doc for MTP api change"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md`; PR body summary: Update ci test and doc for MTP api change..
- Key implementation: `test/srt/test_mla_deepseek_v3.py` modified +57/-0 (57 lines); hunks: -50,6 +50,63 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDeepseekV3MTP, setUpClass, tearDownClass, touching `test_gsm8k, TestDeepseekV3MTP, setUpClass`; `python/sglang/srt/server_args.py` modified +7/-4 (11 lines); hunks: -347,10 +347,13 @@ def __post_init__(self):; symbols: __post_init__, touching `__post_init__`; `docs/references/deepseek.md` modified +2/-4 (6 lines); hunks: -153,12 +153,10 @@ The precompilation process typically takes around 10 minut...; `test/srt/test_full_deepseek_v3.py` modified +0/-2 (2 lines); hunks: -80,8 +80,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/srt/test_mla_deepseek_v3.py` modified +57/-0 (57 lines); hunks: -50,6 +50,63 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestDeepseekV3MTP, setUpClass, tearDownClass
  - `python/sglang/srt/server_args.py` modified +7/-4 (11 lines); hunks: -347,10 +347,13 @@ def __post_init__(self):; symbols: __post_init__
  - `docs/references/deepseek.md` modified +2/-4 (6 lines); hunks: -153,12 +153,10 @@ The precompilation process typically takes around 10 minut...
  - `test/srt/test_full_deepseek_v3.py` modified +0/-2 (2 lines); hunks: -80,8 +80,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/srt/test_mla_flashinfer.py` modified +0/-2 (2 lines); hunks: -118,8 +118,6 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/srt/test_mla_deepseek_v3.py
@@ -50,6 +50,63 @@ def test_gsm8k(self):
+    @classmethod
+    def setUpClass(cls):
+        cls.model = "lmsys/sglang-ci-dsv3-test"
+        cls.base_url = DEFAULT_URL_FOR_TEST
+        other_args = [
+            "--trust-remote-code",
diff -- python/sglang/srt/server_args.py
@@ -347,10 +347,13 @@ def __post_init__(self):
-            if self.speculative_draft_model_path is None and model_arch in [
-                "DeepseekV3ForCausalLM"
-            ]:
-                self.speculative_draft_model_path = self.model_path
+            if model_arch == "DeepseekV3ForCausalLM":
+                if self.speculative_draft_model_path is None:
diff -- docs/references/deepseek.md
@@ -153,12 +153,10 @@ The precompilation process typically takes around 10 minutes to complete.
```

- Reviewed files:
  - tests: `test/srt/test_mla_deepseek_v3.py` modified +57/-0; `test/srt/test_full_deepseek_v3.py` modified +0/-2; `test/srt/test_mla_flashinfer.py` modified +0/-2; `test/srt/test_mla_int8_deepseek_v3.py` modified +0/-2
  - runtime: `python/sglang/srt/server_args.py` modified +7/-4
  - docs: `docs/references/deepseek.md` modified +2/-4
- Risk and verification: The diff ships test coverage in `test/srt/test_full_deepseek_v3.py`, `test/srt/test_mla_deepseek_v3.py`, `test/srt/test_mla_flashinfer.py`, `test/srt/test_mla_int8_deepseek_v3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #5908 - feat: Refactor DeepSeekV3 function call

- Link: https://github.com/sgl-project/sglang/pull/5908
- Status/date: merged / 2025-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv3.jinja`; associated commits `170d1f218a4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +95/-29, 155 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: Refactor DeepSeekV3 function call"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `examples/chat_template/tool_chat_template_deepseekv3.jinja`; PR body summary: Instead of keeping editing the prompt engineering for `deepseekv3` in `adapter.py`, this PR provides a chat template instead. Implements: https://github.com/sgl-project/sglang/p....
- Key implementation: `examples/chat_template/tool_chat_template_deepseekv3.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` added +91/-0 (91 lines); hunks: -0,0 +1,91
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv3.jinja
@@ -0,0 +1,91 @@
+{% if not add_generation_prompt is defined %}
+    {% set add_generation_prompt = false %}
+{% endif %}
+{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='', is_first_sp=true, is_last_user=false) %}
+{%- for message in messages %}
+    {%- if message['role'] == 'system' %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_deepseekv3.jinja` added +91/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/openai_api/adapter.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5977 - Overlap qk norm with two streams

- Link: https://github.com/sgl-project/sglang/pull/5977
- Status/date: merged / 2025-05-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-6, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Overlap qk norm with two streams"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Overlap qk rms norm with two streams in cuda graph. Accuracy Performance main branch: this PR: Profile main branch: this PR: 23.9μm -> 21.5μm.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +26/-6 (32 lines); hunks: -421,6 +421,7 @@ def __init__(; -543,6 +544,8 @@ def __init__(; symbols: __init__, forward_absorb, touching `__init__, forward_absorb`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-6 (32 lines); hunks: -421,6 +421,7 @@ def __init__(; -543,6 +544,8 @@ def __init__(; symbols: __init__, forward_absorb
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -421,6 +421,7 @@ def __init__(
+        alt_stream: Optional[torch.cuda.Stream] = None,
@@ -543,6 +544,8 @@ def __init__(
+        self.alt_stream = alt_stream
@@ -706,14 +709,32 @@ def forward_absorb(
-            q = self.q_a_layernorm(q)
+            k_nope = latent_cache[..., : self.kv_lora_rank]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +26/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6011 - feat: flashinfer_mla speculative decoding with custom mask

- Link: https://github.com/sgl-project/sglang/pull/6011
- Status/date: open / 2025-05-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +70/-3, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: flashinfer_mla speculative decoding with custom mask"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py`; PR body summary: status: not ready. dependent on https://github.com/flashinfer-ai/flashinfer/pull/1017. Support top_k > 1 Eagle speculative decoding with flashinfer MLA backend on custom mask. -....
- Key implementation: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +2/-0 (2 lines); hunks: -634,6 +634,7 @@ def call_begin_forward(; -651,6 +652,7 @@ def call_begin_forward(; symbols: call_begin_forward, touching `call_begin_forward`; `test/srt/test_mla_flashinfer.py` modified +61/-0 (61 lines); hunks: -162,6 +162,67 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestFlashinferMLAMTPTopK, setUpClass, tearDownClass, touching `test_gsm8k, TestFlashinferMLAMTPTopK, setUpClass`; `python/sglang/srt/speculative/eagle_worker.py` modified +7/-3 (10 lines); hunks: -346,9 +346,13 @@ def draft(self, batch: ScheduleBatch):; symbols: draft, touching `draft`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +2/-0 (2 lines); hunks: -634,6 +634,7 @@ def call_begin_forward(; -651,6 +652,7 @@ def call_begin_forward(; symbols: call_begin_forward
  - `test/srt/test_mla_flashinfer.py` modified +61/-0 (61 lines); hunks: -162,6 +162,67 @@ def test_gsm8k(self):; symbols: test_gsm8k, TestFlashinferMLAMTPTopK, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +7/-3 (10 lines); hunks: -346,9 +346,13 @@ def draft(self, batch: ScheduleBatch):; symbols: draft
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -634,6 +634,7 @@ def call_begin_forward(
+                custom_mask=custom_mask,
@@ -651,6 +652,7 @@ def call_begin_forward(
+                custom_mask=custom_mask,
diff -- test/srt/test_mla_flashinfer.py
@@ -162,6 +162,67 @@ def test_gsm8k(self):
+class TestFlashinferMLAMTPTopK(CustomTestCase):
+    @classmethod
+    def setUpClass(cls):
+        cls.model = "lmsys/sglang-ci-dsv3-test"
+        cls.base_url = DEFAULT_URL_FOR_TEST
+        other_args = ["--trust-remote-code"]
diff -- python/sglang/srt/speculative/eagle_worker.py
@@ -346,9 +346,13 @@ def draft(self, batch: ScheduleBatch):
-                raise NotImplementedError(
-                    "page_size > 1 and top_k > 1 are not supported."
-                )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +2/-0; `python/sglang/srt/speculative/eagle_worker.py` modified +7/-3
  - tests: `test/srt/test_mla_flashinfer.py` modified +61/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_mla_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6034 - Update doc for MLA attention backends

- Link: https://github.com/sgl-project/sglang/pull/6034
- Status/date: merged / 2025-05-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-3, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update doc for MLA attention backends"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `docs/references/deepseek.md`, `docs/backend/server_arguments.md`; no usable PR-body summary.
- Key implementation: `docs/references/deepseek.md` modified +2/-2 (4 lines); hunks: -87,7 +87,7 @@ Please refer to [the example](https://github.com/sgl-project/s...; -158,7 +158,7 @@ Add arguments `--speculative-algorithm`, `--speculative-num-...; `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunks: -168,7 +168,7 @@ Please consult the documentation below and [server_args.py](....
- Code diff details:
  - `docs/references/deepseek.md` modified +2/-2 (4 lines); hunks: -87,7 +87,7 @@ Please refer to [the example](https://github.com/sgl-project/s...; -158,7 +158,7 @@ Add arguments `--speculative-algorithm`, `--speculative-num-...
  - `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunks: -168,7 +168,7 @@ Please consult the documentation below and [server_args.py](...
- Key code excerpts:

```diff
diff -- docs/references/deepseek.md
@@ -87,7 +87,7 @@ Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/be
-- **MLA Attention Backends**: Currently SGLang supports different optimized MLA attention backends, including [FlashAttention3](https://github.com/Dao-AILab/flash-attention), [Fla
+- **MLA Attention Backends**: Currently SGLang supports different optimized MLA attention backends, including [FlashAttention3](https://github.com/Dao-AILab/flash-attention), [Fla
@@ -158,7 +158,7 @@ Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--speculati
-When using FlashInfer MLA wrapper (`--attention-backend flashinfer`) with speculative decoding, set the `--speculative-eagle-topk` parameter to `1`. The FlashAttention 3 backend a
+- FlashAttention3 and Triton backend fully supports MTP usage. For FlashInfer backend (`--attention-backend flashinfer`) with speculative decoding,`--speculative-eagle-topk` param
diff -- docs/backend/server_arguments.md
@@ -168,7 +168,7 @@ Please consult the documentation below and [server_args.py](https://github.com/s
-| `attention_backend` | This argument specifies the backend for attention computation and KV cache management, which can be `fa3`, `flashinfer`, `triton`, `cutlass_mla`, or `torch
+| `attention_backend` | This argument specifies the backend for attention computation and KV cache management, which can be `fa3`, `flashinfer`, `triton`, `flashmla`, `cutlass_mla
```

- Reviewed files:
  - docs: `docs/references/deepseek.md` modified +2/-2; `docs/backend/server_arguments.md` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs/backend/server_arguments.md`, `docs/references/deepseek.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #6079 - Clean logs for DeepSeek-V3 launching

- Link: https://github.com/sgl-project/sglang/pull/6079
- Status/date: merged / 2025-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `73600673bb1d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +61/-39, 231 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Clean logs for DeepSeek-V3 launching"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: When launching Deepseek-V3 model, some of the logs only need to be printed once (on tp_rank=0). This PR adds a `should_rank` util function. And print these logs only when `shoul....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunks: -88,6 +88,7; -1487,8 +1488,9 @@ def determine_n_share_experts_fusion(; symbols: determine_n_share_experts_fusion, get_input_embeddings, touching `determine_n_share_experts_fusion, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunks: -88,6 +88,7; -1487,8 +1488,9 @@ def determine_n_share_experts_fusion(; symbols: determine_n_share_experts_fusion, get_input_embeddings
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -88,6 +88,7 @@
+    log_info_on_rank0,
@@ -1487,8 +1488,9 @@ def determine_n_share_experts_fusion(
-                logger.info(
-                    "Only Deepseek V3/R1 can use shared experts fusion optimization. Shared experts fusion optimization is disabled."
+                log_info_on_rank0(
+                    logger,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/device_communicators/pynccl.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5662 - [perf] dsv3 bmm fallback to bf16

- Link: https://github.com/sgl-project/sglang/pull/5662
- Status/date: merged / 2025-05-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `5e02330137a1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +48/-3, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] dsv3 bmm fallback to bf16"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Align to official impl according to released profile data: https://github.com/deepseek-ai/profile-data/blob/ad8f61ed5e590c4984bb6579f370f9689c427416/decode.json Issue1(fixed): e....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-3 (16 lines); hunks: -63,6 +63,7; -1589,13 +1590,22 @@ def post_load_weights(self, is_nextn=False):; symbols: post_load_weights, touching `post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-3 (16 lines); hunks: -63,6 +63,7; -1589,13 +1590,22 @@ def post_load_weights(self, is_nextn=False):; symbols: post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -63,6 +63,7 @@
+    block_quant_dequant,
@@ -1589,13 +1590,22 @@ def post_load_weights(self, is_nextn=False):
-                            and _ENABLE_JIT_DEEPGEMM
-                            block_scale = weight_scale
-                            use_deep_gemm_bmm = True
+                            if _ENABLE_JIT_DEEPGEMM and get_bool_env_var(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6109 - [Feat] Support FlashMLA backend with MTP and FP8 KV cache

- Link: https://github.com/sgl-project/sglang/pull/6109
- Status/date: merged / 2025-05-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +444/-87, 706 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] Support FlashMLA backend with MTP and FP8 KV cache"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; PR body summary: This PR improves flashmla backend by accelerating decode stage with mtp. The implementation utilizes the feature that flashmla can handle `seq_len_q > 1`. To use flashmla with m....
- Key implementation: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +340/-78 (418 lines); hunks: -8,7 +8,7; -30,8 +30,8; symbols: __init__, FlashMLABackend, init_forward_metadata, touching `__init__, FlashMLABackend, init_forward_metadata`; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +8/-4 (12 lines); hunks: -346,7 +346,6 @@ def forward_extend(; -381,6 +380,9 @@ def forward_extend(; symbols: forward_extend, forward_decode, __init__, touching `forward_extend, forward_decode, __init__`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +5/-1 (6 lines); hunks: -30,6 +30,7; -210,7 +211,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/attention/utils.py` modified +2/-2 (4 lines); hunks: -77,8 +77,8 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton, touching `create_flashmla_kv_indices_triton`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +340/-78 (418 lines); hunks: -8,7 +8,7; -30,8 +30,8; symbols: __init__, FlashMLABackend, init_forward_metadata
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +8/-4 (12 lines); hunks: -346,7 +346,6 @@ def forward_extend(; -381,6 +380,9 @@ def forward_extend(; symbols: forward_extend, forward_decode, __init__
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +5/-1 (6 lines); hunks: -30,6 +30,7; -210,7 +211,10 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
  - `python/sglang/srt/layers/attention/utils.py` modified +2/-2 (4 lines); hunks: -77,8 +77,8 @@ def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton
  - `test/srt/test_flashmla.py` modified +68/-0 (68 lines); hunks: -6,6 +6,7; -14,6 +15,7; symbols: test_latency, TestFlashMLAMTP, setUpClass, tearDownClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashmla_backend.py
@@ -8,7 +8,7 @@
-from typing import TYPE_CHECKING, Optional, Union
+from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union
@@ -30,8 +30,8 @@
-# TODO The current setup is hard-coded and will be changed after integrating with MTP.
-Q_LEN = 1
+# FlashMLA FP8 issue: https://github.com/deepseek-ai/FlashMLA/issues/56
diff -- python/sglang/srt/layers/attention/flashinfer_mla_backend.py
@@ -346,7 +346,6 @@ def forward_extend(
-        k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
@@ -381,6 +380,9 @@ def forward_extend(
+            k_buf = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
+                q.dtype
+            )
@@ -442,7 +444,9 @@ def forward_decode(
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -30,6 +30,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashmla_backend.py` modified +340/-78; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +8/-4; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +5/-1; `python/sglang/srt/layers/attention/utils.py` modified +2/-2; `python/sglang/srt/speculative/eagle_worker.py` modified +13/-0
  - tests: `test/srt/test_flashmla.py` modified +68/-0
  - docs: `docs/backend/attention_backend.md` modified +7/-1; `docs/references/deepseek.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_flashmla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6151 - [Feat] optimize Qwen3 on H20 by hybrid Attention Backend

- Link: https://github.com/sgl-project/sglang/pull/6151
- Status/date: closed / 2025-05-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +39/-9, 116 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feat] optimize Qwen3 on H20 by hybrid Attention Backend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py`; PR body summary: FA3 decode performance is significantly lower than flashinfer on H20 5630 https://github.com/Dao-AILab/flash-attention/issues/1572 I have profiled the Qwen3-235B-A22B performanc....
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +24/-4 (28 lines); hunks: -143,6 +143,7 @@ def __init__(; -977,6 +978,12 @@ def init_attention_backend(self):; symbols: __init__, init_attention_backend, init_double_sparsity_channel_config, apply_torch_tp, touching `__init__, init_attention_backend, init_double_sparsity_channel_config`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-5 (14 lines); hunks: -211,9 +211,13 @@ def __init__(self, model_runner: ModelRunner):; -445,7 +449,7 @@ def capture_one_batch_size(self, bs: int, forward: Callable):; symbols: __init__, capture_one_batch_size, replay_prepare, touching `__init__, capture_one_batch_size, replay_prepare`; `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -127,6 +127,7 @@ class ServerArgs:; -873,6 +874,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +24/-4 (28 lines); hunks: -143,6 +143,7 @@ def __init__(; -977,6 +978,12 @@ def init_attention_backend(self):; symbols: __init__, init_attention_backend, init_double_sparsity_channel_config, apply_torch_tp
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-5 (14 lines); hunks: -211,9 +211,13 @@ def __init__(self, model_runner: ModelRunner):; -445,7 +449,7 @@ def capture_one_batch_size(self, bs: int, forward: Callable):; symbols: __init__, capture_one_batch_size, replay_prepare
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: -127,6 +127,7 @@ class ServerArgs:; -873,6 +874,11 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -143,6 +143,7 @@ def __init__(
+        self.decode_attn_backend = None
@@ -977,6 +978,12 @@ def init_attention_backend(self):
+        if self.server_args.enable_flashinfer_attention_decode:
+            from sglang.srt.layers.attention.flashinfer_backend import (
+                FlashInferAttnBackend,
+            )
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -211,9 +211,13 @@ def __init__(self, model_runner: ModelRunner):
-        self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
+        if self.model_runner.decode_attn_backend is None:
+            self.decode_attn_backend = self.model_runner.attn_backend
+        else:
+            self.decode_attn_backend = self.model_runner.decode_attn_backend
+        self.decode_attn_backend.init_cuda_graph_state(self.max_num_token)
diff -- python/sglang/srt/server_args.py
@@ -127,6 +127,7 @@ class ServerArgs:
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +24/-4; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-5; `python/sglang/srt/server_args.py` modified +6/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6655 - fix(deepseekv3): Fix DeepSeekV3Detector tool_index assignment and multi-tool call streaming support

- Link: https://github.com/sgl-project/sglang/pull/6655
- Status/date: merged / 2025-05-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv3_detector.py`; associated commits `461a73028070`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +52/-4, 100 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix(deepseekv3): Fix DeepSeekV3Detector tool_index assignment and multi-tool call streaming support"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv3_detector.py`; PR body summary: This PR focuses to fix the current two issues in `DeepSeekV3Detector`. 1. tool_index: Current the tool_index of each ToolCallItem is assigned based on the index of the function....
- Key implementation: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +52/-4 (56 lines); hunks: -31,6 +31,7 @@ def __init__(self):; -75,7 +76,12 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment, touching `__init__, has_tool_call, parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +52/-4 (56 lines); hunks: -31,6 +31,7 @@ def __init__(self):; -75,7 +76,12 @@ def parse_streaming_increment(; symbols: __init__, has_tool_call, parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv3_detector.py
@@ -31,6 +31,7 @@ def __init__(self):
+        self.current_tool_id = -1
@@ -75,7 +76,12 @@ def parse_streaming_increment(
-        if self.bot_token not in current_text:
+        # Check if we have a tool call (either the start token or individual tool call)
+        has_tool_call = (
+            self.bot_token in current_text or "<｜tool▁call▁begin｜>" in current_text
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +52/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv3_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6738 - Partially supports using MHA kernels in MLA forward when page-size > 1.

- Link: https://github.com/sgl-project/sglang/pull/6738
- Status/date: open / 2025-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-0, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Partially supports using MHA kernels in MLA forward when page-size > 1."; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: add Partially support to use MHA kernels when page-size > 1. related to #6411 add AttnForwardMethod.MHA_PAGED_PREFILL, and when any(forward_batch.extend_prefix_lens_cpu) == Fals....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +41/-0 (41 lines); hunks: -152,6 +152,10 @@ class AttnForwardMethod(IntEnum):; -1005,6 +1009,7 @@ def _dispatch_mla_subtype():; symbols: AttnForwardMethod, _dispatch_mla_subtype, forward_prepare, forward_core, touching `AttnForwardMethod, _dispatch_mla_subtype, forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +41/-0 (41 lines); hunks: -152,6 +152,10 @@ class AttnForwardMethod(IntEnum):; -1005,6 +1009,7 @@ def _dispatch_mla_subtype():; symbols: AttnForwardMethod, _dispatch_mla_subtype, forward_prepare, forward_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -152,6 +152,10 @@ class AttnForwardMethod(IntEnum):
+    # Use multi-head attention, but with paged KV cache.
+    # This method only support MHA without prefix-cache for now.
+    MHA_PAGED_PREFILL = auto()
@@ -1005,6 +1009,7 @@ def _dispatch_mla_subtype():
+            sum_extend_prefix_lens = 0
@@ -1018,6 +1023,16 @@ def _dispatch_mla_subtype():
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +41/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6265 - [fix][RL] Fix DeepSeekV3ForCausalLM.post_load_weights for multiple update weight

- Link: https://github.com/sgl-project/sglang/pull/6265
- Status/date: merged / 2025-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `51cdd81f9720`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +47/-14, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix][RL] Fix DeepSeekV3ForCausalLM.post_load_weights for multiple update weight"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: When doing RL, which will call `post_load_weights` a second time apart from the intialization, the origin `post_load_weights` will somehow corrupt the weight when recreate w_kv....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +39/-14 (53 lines); hunks: -92,6 +92,7; -1713,14 +1714,23 @@ def forward(; symbols: forward, post_load_weights, touching `forward, post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +39/-14 (53 lines); hunks: -92,6 +92,7; -1713,14 +1714,23 @@ def forward(; symbols: forward, post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -92,6 +92,7 @@
+    bind_or_assign,
@@ -1713,14 +1714,23 @@ def forward(
-    def post_load_weights(self, is_nextn=False):
+    def post_load_weights(self, is_nextn=False, weight_names=None):
-        layer_ids = (
-            range(self.config.num_hidden_layers)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +39/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6890 - Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa

- Link: https://github.com/sgl-project/sglang/pull/6890
- Status/date: merged / 2025-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_utils.py`; PR body summary: ~~do not merge now, I will test and check more first~~ Test: * triton: 41us * change to DeepGEMM: 12us thus more than 3x speedup for this kernel change in my scenario.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2 (4 lines); hunks: -227,8 +227,8 @@ def deepgemm_w8a8_block_fp8_linear_with_fallback(; symbols: deepgemm_w8a8_block_fp8_linear_with_fallback, touching `deepgemm_w8a8_block_fp8_linear_with_fallback`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2 (4 lines); hunks: -227,8 +227,8 @@ def deepgemm_w8a8_block_fp8_linear_with_fallback(; symbols: deepgemm_w8a8_block_fp8_linear_with_fallback
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -227,8 +227,8 @@ def deepgemm_w8a8_block_fp8_linear_with_fallback(
-    # TODO: add more robust shape check here
-    shape_supported = weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0
+    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737
+    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6220 - Fuse routed scaling factor in topk_reduce kernel

- Link: https://github.com/sgl-project/sglang/pull/6220
- Status/date: merged / 2025-06-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +331/-9, 490 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse routed scaling factor in topk_reduce kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; PR body summary: gsm8k acc in h200 benchmark in H200 command: Micro benchmark in h200 micro benchmark in h200: And I use the below code to select triton or `torch.compile` kernel:.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +124/-8 (132 lines); hunks: -1155,6 +1155,7 @@ def inplace_fused_experts(; -1177,6 +1178,8 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts, touching `inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts`; `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -346,7 +346,7 @@ def forward_normal(self, hidden_states: torch.Tensor) -> tor...; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: -225,6 +225,7 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu, touching `forward_cuda, forward_cpu`; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: -411,4 +411,5 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +124/-8 (132 lines); hunks: -1155,6 +1155,7 @@ def inplace_fused_experts(; -1177,6 +1178,8 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -346,7 +346,7 @@ def forward_normal(self, hidden_states: torch.Tensor) -> tor...; symbols: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: -225,6 +225,7 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: -411,4 +411,5 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-0 (1 lines); hunks: -317,6 +317,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -1155,6 +1155,7 @@ def inplace_fused_experts(
+    routed_scaling_factor: Optional[float] = None,
@@ -1177,6 +1178,8 @@ def inplace_fused_experts(
+        False,
+        routed_scaling_factor,
@@ -1200,6 +1203,7 @@ def inplace_fused_experts_fake(
+    routed_scaling_factor: Optional[float] = None,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -346,7 +346,7 @@ def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        final_hidden_states *= self.routed_scaling_factor
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -225,6 +225,7 @@ def forward_cuda(
+                routed_scaling_factor=routed_scaling_factor,
diff -- python/sglang/srt/layers/quantization/blockwise_int8.py
@@ -411,4 +411,5 @@ def apply(
+            routed_scaling_factor=routed_scaling_factor,
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +124/-8; `python/sglang/srt/models/deepseek_v2.py` modified +1/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/blockwise_int8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6853 - [DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model

- Link: https://github.com/sgl-project/sglang/pull/6853
- Status/date: merged / 2025-06-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `c2c4f57f6311`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +386/-13, 506 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepseekR1-FP4] Add Support for nvidia/DeepSeekR1-FP4 model"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Adds Model support for DeepSeek R1 FP4 Model. (Functional enablement - kernel optimizations underway) Adds ModelOptFP4FusedMoEMethod and CutlassMoEParams dataclass to initialize....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +33/-5 (38 lines); hunks: -1746,7 +1746,7 @@ def determine_num_fused_shared_experts(; -1926,6 +1926,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: determine_num_fused_shared_experts, get_input_embeddings, post_load_weights, load_weights, touching `determine_num_fused_shared_experts, get_input_embeddings, post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +33/-5 (38 lines); hunks: -1746,7 +1746,7 @@ def determine_num_fused_shared_experts(; -1926,6 +1926,7 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: determine_num_fused_shared_experts, get_input_embeddings, post_load_weights, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1746,7 +1746,7 @@ def determine_num_fused_shared_experts(
-                    "Deepseek V3/R1 with fp8 can use shared experts fusion optimization when SM version >=90. Shared experts fusion optimization is enabled.",
+                    "Deepseek V3/R1 with fp8/fp4 can use shared experts fusion optimization when SM version >=90. Shared experts fusion optimization is enabled.",
@@ -1926,6 +1926,7 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
@@ -1982,6 +1983,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+                elif self.quant_config.get_name() == "modelopt_fp4":
+                    suffix_list = [
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +33/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6970 - Fuse routed scaling factor in deepseek

- Link: https://github.com/sgl-project/sglang/pull/6970
- Status/date: merged / 2025-06-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +338/-15, 498 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse routed scaling factor in deepseek"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +130/-14 (144 lines); hunks: -1155,6 +1155,7 @@ def inplace_fused_experts(; -1177,6 +1178,8 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts, touching `inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts`; `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -346,7 +346,8 @@ def forward_normal(self, hidden_states: torch.Tensor) -> tor...; symbols: forward_normal, touching `forward_normal`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: -225,6 +225,7 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu, touching `forward_cuda, forward_cpu`; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: -411,4 +411,5 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +130/-14 (144 lines); hunks: -1155,6 +1155,7 @@ def inplace_fused_experts(; -1177,6 +1178,8 @@ def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -346,7 +346,8 @@ def forward_normal(self, hidden_states: torch.Tensor) -> tor...; symbols: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: -225,6 +225,7 @@ def forward_cuda(; symbols: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: -411,4 +411,5 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-0 (1 lines); hunks: -317,6 +317,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
@@ -1155,6 +1155,7 @@ def inplace_fused_experts(
+    routed_scaling_factor: Optional[float] = None,
@@ -1177,6 +1178,8 @@ def inplace_fused_experts(
+        False,
+        routed_scaling_factor,
@@ -1200,6 +1203,7 @@ def inplace_fused_experts_fake(
+    routed_scaling_factor: Optional[float] = None,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -346,7 +346,8 @@ def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        final_hidden_states *= self.routed_scaling_factor
+        if not _is_cuda:
+            final_hidden_states *= self.routed_scaling_factor
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -225,6 +225,7 @@ def forward_cuda(
+                routed_scaling_factor=routed_scaling_factor,
diff -- python/sglang/srt/layers/quantization/blockwise_int8.py
@@ -411,4 +411,5 @@ def apply(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +130/-14; `python/sglang/srt/models/deepseek_v2.py` modified +2/-1; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +1/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/blockwise_int8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7005 - FlowMLA: zero-overhead DP MLA throughput boost via memory optimization

- Link: https://github.com/sgl-project/sglang/pull/7005
- Status/date: open / 2025-06-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +601/-19, 856 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "FlowMLA: zero-overhead DP MLA throughput boost via memory optimization"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; PR body summary: This PR is part of our upcoming work FlowMLA @antgroup. Targeting MLA in Data Parallel scenarios, we enable serving DeepSeek-like models on fewer devices through seamless memory....
- Key implementation: `python/sglang/srt/layers/linear.py` modified +107/-8 (115 lines); hunks: -7,6 +7,7; -17,6 +18,8; symbols: adjust_marlin_shard, create_weights, __init__, touching `adjust_marlin_shard, create_weights, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +97/-6 (103 lines); hunks: -23,6 +23,7; -124,6 +125,25; symbols: PrefetchStreamManager, initialize_stream, get_stream, AttnForwardMethod, touching `PrefetchStreamManager, initialize_stream, get_stream`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +19/-2 (21 lines); hunks: -287,6 +287,15 @@ def __init__(self, model_runner: ModelRunner):; -556,8 +565,16 @@ def run_once():; symbols: __init__, run_once, touching `__init__, run_once`; `python/sglang/srt/layers/quantization/fp8.py` modified +9/-2 (11 lines); hunks: -219,6 +219,7 @@ def create_weights(; -263,8 +264,14 @@ def create_weights(; symbols: create_weights, touching `create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/linear.py` modified +107/-8 (115 lines); hunks: -7,6 +7,7; -17,6 +18,8; symbols: adjust_marlin_shard, create_weights, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +97/-6 (103 lines); hunks: -23,6 +23,7; -124,6 +125,25; symbols: PrefetchStreamManager, initialize_stream, get_stream, AttnForwardMethod
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +19/-2 (21 lines); hunks: -287,6 +287,15 @@ def __init__(self, model_runner: ModelRunner):; -556,8 +565,16 @@ def run_once():; symbols: __init__, run_once
  - `python/sglang/srt/layers/quantization/fp8.py` modified +9/-2 (11 lines); hunks: -219,6 +219,7 @@ def create_weights(; -263,8 +264,14 @@ def create_weights(; symbols: create_weights
  - `vtensor/vtensor.cpp` added +196/-0 (196 lines); hunks: -0,0 +1,196
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/linear.py
@@ -7,6 +7,7 @@
+import vTensor
@@ -17,6 +18,8 @@
+from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
+from sglang.srt.distributed.utils import StatelessProcessGroup
@@ -52,6 +55,8 @@
+vtensor_pynccl = None
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -23,6 +23,7 @@
+import vTensor
@@ -124,6 +125,25 @@
+class PrefetchStreamManager:
+    _stream_odd = None
+    _stream_even = None
+    @classmethod
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -287,6 +287,15 @@ def __init__(self, model_runner: ModelRunner):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/linear.py` modified +107/-8; `python/sglang/srt/models/deepseek_v2.py` modified +97/-6; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +19/-2; `python/sglang/srt/layers/quantization/fp8.py` modified +9/-2
  - other: `vtensor/vtensor.cpp` added +196/-0; `vtensor/vtensor.h` added +101/-0; `vtensor/setup.py` added +28/-0
  - tests: `vtensor/test/test_vtensor.py` added +25/-0
- Risk and verification: The diff ships test coverage in `vtensor/test/test_vtensor.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7146 - Support new DeepGEMM format in per token group quant

- Link: https://github.com/sgl-project/sglang/pull/7146
- Status/date: merged / 2025-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +92/-44, 216 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support new DeepGEMM format in per token group quant"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h`; no usable PR-body summary.
- Key implementation: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +83/-40 (123 lines); hunks: -16,11 +16,16 @@ __device__ __forceinline__ float GroupReduceMax(float val, c...; -39,15 +44,24 @@ __global__ void per_token_group_quant_8bit_kernel(; `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +4/-1 (5 lines); hunks: -255,7 +255,10 @@ def sglang_per_token_group_quant_8bit(; symbols: sglang_per_token_group_quant_8bit, touching `sglang_per_token_group_quant_8bit`; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: -175,7 +175,8 @@ void sgl_per_token_group_quant_fp8(; `sgl-kernel/python/sgl_kernel/gemm.py` modified +2/-1 (3 lines); hunks: -90,9 +90,10 @@ def sgl_per_token_group_quant_fp8(; symbols: sgl_per_token_group_quant_fp8, touching `sgl_per_token_group_quant_fp8`.
- Code diff details:
  - `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +83/-40 (123 lines); hunks: -16,11 +16,16 @@ __device__ __forceinline__ float GroupReduceMax(float val, c...; -39,15 +44,24 @@ __global__ void per_token_group_quant_8bit_kernel(
  - `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +4/-1 (5 lines); hunks: -255,7 +255,10 @@ def sglang_per_token_group_quant_8bit(; symbols: sglang_per_token_group_quant_8bit
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: -175,7 +175,8 @@ void sgl_per_token_group_quant_fp8(
  - `sgl-kernel/python/sgl_kernel/gemm.py` modified +2/-1 (3 lines); hunks: -90,9 +90,10 @@ def sgl_per_token_group_quant_fp8(; symbols: sgl_per_token_group_quant_fp8
  - `sgl-kernel/csrc/common_extension.cc` modified +1/-1 (2 lines); hunks: -116,7 +116,7 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu
@@ -16,11 +16,16 @@ __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
-template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR = false>
+template <
+    typename T,
+    typename DST_DTYPE,
+    bool IS_COLUMN_MAJOR = false,
+    bool SCALE_UE8M0 = false,
diff -- sgl-kernel/tests/test_per_token_group_quant_8bit.py
@@ -255,7 +255,10 @@ def sglang_per_token_group_quant_8bit(
-        sgl_per_token_group_quant_fp8(x, x_q, x_s, group_size, eps, fp8_min, fp8_max)
+        scale_ue8m0 = False  # TODO also test true
+        sgl_per_token_group_quant_fp8(
+            x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
+        )
diff -- sgl-kernel/include/sgl_kernel_ops.h
@@ -175,7 +175,8 @@ void sgl_per_token_group_quant_fp8(
-    double fp8_max);
```

- Reviewed files:
  - other: `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +83/-40; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1; `sgl-kernel/python/sgl_kernel/gemm.py` modified +2/-1; `sgl-kernel/csrc/common_extension.cc` modified +1/-1
  - tests: `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_per_token_group_quant_8bit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7155 - Support new DeepGEMM format in per token group quant (part 2: srt)

- Link: https://github.com/sgl-project/sglang/pull/7155
- Status/date: merged / 2025-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +19/-4, 59 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support new DeepGEMM format in per token group quant (part 2: srt)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/entrypoints/engine.py`, `python/pyproject.toml`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +17/-2 (19 lines); hunks: -280,15 +280,28 @@ def sglang_per_token_group_quant_fp8(; -309,7 +322,9 @@ def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8, touching `sglang_per_token_group_quant_fp8`; `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -605,7 +605,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config, touching `_set_envs_and_config`; `python/pyproject.toml` modified +1/-1 (2 lines); hunks: -49,7 +49,7 @@ runtime_common = [.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +17/-2 (19 lines); hunks: -280,15 +280,28 @@ def sglang_per_token_group_quant_fp8(; -309,7 +322,9 @@ def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -605,7 +605,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunks: -49,7 +49,7 @@ runtime_common = [
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -280,15 +280,28 @@ def sglang_per_token_group_quant_fp8(
+    scale_ue8m0: bool = False,
-    if column_major_scales:
+    if scale_ue8m0:
+        assert column_major_scales and scale_tma_aligned
+        x_q_mn, x_q_k = x.shape
+        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
diff -- python/sglang/srt/entrypoints/engine.py
@@ -605,7 +605,7 @@ def _set_envs_and_config(server_args: ServerArgs):
-            "0.1.7",
+            "0.1.8.post1",
diff -- python/pyproject.toml
@@ -49,7 +49,7 @@ runtime_common = [
-    "sgl-kernel==0.1.7",
+    "sgl-kernel==0.1.8.post1",
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +17/-2; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/pyproject.toml` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7156 - Re-quantize DeepSeek model weights to support DeepGEMM new input format

- Link: https://github.com/sgl-project/sglang/pull/7156
- Status/date: merged / 2025-06-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +125/-0, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Re-quantize DeepSeek model weights to support DeepGEMM new input format"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +61/-0 (61 lines); hunks: -4,6 +4,7; -390,6 +391,66 @@ def block_quant_dequant(; symbols: block_quant_dequant, requant_weight_ue8m0_inplace, _requant_weight_ue8m0, _transform_scale, touching `block_quant_dequant, requant_weight_ue8m0_inplace, _requant_weight_ue8m0`; `python/sglang/srt/models/deepseek_v2.py` modified +56/-0 (56 lines); hunks: -66,6 +66,7; -1931,6 +1932,61 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights, _weight_requant_ue8m0, load_weights, touching `post_load_weights, _weight_requant_ue8m0, load_weights`; `python/sglang/math_utils.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: align, ceil_div, touching `align, ceil_div`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +61/-0 (61 lines); hunks: -4,6 +4,7; -390,6 +391,66 @@ def block_quant_dequant(; symbols: block_quant_dequant, requant_weight_ue8m0_inplace, _requant_weight_ue8m0, _transform_scale
  - `python/sglang/srt/models/deepseek_v2.py` modified +56/-0 (56 lines); hunks: -66,6 +66,7; -1931,6 +1932,61 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights, _weight_requant_ue8m0, load_weights
  - `python/sglang/math_utils.py` added +8/-0 (8 lines); hunks: -0,0 +1,8; symbols: align, ceil_div
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -4,6 +4,7 @@
+from sglang.math_utils import align
@@ -390,6 +391,66 @@ def block_quant_dequant(
+def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
+    assert isinstance(weight, torch.nn.Parameter)
+    assert isinstance(weight_scale_inv, torch.nn.Parameter)
+    weight.data, weight_scale_inv.data = _requant_weight_ue8m0(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -66,6 +66,7 @@
+    requant_weight_ue8m0_inplace,
@@ -1931,6 +1932,61 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
+        if False:  # TODO (pr-chain)
+            self._weight_requant_ue8m0()
+    def _weight_requant_ue8m0(self):
+        weight_block_size = self.quant_config.weight_block_size
diff -- python/sglang/math_utils.py
@@ -0,0 +1,8 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +61/-0; `python/sglang/srt/models/deepseek_v2.py` modified +56/-0; `python/sglang/math_utils.py` added +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/math_utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7150 - Refactor DeepGEMM integration

- Link: https://github.com/sgl-project/sglang/pull/7150
- Status/date: merged / 2025-06-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +207/-147, 685 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor DeepGEMM integration"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py` renamed +22/-76 (98 lines); hunks: -5,33 +5,24; -52,8 +43,10 @@ def get_enable_jit_deepgemm():; symbols: get_enable_jit_deepgemm, DeepGemmKernelHelper, _compile_warning_1, touching `get_enable_jit_deepgemm, DeepGemmKernelHelper, _compile_warning_1`; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: grouped_gemm_nt_f8f8bf16_masked, grouped_gemm_nt_f8f8bf16_contig, gemm_nt_f8f8bf16, update_deep_gemm_config, touching `grouped_gemm_nt_f8f8bf16_masked, grouped_gemm_nt_f8f8bf16_contig, gemm_nt_f8f8bf16`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +35/-32 (67 lines); hunks: -1,30 +1,11; -45,17 +26,28; symbols: create_weights, __init__, forward, forward_deepgemm_contiguous, touching `create_weights, __init__, forward`; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` added +26/-0 (26 lines); hunks: -0,0 +1,26; symbols: _compute_enable_deep_gemm, touching `_compute_enable_deep_gemm`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py` renamed +22/-76 (98 lines); hunks: -5,33 +5,24; -52,8 +43,10 @@ def get_enable_jit_deepgemm():; symbols: get_enable_jit_deepgemm, DeepGemmKernelHelper, _compile_warning_1
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: grouped_gemm_nt_f8f8bf16_masked, grouped_gemm_nt_f8f8bf16_contig, gemm_nt_f8f8bf16, update_deep_gemm_config
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +35/-32 (67 lines); hunks: -1,30 +1,11; -45,17 +26,28; symbols: create_weights, __init__, forward, forward_deepgemm_contiguous
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` added +26/-0 (26 lines); hunks: -0,0 +1,26; symbols: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +6/-10 (16 lines); hunks: -23,7 +23,8; -44,10 +45,6; symbols: is_fp8_fnuz, deep_gemm_fp8_fp8_bf16_nt, deep_gemm_fp8_fp8_bf16_nt_fake, w8a8_block_fp8_matmul_deepgemm
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py
@@ -5,33 +5,24 @@
-import torch
+from sglang.srt.layers.quantization.deep_gemm_wrapper.configurer import (
+    DEEPGEMM_V202506,
+    ENABLE_JIT_DEEPGEMM,
+)
-from sglang.srt.utils import get_bool_env_var, get_device_sm, get_int_env_var, is_cuda
diff -- python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py
@@ -0,0 +1,95 @@
+import logging
+from contextlib import contextmanager
+from typing import Tuple
+import torch
+from sglang.srt.layers.quantization.deep_gemm_wrapper import compile_utils
+from sglang.srt.layers.quantization.deep_gemm_wrapper.configurer import (
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -1,30 +1,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py` renamed +22/-76; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` added +95/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +35/-32; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` added +26/-0; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +6/-10; `python/sglang/srt/model_executor/model_runner.py` modified +6/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7172 - Support new DeepGEMM

- Link: https://github.com/sgl-project/sglang/pull/7172
- Status/date: merged / 2025-06-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +59/-19, 161 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support new DeepGEMM"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` modified +18/-8 (26 lines); hunks: -16,14 +16,24; symbols: grouped_gemm_nt_f8f8bf16_masked, touching `grouped_gemm_nt_f8f8bf16_masked`; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +20/-3 (23 lines); hunks: -765,16 +765,33 @@ def prepare_block_fp8_matmul_inputs(; symbols: prepare_block_fp8_matmul_inputs, touching `prepare_block_fp8_matmul_inputs`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunks: -1231,14 +1231,21 @@ def forward_deepgemm_masked(; symbols: forward_deepgemm_masked, touching `forward_deepgemm_masked`; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` modified +7/-1 (8 lines); hunks: -21,6 +21,12 @@ def _compute_enable_deep_gemm():; symbols: _compute_enable_deep_gemm, touching `_compute_enable_deep_gemm`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` modified +18/-8 (26 lines); hunks: -16,14 +16,24; symbols: grouped_gemm_nt_f8f8bf16_masked
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +20/-3 (23 lines); hunks: -765,16 +765,33 @@ def prepare_block_fp8_matmul_inputs(; symbols: prepare_block_fp8_matmul_inputs
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunks: -1231,14 +1231,21 @@ def forward_deepgemm_masked(; symbols: forward_deepgemm_masked
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` modified +7/-1 (8 lines); hunks: -21,6 +21,12 @@ def _compute_enable_deep_gemm():; symbols: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-4 (5 lines); hunks: -12,6 +12,7; -518,10 +519,6 @@ def fused_moe_kernel(; symbols: fused_moe_kernel, ceil_div, moe_align_block_size_stage1
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py
@@ -16,14 +16,24 @@
-    from deep_gemm import gemm_fp8_fp8_bf16_nt as _gemm_nt_f8f8bf16_raw
-    from deep_gemm import get_col_major_tma_aligned_tensor
-    from deep_gemm import (
-        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous as _grouped_gemm_nt_f8f8bf16_contig_raw,
-    )
-    from deep_gemm import (
diff -- python/sglang/srt/layers/quantization/fp8_kernel.py
@@ -765,16 +765,33 @@ def prepare_block_fp8_matmul_inputs(
-    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
+    if As.dtype == torch.float:
+        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
+    elif Bs.dtype == torch.int:
+        assert (
+            triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1]
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -1231,14 +1231,21 @@ def forward_deepgemm_masked(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` modified +18/-8; `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +20/-3; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1; `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` modified +7/-1; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-4; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7181 - [fix] fix dsv3 weight loader tqdm and simplify shared experts fusion

- Link: https://github.com/sgl-project/sglang/pull/7181
- Status/date: merged / 2025-06-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `ed54bf9d19f1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-96, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] fix dsv3 weight loader tqdm and simplify shared experts fusion"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Using `weights = [w for w in weights_list if w[0] not in names_to_remove]` to update `weights` will change it from async weight iterator to a list. Thats will make the tqdm look....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-96 (107 lines); hunks: -1033,7 +1033,7 @@ def forward_absorb_core(; -2008,101 +2008,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: forward_absorb_core, load_weights, touching `forward_absorb_core, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-96 (107 lines); hunks: -1033,7 +1033,7 @@ def forward_absorb_core(; -2008,101 +2008,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: forward_absorb_core, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1033,7 +1033,7 @@ def forward_absorb_core(
-            deep_gemm_grouped_gemm_nt_f8f8bf16_masked(
+            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
@@ -2008,101 +2008,6 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-        if self.num_fused_shared_experts > 0:
-            assert self.num_fused_shared_experts == 1
-            weights_list = list(weights)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-96
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7180 - [fix] fix determine_num_fused_shared_experts

- Link: https://github.com/sgl-project/sglang/pull/7180
- Status/date: merged / 2025-06-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-47, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] fix determine_num_fused_shared_experts"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +29/-47 (76 lines); hunks: -1709,53 +1709,35 @@ def routed_experts_weights_of_layer(self):; symbols: routed_experts_weights_of_layer, determine_num_fused_shared_experts, get_input_embeddings, touching `routed_experts_weights_of_layer, determine_num_fused_shared_experts, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +29/-47 (76 lines); hunks: -1709,53 +1709,35 @@ def routed_experts_weights_of_layer(self):; symbols: routed_experts_weights_of_layer, determine_num_fused_shared_experts, get_input_embeddings
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1709,53 +1709,35 @@ def routed_experts_weights_of_layer(self):
-        self.num_fused_shared_experts = (
-            0
-            if global_server_args_dict["disable_shared_experts_fusion"]
-            else self.config.n_shared_experts
-        )
-        if self.num_fused_shared_experts > 0:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +29/-47
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6081 - feat: mtp support dp-attention

- Link: https://github.com/sgl-project/sglang/pull/6081
- Status/date: merged / 2025-06-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +636/-146, 1516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: mtp support dp-attention"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/layers/attention/triton_backend.py`; PR body summary: mtp support dp-attention - implemented MTP for DP-attention, also fixed related bugs #4783 #4847 . - Enabled CUDA Graph support for both target and draft models at dp-attention.....
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +18/-26 (44 lines); hunks: -22,7 +22,6; -77,6 +76,7 @@ def forward(; symbols: forward, __init__, touching `forward, __init__`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +21/-15 (36 lines); hunks: -242,13 +242,13 @@ def __init__(self, model_runner: ModelRunner):; -323,12 +323,15 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, can_run, capture_one_batch_size, replay_prepare, touching `__init__, can_run, capture_one_batch_size`; `python/sglang/srt/layers/attention/triton_backend.py` modified +19/-11 (30 lines); hunks: -261,6 +261,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -335,24 +336,27 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph, touching `init_forward_metadata, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +18/-5 (23 lines); hunks: -320,17 +320,30 @@ def init_new(; symbols: init_new, touching `init_new`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +18/-26 (44 lines); hunks: -22,7 +22,6; -77,6 +76,7 @@ def forward(; symbols: forward, __init__
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +21/-15 (36 lines); hunks: -242,13 +242,13 @@ def __init__(self, model_runner: ModelRunner):; -323,12 +323,15 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, can_run, capture_one_batch_size, replay_prepare
  - `python/sglang/srt/layers/attention/triton_backend.py` modified +19/-11 (30 lines); hunks: -261,6 +261,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -335,24 +336,27 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +18/-5 (23 lines); hunks: -320,17 +320,30 @@ def init_new(; symbols: init_new
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-5 (13 lines); hunks: -262,11 +262,14 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -285,7 +288,7 @@ def init_cuda_graph_state(; symbols: init_forward_metadata, init_cuda_graph_state, call_fn
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -22,7 +22,6 @@
-from sglang.srt.layers.linear import ReplicatedLinear
@@ -77,6 +76,7 @@ def forward(
@@ -90,15 +90,16 @@ def forward(
-        hidden_states = self.eh_proj(
-            torch.cat(
-                (
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -242,13 +242,13 @@ def __init__(self, model_runner: ModelRunner):
-        if global_server_args_dict["attention_backend"] == "flashmla":
-            self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs)
-        else:
-            self.model_runner.attn_backend.init_cuda_graph_state(self.max_num_token)
+        self.model_runner.attn_backend.init_cuda_graph_state(
+            self.max_bs, self.max_num_token
diff -- python/sglang/srt/layers/attention/triton_backend.py
@@ -261,6 +261,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +18/-26; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +21/-15; `python/sglang/srt/layers/attention/triton_backend.py` modified +19/-11; `python/sglang/srt/model_executor/forward_batch_info.py` modified +18/-5; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +8/-5; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +6/-3
- Risk and verification: The diff ships test coverage in `test/srt/test_dp_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7164 - Fix Deepseek R1 0528 FP4 tensor name mismatch issue during weights loading.

- Link: https://github.com/sgl-project/sglang/pull/7164
- Status/date: merged / 2025-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `8c16da334e93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +10/-7, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Deepseek R1 0528 FP4 tensor name mismatch issue during weights loading."; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This PR tries to fix a weights loading issue for Deepseek R1 0528 Fp4 from Nvidia. To reproduce the issue: (i'm using blackwell image) If we look into the weight file and model....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-6 (11 lines); hunks: -1919,6 +1919,8 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; -2151,12 +2153,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: post_load_weights, load_weights, touching `post_load_weights, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-6 (11 lines); hunks: -1919,6 +1919,8 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; -2151,12 +2153,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: post_load_weights, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1919,6 +1919,8 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
+            and hasattr(self.quant_config, "weight_block_size")
+            and self.quant_config.weight_block_size is not None
@@ -2151,12 +2153,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-                            if any(scale in name for scale in ["k_scale", "v_scale"]):
-                                name = name.replace("_proj", "attn_mqa")
-                            else:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7286 - fix: resolve b200 dsv3 mtp issue

- Link: https://github.com/sgl-project/sglang/pull/7286
- Status/date: merged / 2025-06-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `4f204db57cf9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: resolve b200 dsv3 mtp issue"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1932,6 +1932,8 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights, _weight_requant_ue8m0, touching `post_load_weights, _weight_requant_ue8m0`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -1932,6 +1932,8 @@ def post_load_weights(self, is_nextn=False, weight_names=N...; symbols: post_load_weights, _weight_requant_ue8m0
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1932,6 +1932,8 @@ def post_load_weights(self, is_nextn=False, weight_names=None):
+        if self.config.architectures[0] == "DeepseekV3ForCausalLMNextN":
+            return
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7371 - Fix judgment condition for enabling Deepseek V3/R1 shared expert fusion optimization

- Link: https://github.com/sgl-project/sglang/pull/7371
- Status/date: merged / 2025-06-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `a06912ad8b35`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix judgment condition for enabling Deepseek V3/R1 shared expert fusion optimization"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: ref: https://github.com/sgl-project/sglang/pull/7180 Modify the compatibility version judgment conditions. The following results were tested in the A800 environment: - main - pr.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -1731,12 +1731,12 @@ def determine_num_fused_shared_experts(; -2040,7 +2040,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights, touching `determine_num_fused_shared_experts, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -1731,12 +1731,12 @@ def determine_num_fused_shared_experts(; -2040,7 +2040,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1731,12 +1731,12 @@ def determine_num_fused_shared_experts(
-            or torch.cuda.get_device_capability("cuda") < (9, 0)
+            or torch.cuda.get_device_capability("cuda") < (8, 0)
-            disable_reason = "Only Deepseek V3/R1 on NV-platform with capability >= 90 can use shared experts fusion optimization."
+            disable_reason = "Only Deepseek V3/R1 on NV-platform with capability >= 80 can use shared experts fusion optimization."
@@ -2040,7 +2040,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-            logger.info("Shared experts fusion optimization enabled.")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7481 - [perf] slightly imporve DeepSeek-R1-FP4 TP8

- Link: https://github.com/sgl-project/sglang/pull/7481
- Status/date: merged / 2025-06-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `bdbb8d009ae1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf] slightly imporve DeepSeek-R1-FP4 TP8"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Tuned dual stream kernel order, 79->81.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +4/-2 (6 lines); hunks: -362,12 +362,14 @@ def forward(; symbols: forward, forward_normal_dual_stream, touching `forward, forward_normal_dual_stream`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-2 (6 lines); hunks: -362,12 +362,14 @@ def forward(; symbols: forward, forward_normal_dual_stream
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -362,12 +362,14 @@ def forward(
+        # router_logits: (num_tokens, n_experts)
+        router_logits = self.gate(hidden_states)
-            # router_logits: (num_tokens, n_experts)
-            router_logits = self.gate(hidden_states)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7376 - Fix MTP with Deepseek R1 Fp4

- Link: https://github.com/sgl-project/sglang/pull/7376
- Status/date: merged / 2025-06-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `15f34013432f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-1, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix MTP with Deepseek R1 Fp4"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: https://github.com/sgl-project/sglang/issues/7365 Update quant config to None for MTP module in Deepseek modelopt fp4. It's not quantized. See https://github.com/NVIDIA/TensorRT....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: -2201,7 +2201,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -2232,6 +2232,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: -2201,7 +2201,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; -2232,6 +2232,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2201,7 +2201,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
-                            if (
+                            if self.quant_config is not None and (
@@ -2232,6 +2232,13 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+                                    break
+                        if name not in params_dict:
+                            # modelopt ckpt contains not needed weights for MTP module:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7635 - Apply dsv3_fused_a_gemm kernel

- Link: https://github.com/sgl-project/sglang/pull/7635
- Status/date: merged / 2025-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `00aec6ad6c34`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-2, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply dsv3_fused_a_gemm kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Apply `dsv3_fused_a_gemm kernel` in https://github.com/sgl-project/sglang/pull/7630. Profile and accuracy test are already listed in the previous PR. Dependency: release new ver....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +18/-2 (20 lines); hunks: -96,6 +96,7; -112,7 +113,7; symbols: __init__, forward_absorb_prepare, touching `__init__, forward_absorb_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-2 (20 lines); hunks: -96,6 +96,7; -112,7 +113,7; symbols: __init__, forward_absorb_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -96,6 +96,7 @@
+    get_device_sm,
@@ -112,7 +113,7 @@
-    from sgl_kernel import awq_dequantize, bmm_fp8, merge_state_v2
+    from sgl_kernel import awq_dequantize, bmm_fp8, dsv3_fused_a_gemm, merge_state_v2
@@ -875,6 +876,15 @@ def __init__(
+        self.use_min_latency_fused_a_gemm = (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +18/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7677 - Apply dsv3 router gemm kernel for deepseek-r1 fp4

- Link: https://github.com/sgl-project/sglang/pull/7677
- Status/date: merged / 2025-07-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `88f484ce4c73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-3, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Apply dsv3 router gemm kernel for deepseek-r1 fp4"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Apply kernel added in #7627 to dpsk-r1 fp4 model. Currently we have to convert the dtype of router logits back to bf16, otherwise there will be cuda illegal memory bug. We shoul....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +21/-3 (24 lines); hunks: -111,9 +111,16; -225,7 +232,18 @@ def forward(self, hidden_states):; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-3 (24 lines); hunks: -111,9 +111,16; -225,7 +232,18 @@ def forward(self, hidden_states):; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -111,9 +111,16 @@
+_device_sm = get_device_sm()
-    from sgl_kernel import awq_dequantize, bmm_fp8, dsv3_fused_a_gemm, merge_state_v2
+    from sgl_kernel import (
+        awq_dequantize,
+        bmm_fp8,
+        dsv3_fused_a_gemm,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +21/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7735 - fix awq and dsv3 fused gemm compatible

- Link: https://github.com/sgl-project/sglang/pull/7735
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `84f2e4a0f847`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-2, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix awq and dsv3 fused gemm compatible"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: added a check to avoid compatibility issues when loading AWQ weights with dsv3 fused gemm enabled related issue https://github.com/sgl-project/sglang/issues/7728.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -318,11 +318,17 @@ def __init__(; -876,8 +882,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -318,11 +318,17 @@ def __init__(; -876,8 +882,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -318,11 +318,17 @@ def __init__(
+            is_packed_weight = (
+                self.shared_experts.gate_up_proj.quant_method.quant_config.get_name()
+                in ["awq", "moe_wna16"]
+            )
-                self.shared_experts.gate_up_proj.weight.dtype == torch.int8
+                not is_packed_weight
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +15/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7738 - fix dsv3 fused proj check

- Link: https://github.com/sgl-project/sglang/pull/7738
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `b58226510f7d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-14, 61 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix dsv3 fused proj check"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: fix dsv3 fused proj check in https://github.com/sgl-project/sglang/pull/7735.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +15/-14 (29 lines); hunks: -336,10 +336,12 @@ def __init__(; -891,21 +893,20 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-14 (29 lines); hunks: -336,10 +336,12 @@ def __init__(; -891,21 +893,20 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -336,10 +336,12 @@ def __init__(
-            is_packed_weight = (
-                self.shared_experts.gate_up_proj.quant_method.quant_config.get_name()
-                in ["awq", "moe_wna16"]
-            )
+            is_packed_weight = hasattr(
+                self.shared_experts.gate_up_proj.quant_method, "quant_config"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +15/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7750 - [fix] fix dsv3_router_gemm filter

- Link: https://github.com/sgl-project/sglang/pull/7750
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `ac49dac00946`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[fix] fix dsv3_router_gemm filter"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: fix ci on main: https://github.com/sgl-project/sglang/actions/runs/16045971205/job/45277302233#step:5:3827.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -233,7 +233,8 @@ def forward(self, hidden_states):; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: -233,7 +233,8 @@ def forward(self, hidden_states):; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -233,7 +233,8 @@ def forward(self, hidden_states):
-            hidden_states.shape[0] < 4
+            _is_cuda
+            and hidden_states.shape[0] < 4
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7793 - fix: disable dsv3_router_gemm in dsv3_nextn

- Link: https://github.com/sgl-project/sglang/pull/7793
- Status/date: merged / 2025-07-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `54411f6afa27`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-1, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: disable dsv3_router_gemm in dsv3_nextn"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: It has been confirmed that using dsv3_router_gemm leads to a degradation in the acc_len of the dsr1-fp4 nextn module.: https://github.com/sgl-project/sglang/pull/7711/files#diff....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -210,8 +210,10 @@ def __init__(; -233,8 +235,10 @@ def forward(self, hidden_states):; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -210,8 +210,10 @@ def __init__(; -233,8 +235,10 @@ def forward(self, hidden_states):; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -210,8 +210,10 @@ def __init__(
+        is_nextn: bool = False,
+        self.is_nextn = is_nextn
@@ -233,8 +235,10 @@ def forward(self, hidden_states):
+        # NOTE: For some unknown reason, router_gemm seems degrade accept length.
+            and not self.is_nextn
@@ -258,6 +262,7 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7762 - feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode

- Link: https://github.com/sgl-project/sglang/pull/7762
- Status/date: merged / 2025-07-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `cb9d91ea8a71`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1006/-9, 1203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This PR supports running DeepSeek-R1-W4AFP8 model with ep-moe mode(deepep mode support is on the way~) Due to the reduced space required for model weights and decreased bandwidt....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -2363,6 +2363,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -2363,6 +2363,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2363,6 +2363,12 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        if self.quant_config and self.quant_config.get_name() == "w4afp8":
+            expert_params_mapping += (
+                get_moe_impl_class().make_expert_input_scale_params_mapping(
+                    num_experts=self.config.n_routed_experts
+                )
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_cutlass_w4a8_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7562 - Fix incomplete tool call capture issue in streaming response of DeepSeek-V3 when enable MTP

- Link: https://github.com/sgl-project/sglang/pull/7562
- Status/date: merged / 2025-07-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv3_detector.py`; associated commits `624a3b8d1f10`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +90/-1, 104 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix incomplete tool call capture issue in streaming response of DeepSeek-V3 when enable MTP"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv3_detector.py`; PR body summary: When using DeepSeeker V3 and enabling MTP and tool call parsing, we found that there are have some unexpected string in the streaming reponse like " \n ". The problem is the reg....
- Key implementation: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1 (2 lines); hunks: -98,7 +98,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1 (2 lines); hunks: -98,7 +98,7 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv3_detector.py
@@ -98,7 +98,7 @@ def parse_streaming_increment(
-                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n'''json\n(.*)",
+                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n'''json\n(.*)\n'''.*",
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8311 - Support DeepSeek-R1 w4a8 low latency deepep

- Link: https://github.com/sgl-project/sglang/pull/8311
- Status/date: closed / 2025-07-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 219 files, +16292/-3879, 27289 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-R1 w4a8 low latency deepep"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/glm4_moe.py`, `test/srt/test_function_call_parser.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; PR body summary: Follow https://github.com/sgl-project/sglang/pull/8247 https://github.com/sgl-project/sglang/pull/7762. Support deepep low latency mode for DeepSeek-R1 w4a8 model add forward_cu....
- Key implementation: `python/sglang/srt/models/glm4_moe.py` added +1035/-0 (1035 lines); hunks: -0,0 +1,1035; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention, touching `Glm4MoeMLP, __init__, forward`; `test/srt/test_function_call_parser.py` modified +639/-0 (639 lines); hunks: -6,10 +6,12; -507,7 +509,9 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf, touching `setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +211/-203 (414 lines); hunks: -47,13 +47,15; -64,6 +66,13; symbols: forward, EPMoE, _get_tile_tokens_dim, __init__, touching `forward, EPMoE, _get_tile_tokens_dim`; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +360/-4 (364 lines); hunks: -16,6 +16,20; -73,9 +87,13 @@ def test_function_calling_format(self):; symbols: TestOpenAIServerFunctionCalling, setUpClass, test_function_calling_format, _test_function_calling_multiturn, touching `TestOpenAIServerFunctionCalling, setUpClass, test_function_calling_format`.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` added +1035/-0 (1035 lines); hunks: -0,0 +1,1035; symbols: Glm4MoeMLP, __init__, forward, Glm4MoeAttention
  - `test/srt/test_function_call_parser.py` modified +639/-0 (639 lines); hunks: -6,10 +6,12; -507,7 +509,9 @@ def setUp(self):; symbols: setUp, test_pythonic_detector_ebnf, test_qwen25_detector_ebnf, test_glm45_detector_ebnf
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +211/-203 (414 lines); hunks: -47,13 +47,15; -64,6 +66,13; symbols: forward, EPMoE, _get_tile_tokens_dim, __init__
  - `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +360/-4 (364 lines); hunks: -16,6 +16,20; -73,9 +87,13 @@ def test_function_calling_format(self):; symbols: TestOpenAIServerFunctionCalling, setUpClass, test_function_calling_format, _test_function_calling_multiturn
  - `python/sglang/srt/models/interns1.py` added +328/-0 (328 lines); hunks: -0,0 +1,328; symbols: InternS1ForConditionalGeneration, __init__, _update_hf_config, pixel_shuffle
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/glm4_moe.py
@@ -0,0 +1,1035 @@
+# Copyright 2025-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- test/srt/test_function_call_parser.py
@@ -6,10 +6,12 @@
+from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
+from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
@@ -507,7 +509,9 @@ def setUp(self):
+        self.qwen3_coder_detector = Qwen3CoderDetector()
+        self.glm45_detector = Glm4MoeDetector()
@@ -620,6 +624,49 @@ def test_qwen25_detector_ebnf(self):
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -47,13 +47,15 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/glm4_moe.py` added +1035/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +211/-203; `python/sglang/srt/models/interns1.py` added +328/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +25/-247; `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +25/-224; `python/sglang/srt/model_executor/forward_batch_info.py` modified +201/-29
  - tests: `test/srt/test_function_call_parser.py` modified +639/-0; `test/srt/openai_server/function_call/test_openai_function_calling.py` modified +360/-4
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `python/sglang/test/test_activation.py`, `python/sglang/test/test_utils.py`, `sgl-kernel/tests/test_kvcacheio.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8051 - Update batch size limitation of dsv3_router_gemm kernel to 16

- Link: https://github.com/sgl-project/sglang/pull/8051
- Status/date: merged / 2025-08-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `e7e5a3050a64`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update batch size limitation of dsv3_router_gemm kernel to 16"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Raise the batch size limitation to from 4 to 16 for dsv3_router_gemm kernel.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -248,8 +248,7 @@ def forward(self, hidden_states):; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-2 (3 lines); hunks: -248,8 +248,7 @@ def forward(self, hidden_states):; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -248,8 +248,7 @@ def forward(self, hidden_states):
-            and not self.is_nextn
-            and hidden_states.shape[0] < 4
+            and hidden_states.shape[0] <= 16
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9525 - tool-call(dsv3): Improve deepseek-v3 chat template and tool_choice = `required`

- Link: https://github.com/sgl-project/sglang/pull/9525
- Status/date: merged / 2025-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `python/sglang/srt/function_call/deepseekv3_detector.py`; associated commits `c9dd70fbde92`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +17/-17, 98 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "tool-call(dsv3): Improve deepseek-v3 chat template and tool_choice = `required`"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/deepseekv3_detector.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`; PR body summary: Improve deepseek-v3 chat template, following #8576 - Improve deepseek-v3 chat template - Remove spaces in the `build_ebnf` Actions Please wait for CI to pass, to make sure this....
- Key implementation: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1 (2 lines); hunks: -215,6 +215,6 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf, touching `build_ebnf`; `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +16/-16 (32 lines); hunks: -12,7 +12,7; -23,13 +23,13.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1 (2 lines); hunks: -215,6 +215,6 @@ def build_ebnf(self, tools: List[Tool]):; symbols: build_ebnf
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +16/-16 (32 lines); hunks: -12,7 +12,7; -23,13 +23,13
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv3_detector.py
@@ -215,6 +215,6 @@ def build_ebnf(self, tools: List[Tool]):
-            call_rule_fmt='"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n'''json\\n" {arguments_rule} "\\n'''<｜tool▁call▁end｜>"',
+            call_rule_fmt='"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n'''json\\n"{arguments_rule}"\\n'''<｜tool▁call▁end｜>"',
diff -- examples/chat_template/tool_chat_template_deepseekv3.jinja
@@ -12,7 +12,7 @@
-{%- endfor %}
+{%- endfor -%}
@@ -23,13 +23,13 @@
-        {% set tool_ns.text = tool_ns.text + '- `' + tool['name'] + '`:\n'''json\n' + (tool | tojson) + '\n'''\n' %}
+        {% set tool_ns.text = tool_ns.text + '\n'''json\n' + (tool | tojson) + '\n'''\n' %}
-{{ bos_token }}
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv3_detector.py` modified +1/-1
  - docs: `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +16/-16
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv3_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9446 - Support DeepSeek-V3.1 tool call

- Link: https://github.com/sgl-project/sglang/pull/9446
- Status/date: merged / 2025-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `b9683be6538e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +315/-0, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek-V3.1 tool call"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`; PR body summary: Support tool call for DeepSeek-V3.1 The tool call format of DeepSeek-V3.1 is different from DeepSeek-V3/R1: DeepSeek-V3.1: tool_call_name tool_call_arguments DeepSeek-R1/V3: fun....
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

### PR #9815 - fix: dsv3 lite q_lora_rank none

- Link: https://github.com/sgl-project/sglang/pull/9815
- Status/date: merged / 2025-08-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `8abe8deae6cd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-4, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "fix: dsv3 lite q_lora_rank none"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +12/-4 (16 lines); hunks: -2414,18 +2414,26 @@ def _weight_requant_ue8m0(self, is_nextn=False):; symbols: _weight_requant_ue8m0, touching `_weight_requant_ue8m0`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-4 (16 lines); hunks: -2414,18 +2414,26 @@ def _weight_requant_ue8m0(self, is_nextn=False):; symbols: _weight_requant_ue8m0
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2414,18 +2414,26 @@ def _weight_requant_ue8m0(self, is_nextn=False):
-            for module in [
-                layer.self_attn.fused_qkv_a_proj_with_mqa,
-                layer.self_attn.q_b_proj,
+            module_list = [
-            ]:
+            ]
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +12/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8118 - [feat] Support tp mode for DeepSeek-R1-W4AFP8

- Link: https://github.com/sgl-project/sglang/pull/8118
- Status/date: merged / 2025-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `d4a938417d2c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +291/-120, 710 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feat] Support tp mode for DeepSeek-R1-W4AFP8"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Support tp mode for DeepSeek w4a8 model, which has a better performace than ep mode. 1. Add W4AFp8MoEMethod and associated `create_weights`, `process_weights_after_loading` func....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights, touching `determine_num_fused_shared_experts, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(
+        elif self.quant_config.get_name() == "w4afp8":
+            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."
@@ -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal
+        # Params for special naming rules in mixed-precision models, for example:
+        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
+        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_cutlass_w4a8_moe.py`, `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9895 - Update tool_chat_template_deepseekv31.jinja

- Link: https://github.com/sgl-project/sglang/pull/9895
- Status/date: merged / 2025-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`; associated commits `cc9a31c66226`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update tool_chat_template_deepseekv31.jinja"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`; PR body summary: 增加tojson，解决多轮工具调用报错的问题： https://github.com/sgl-project/sglang/issues/9891.
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

### PR #9671 - Optimized deepseek-v3/r1 model performance on mxfp4 run

- Link: https://github.com/sgl-project/sglang/pull/9671
- Status/date: merged / 2025-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `0dfd54d11d06`, `1b2ff4fb7f05`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +458/-62, 763 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimized deepseek-v3/r1 model performance on mxfp4 run"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: In order to decrease the activation tensor quantized overhead, fused the quantized behavior into the different ops (activation, layernorm, gemm, flatten) Test below commands, we....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +202/-27 (229 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +202/-27 (229 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -112,6 +112,7 @@
+    is_gfx95_supported,
@@ -129,6 +130,22 @@
+_is_gfx95_supported = is_gfx95_supported()
+_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
+if _use_aiter_gfx95:
+    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +202/-27
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9959 - Revert "Optimized deepseek-v3/r1 model performance on mxfp4 run (#9671)"

- Link: https://github.com/sgl-project/sglang/pull/9959
- Status/date: merged / 2025-09-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `1b2ff4fb7f05`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +62/-458, 763 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "Optimized deepseek-v3/r1 model performance on mxfp4 run (#9671)""; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This reverts commit 0dfd54d11d06eb8363bc7fb3cf9a1f464368caf8. Because #9671 breaks on B200.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +27/-202 (229 lines); hunks: -112,7 +112,6; -130,22 +129,6; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +27/-202 (229 lines); hunks: -112,7 +112,6; -130,22 +129,6; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -112,7 +112,6 @@
-    is_gfx95_supported,
@@ -130,22 +129,6 @@
-_is_gfx95_supported = is_gfx95_supported()
-_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
-if _use_aiter_gfx95:
-    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +27/-202
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10008 - Optimized deepseek-v3/r1 model performance on mxfp4 run

- Link: https://github.com/sgl-project/sglang/pull/10008
- Status/date: merged / 2025-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `e96973742c32`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +489/-67, 850 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimized deepseek-v3/r1 model performance on mxfp4 run"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: In order to decrease the activation tensor quantized overhead, fused the quantized behavior into the different ops (activation, layernorm, gemm, flatten) Test below commands, we....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +228/-32 (260 lines); hunks: -112,6 +112,7; -129,6 +130,22; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -112,6 +112,7 @@
+    is_gfx95_supported,
@@ -129,6 +130,22 @@
+_is_gfx95_supported = is_gfx95_supported()
+_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
+if _use_aiter_gfx95:
+    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +228/-32
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py`, `python/sglang/srt/layers/quantization/quark/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8677 - Fix accuracy drop of dsv3 run in dp enablement

- Link: https://github.com/sgl-project/sglang/pull/8677
- Status/date: merged / 2025-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `918e3d4c27c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +100/-69, 210 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix accuracy drop of dsv3 run in dp enablement"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Got an accuracy issue of dsv3 model run on aiter backend with dp enablement. Issue link: https://github.com/sgl-project/sglang/issues/7692 For extend attention, use absorb atten....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -1077,7 +1077,13 @@ def _dispatch_mla_subtype():; symbols: _dispatch_mla_subtype, touching `_dispatch_mla_subtype`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -1077,7 +1077,13 @@ def _dispatch_mla_subtype():; symbols: _dispatch_mla_subtype
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1077,7 +1077,13 @@ def _dispatch_mla_subtype():
-                return AttnForwardMethod.MHA
+                if is_dp_attention_enabled():
+                    if sum(forward_batch.extend_prefix_lens_cpu) == 0:
+                        return AttnForwardMethod.MHA
+                    else:
+                        return AttnForwardMethod.MLA
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9834 - perf: Avoid unnecessary data type conversions for DeepSeek-V3 on Blackwell

- Link: https://github.com/sgl-project/sglang/pull/9834
- Status/date: merged / 2025-09-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `012584ecd539`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +13/-4, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "perf: Avoid unnecessary data type conversions for DeepSeek-V3 on Blackwell"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Optimize the performance of DeepSeek-V3 on Blackwell for small batch sizes by - Change the output data type of router GEMM to FP32 - Convert the data type of `correction_bias` i....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-3 (14 lines); hunks: -67,7 +67,10; -299,7 +302,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator:...; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-3 (14 lines); hunks: -67,7 +67,10; -299,7 +302,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator:...; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -67,7 +67,10 @@
-from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
+from sglang.srt.layers.moe.fused_moe_triton.layer import (
+    FusedMoE,
+    _is_fp4_quantization_enabled,
+)
@@ -299,7 +302,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = Non
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10104 - Fix run time error in dsv3-fp8 model on mi35x

- Link: https://github.com/sgl-project/sglang/pull/10104
- Status/date: merged / 2025-09-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `400d3b97aebc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix run time error in dsv3-fp8 model on mi35x"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Got run-time error in running dsv3-fp8 model on mi35x. The error is caused by the activation tensor is changed to tuple of tensors type for the next mxfp4 linear computation. Bu....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -249,7 +249,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -249,7 +249,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -249,7 +249,11 @@ def forward(
-        if gemm_output_zero_allocator != None and x.shape[0] <= 256:
+        if (
+            gemm_output_zero_allocator is not None
+            and x.shape[0] <= 256
+            and self.gate_up_proj.weight.dtype == torch.uint8
+        ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10209 - tool-call(dsv3): Fixed a parse problem when there are multiple function definitions in tool_calls

- Link: https://github.com/sgl-project/sglang/pull/10209
- Status/date: merged / 2025-09-09
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv3.jinja`; associated commits `2cd94dd07eea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "tool-call(dsv3): Fixed a parse problem when there are multiple function definitions in tool_calls"; model line: DeepSeek V3/R1; category: bug fix; main diff: `examples/chat_template/tool_chat_template_deepseekv3.jinja`; PR body summary: When there are multiple function definitions in tool_calls, an error will be reported **TypeError: can only concatenate str (not "dict") to str** Add what is missing `|tojson` r....
- Key implementation: `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +1/-1 (2 lines); hunks: -55,7 +55,7.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +1/-1 (2 lines); hunks: -55,7 +55,7
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv3.jinja
@@ -55,7 +55,7 @@
-                {{- '\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['arguments'] + '\n' + ''''
+                {{- '\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\n' + ''''json' + '\n' + tool['function']['arguments']|tojson + '\n'
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +1/-1
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_deepseekv3.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #10361 - Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile

- Link: https://github.com/sgl-project/sglang/pull/10361
- Status/date: merged / 2025-09-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +39/-5, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`; PR body summary: Got GPU fault when running "python3 -m sglang.launch_server --model-path DeepSeek-V3.1 --tp 8 --trust-remote-code --chunked-prefill-size 131072 --dp-size 8 --enable-dp-attention....
- Key implementation: `python/sglang/srt/layers/dp_attention.py` modified +24/-0 (24 lines); hunks: -119,6 +119,18 @@ def get_local_dp_buffer_len(cls) -> int:; -150,6 +162,18 @@ def get_dp_global_num_tokens() -> List[int]:; symbols: get_local_dp_buffer_len, get_dp_global_num_tokens, get_dp_hidden_size, get_dp_dtype, touching `get_local_dp_buffer_len, get_dp_global_num_tokens, get_dp_hidden_size`; `python/sglang/srt/layers/logits_processor.py` modified +15/-5 (20 lines); hunks: -35,6 +35,9; -187,16 +190,23 @@ def compute_dp_attention_metadata(self):; symbols: compute_dp_attention_metadata, _get_logits, touching `compute_dp_attention_metadata, _get_logits`.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +24/-0 (24 lines); hunks: -119,6 +119,18 @@ def get_local_dp_buffer_len(cls) -> int:; -150,6 +162,18 @@ def get_dp_global_num_tokens() -> List[int]:; symbols: get_local_dp_buffer_len, get_dp_global_num_tokens, get_dp_hidden_size, get_dp_dtype
  - `python/sglang/srt/layers/logits_processor.py` modified +15/-5 (20 lines); hunks: -35,6 +35,9; -187,16 +190,23 @@ def compute_dp_attention_metadata(self):; symbols: compute_dp_attention_metadata, _get_logits
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/dp_attention.py
@@ -119,6 +119,18 @@ def get_local_dp_buffer_len(cls) -> int:
+    @classmethod
+    def get_dp_hidden_size(cls) -> int:
+        return cls._hidden_size
+    @classmethod
+    def get_dp_dtype(cls) -> torch.dtype:
+        return cls._dtype
diff -- python/sglang/srt/layers/logits_processor.py
@@ -35,6 +35,9 @@
+    get_dp_device,
+    get_dp_dtype,
+    get_dp_hidden_size,
@@ -187,16 +190,23 @@ def compute_dp_attention_metadata(self):
+        hidden_size = get_dp_hidden_size()
+        dtype = get_dp_dtype()
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/dp_attention.py` modified +24/-0; `python/sglang/srt/layers/logits_processor.py` modified +15/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11063 - Add DeepSeek-V3.2 Tool Call Template

- Link: https://github.com/sgl-project/sglang/pull/11063
- Status/date: merged / 2025-10-05
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv32.jinja`; associated commits `85c1f7937781`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +100/-0, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add DeepSeek-V3.2 Tool Call Template"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `examples/chat_template/tool_chat_template_deepseekv32.jinja`; no usable PR-body summary.
- Key implementation: `examples/chat_template/tool_chat_template_deepseekv32.jinja` added +100/-0 (100 lines); hunks: -0,0 +1,100.
- Code diff details:
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` added +100/-0 (100 lines); hunks: -0,0 +1,100
- Key code excerpts:

```diff
diff -- examples/chat_template/tool_chat_template_deepseekv32.jinja
@@ -0,0 +1,100 @@
+{% if not add_generation_prompt is defined %}
+  {% set add_generation_prompt = false %}
+{% endif %}
+{% if not thinking is defined %}
+  {% set thinking = false %}
+{% endif %}
```

- Reviewed files:
  - docs: `examples/chat_template/tool_chat_template_deepseekv32.jinja` added +100/-0
- Risk and verification: This is mostly docs/examples in `examples/chat_template/tool_chat_template_deepseekv32.jinja`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #11061 - Support DeepSeek V3.2 Exp

- Link: https://github.com/sgl-project/sglang/pull/11061
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` and 9 files; associated commits `efbc687c2817`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +4542/-141, 5365 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support DeepSeek V3.2 Exp"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: We now support DeepSeek V3.2. Try this model in the tracking issue #11060 Authors: General: @fzyzcjy @DarkSharpness @hnyls2002 @hebiao064 @Fridge003; AMD: @HaiShaw @kkHuang-amd....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` added +887/-0 (887 lines); hunks: -0,0 +1,887; symbols: NSAFlashMLAMetadata, slice, copy_, NSAMetadata, touching `NSAFlashMLAMetadata, slice, copy_`; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` added +785/-0 (785 lines); hunks: -0,0 +1,785; symbols: fast_log2_ceil, fast_pow2, fast_round_scale, act_quant_kernel, touching `fast_log2_ceil, fast_pow2, fast_round_scale`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` added +761/-0 (761 lines); hunks: -0,0 +1,761; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded, touching `BaseIndexerMetadata, get_seqlens_int32, get_page_table_64`; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` added +354/-0 (354 lines); hunks: -0,0 +1,354; symbols: GetK, execute, slow, torch_fast, touching `GetK, execute, slow`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` added +887/-0 (887 lines); hunks: -0,0 +1,887; symbols: NSAFlashMLAMetadata, slice, copy_, NSAMetadata
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` added +785/-0 (785 lines); hunks: -0,0 +1,785; symbols: fast_log2_ceil, fast_pow2, fast_round_scale, act_quant_kernel
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` added +761/-0 (761 lines); hunks: -0,0 +1,761; symbols: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` added +354/-0 (354 lines); hunks: -0,0 +1,354; symbols: GetK, execute, slow, torch_fast
  - `python/sglang/srt/models/deepseek_v2.py` modified +329/-17 (346 lines); hunks: -15,6 +15,7; -25,10 +26,16; symbols: AttnForwardMethod, handle_attention_ascend, _get_sum_extend_prefix_lens, _is_extend_without_speculative
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -0,0 +1,887 @@
+from __future__ import annotations
+import sys
+from dataclasses import dataclass
+from typing import TYPE_CHECKING, Dict, List, Literal, Optional, TypeAlias
+import torch
+from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -0,0 +1,785 @@
+from typing import Optional, Tuple
+import tilelang
+import tilelang.language as T
+import torch
+from sglang.srt.utils import is_hip
+tilelang.set_log_level("WARNING")
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -0,0 +1,761 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` added +887/-0; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` added +785/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` added +761/-0; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` added +354/-0; `python/sglang/srt/models/deepseek_v2.py` modified +329/-17; `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` added +255/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/get_logits_ut.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11512 - Update DeepSeek-R1-FP4 default config on blackwell

- Link: https://github.com/sgl-project/sglang/pull/11512
- Status/date: merged / 2025-10-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-1, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update DeepSeek-R1-FP4 default config on blackwell"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/server_args.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/server_args.py` modified +26/-1 (27 lines); hunks: -799,7 +799,32 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +26/-1 (27 lines); hunks: -799,7 +799,32 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -799,7 +799,32 @@ def _handle_model_specific_adjustments(self):
-        if model_arch in ["GptOssForCausalLM"]:
+        if model_arch in ["DeepseekV3ForCausalLM"]:
+            if is_cuda() and is_sm100_supported():
+                if (
+                    self.attention_backend is None
+                    and self.prefill_attention_backend is None
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +26/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11565 - [DSv32] Use torch.compile for _get_logits_head_gate

- Link: https://github.com/sgl-project/sglang/pull/11565
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `384733639a91`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32] Use torch.compile for _get_logits_head_gate"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: fuse 2 unary mul + 1 binary mul in _get_logits_head_gate. 1.8% e2e total throughput With torch.compile Without.
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-0 (1 lines); hunks: -205,6 +205,7 @@ def _forward_fake(; symbols: _forward_fake, _get_logits_head_gate, touching `_forward_fake, _get_logits_head_gate`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-0 (1 lines); hunks: -205,6 +205,7 @@ def _forward_fake(; symbols: _forward_fake, _get_logits_head_gate
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -205,6 +205,7 @@ def _forward_fake(
+    @torch.compile(dynamic=True)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8247 - [1/N]Support DeepSeek-R1 w4a8 normal deepep

- Link: https://github.com/sgl-project/sglang/pull/8247
- Status/date: merged / 2025-10-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +334/-7, 475 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N]Support DeepSeek-R1 w4a8 normal deepep"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; PR body summary: Support deepep normal mode for DeepSeek w4a8 model with @yangsijia-serena @rainj-me add forward_cutlass_w4a8 for deepep normal mode Command `SGL_ENABLE_JIT_DEEPGEMM=1 python3 -m....
- Key implementation: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +196/-0 (196 lines); hunks: -1,5 +1,6; -11,6 +12,9; symbols: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal, touching `cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +47/-1 (48 lines); hunks: -1,7 +1,7; -21,8 +21,10; symbols: apply, apply_deepep_normal, touching `apply, apply_deepep_normal`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +21/-2 (23 lines); hunks: -29,6 +29,7; -95,6 +96,11 @@ def __init__(; symbols: __init__, moe_impl, forward_flashinfer_cutedsl, forward_cutlass_w4afp8, touching `__init__, moe_impl, forward_flashinfer_cutedsl`; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +10/-4 (14 lines); hunks: -14,7 +14,12; -340,7 +345,10 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core, touching `dispatch_a, _dispatch_core`.
- Code diff details:
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +196/-0 (196 lines); hunks: -1,5 +1,6; -11,6 +12,9; symbols: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +47/-1 (48 lines); hunks: -1,7 +1,7; -21,8 +21,10; symbols: apply, apply_deepep_normal
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +21/-2 (23 lines); hunks: -29,6 +29,7; -95,6 +96,11 @@ def __init__(; symbols: __init__, moe_impl, forward_flashinfer_cutedsl, forward_cutlass_w4afp8
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +10/-4 (14 lines); hunks: -14,7 +14,12; -340,7 +345,10 @@ def dispatch_a(; symbols: dispatch_a, _dispatch_core
  - `python/sglang/srt/layers/moe/utils.py` modified +4/-0 (4 lines); hunks: -51,6 +51,7 @@ class MoeRunnerBackend(Enum):; -76,6 +77,9 @@ def is_flashinfer_cutedsl(self):; symbols: MoeRunnerBackend, is_auto, is_flashinfer_cutedsl, is_flashinfer_mxfp4
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -1,5 +1,6 @@
+import logging
@@ -11,6 +12,9 @@
+    deepep_permute_triton_kernel,
+    deepep_post_reorder_triton_kernel,
+    deepep_run_moe_deep_preprocess,
@@ -201,3 +205,195 @@ def cutlass_w4a8_moe(
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -1,7 +1,7 @@
-from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
+from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
@@ -21,8 +21,10 @@
+    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
+        DeepEPNormalOutput,
@@ -326,3 +328,47 @@ def apply(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -29,6 +29,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +196/-0; `python/sglang/srt/layers/quantization/w4afp8.py` modified +47/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +21/-2; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +10/-4; `python/sglang/srt/layers/moe/utils.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +1/-0
  - tests: `test/srt/quant/test_w4a8_deepseek_v3.py` modified +55/-0
- Risk and verification: The diff ships test coverage in `test/srt/quant/test_w4a8_deepseek_v3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11510 - [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support

- Link: https://github.com/sgl-project/sglang/pull/11510
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `3cceaa381ad3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +102/-33, 359 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3/DSV3/DSV3.2 model support"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This pr fixes very models that failed to start on npu after dsv3.2 support pr was merged. ci and image release infra was also improved. - Fix mla prefill padding mismatch with a....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def forward_prepare(; symbols: forward_prepare, touching `forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -1357,6 +1357,7 @@ def forward_prepare(; symbols: forward_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1357,6 +1357,7 @@ def forward_prepare(
+                inner_state = (*inner_state, None)  # add a position for topk_indices
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/ascend/test_ascend_deepep.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11109 - [Draft] Support MTP for DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/11109
- Status/date: closed / 2025-10-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +180/-25, 309 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Draft] Support MTP for DeepSeek-V3.2"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py`; no usable PR-body summary.
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +146/-21 (167 lines); hunks: -149,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; -186,6 +193,14 @@ def __init__(self, model_runner: ModelRunner):; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, touching `compute_cu_seqlens, NativeSparseAttnBackend, __init__`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +11/-3 (14 lines); hunks: -373,6 +373,9 @@ def _get_topk_ragged(; -385,7 +388,9 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, _forward, touching `_get_topk_ragged, _forward`; `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: is_deepseek_nsa, touching `is_deepseek_nsa`; `python/sglang/srt/speculative/eagle_worker.py` modified +18/-0 (18 lines); hunks: -225,6 +225,7 @@ def _create_decode_backend(self):; -247,6 +248,7 @@ def _create_draft_extend_backend(self):; symbols: _create_decode_backend, _create_draft_extend_backend, _create_flashmla_decode_backend, _create_nsa_decode_backend, touching `_create_decode_backend, _create_draft_extend_backend, _create_flashmla_decode_backend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +146/-21 (167 lines); hunks: -149,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; -186,6 +193,14 @@ def __init__(self, model_runner: ModelRunner):; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +11/-3 (14 lines); hunks: -373,6 +373,9 @@ def _get_topk_ragged(; -385,7 +388,9 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, _forward
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: is_deepseek_nsa
  - `python/sglang/srt/speculative/eagle_worker.py` modified +18/-0 (18 lines); hunks: -225,6 +225,7 @@ def _create_decode_backend(self):; -247,6 +248,7 @@ def _create_draft_extend_backend(self):; symbols: _create_decode_backend, _create_draft_extend_backend, _create_flashmla_decode_backend, _create_nsa_decode_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -149,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
+        model_runner: ModelRunner,
+        skip_prefill: bool = False,
+        speculative_step_id=0,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -373,6 +373,9 @@ def _get_topk_ragged(
+            print(
+                f"forward_batch.mode: {forward_batch.forward_mode.name}, extend_seq_len: {extend_seq_len}"
+            )
@@ -385,7 +388,9 @@ def _get_topk_ragged(
+        print(
+            f"forward_batch.mode: {forward_batch.forward_mode.name}, q_fp8.shape: {q_fp8.shape}, ks.shape: {ks.shape}, ke.shape: {ke.shape}"
diff -- python/sglang/srt/configs/model_config.py
@@ -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +146/-21; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +11/-3; `python/sglang/srt/configs/model_config.py` modified +5/-1; `python/sglang/srt/speculative/eagle_worker.py` modified +18/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11682 - Cleaning indexer for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11682
- Status/date: merged / 2025-10-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; associated commits `20b8d2306c3d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-66, 122 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Cleaning indexer for DeepSeek V3.2"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; PR body summary: - Fake indexer is for testing accuracy during early development, so can be deprecated now. - Cleaning forward logic..
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-65 (68 lines); hunks: -17,7 +17,7; -168,43 +168,6 @@ def __init__(; symbols: __init__, _forward_fake, _get_logits_head_gate, _get_topk_ragged, touching `__init__, _forward_fake, _get_logits_head_gate`; `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-1 (1 lines); hunks: -1,7 +1,6.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-65 (68 lines); hunks: -17,7 +17,7; -168,43 +168,6 @@ def __init__(; symbols: __init__, _forward_fake, _get_logits_head_gate, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-1 (1 lines); hunks: -1,7 +1,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -17,7 +17,7 @@
-from sglang.srt.layers.attention.nsa.utils import NSA_DUAL_STREAM, NSA_USE_REAL_INDEXER
+from sglang.srt.layers.attention.nsa.utils import NSA_DUAL_STREAM
@@ -168,43 +168,6 @@ def __init__(
-    def _forward_fake(
-        self,
-        x: torch.Tensor,
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -1,7 +1,6 @@
-NSA_USE_REAL_INDEXER = get_bool_env_var("SGLANG_NSA_USE_REAL_INDEXER", "true")
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-65; `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11652 - [Spec Decoding] Support MTP for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/11652
- Status/date: merged / 2025-10-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `efa473348bc9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +445/-79, 814 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Spec Decoding] Support MTP for dsv3.2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Based on https://github.com/sgl-project/sglang/pull/11109 We have implemented MTP support for DS v3.2 and ***cuda graph*** in our in-house maintained version of sglang. Since th....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +385/-68 (453 lines); hunks: -29,6 +29,7; -148,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, touching `compute_cu_seqlens, NativeSparseAttnBackend, __init__`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +23/-10 (33 lines); hunks: -266,7 +266,10 @@ def _get_topk_paged(; -317,8 +320,9 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _get_topk_ragged, forward_indexer, forward_cuda, touching `_get_topk_paged, _get_topk_ragged, forward_indexer`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +385/-68 (453 lines); hunks: -29,6 +29,7; -148,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +23/-10 (33 lines); hunks: -266,7 +266,10 @@ def _get_topk_paged(; -317,8 +320,9 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _get_topk_ragged, forward_indexer, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -29,6 +29,7 @@
@@ -148,7 +149,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
+        model_runner: ModelRunner,
+        skip_prefill: bool = False,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -266,7 +266,10 @@ def _get_topk_paged(
-        seqlens_32 = metadata.get_seqlens_int32()
+        if forward_batch.forward_mode.is_target_verify():
+            seqlens_32 = metadata.get_seqlens_expanded()
+        else:
+            seqlens_32 = metadata.get_seqlens_int32()
@@ -317,8 +320,9 @@ def _get_topk_ragged(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +385/-68; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +23/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12000 - [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU

- Link: https://github.com/sgl-project/sglang/pull/12000
- Status/date: merged / 2025-10-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +64/-5, 132 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`; PR body summary: Part of this Issue: https://github.com/sgl-project/sglang/issues/10278 As part of deepseek deterministic inference support, this change ensures deterministic inference results f....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -350,7 +350,11 @@ def handle_attention_flashinfer(attn, forward_batch):; -394,6 +398,10 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_flashinfer, handle_attention_fa3, handle_attention_flashmla, handle_attention_nsa, touching `handle_attention_flashinfer, handle_attention_fa3, handle_attention_flashmla`; `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +31/-2 (33 lines); hunks: -495,16 +495,37 @@ def mean_batch_invariant(input, dim, keepdim=False, dtype:...; -516,12 +537,20 @@ def enable_batch_invariant_mode():; symbols: mean_batch_invariant, bmm_batch_invariant, is_batch_invariant_mode_enabled, enable_batch_invariant_mode, touching `mean_batch_invariant, bmm_batch_invariant, is_batch_invariant_mode_enabled`; `python/sglang/srt/server_args.py` modified +24/-2 (26 lines); hunks: -1532,13 +1532,30 @@ def _handle_deterministic_inference(self):; -1553,8 +1570,13 @@ def _handle_deterministic_inference(self):; symbols: _handle_deterministic_inference, touching `_handle_deterministic_inference`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -350,7 +350,11 @@ def handle_attention_flashinfer(attn, forward_batch):; -394,6 +398,10 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_flashinfer, handle_attention_fa3, handle_attention_flashmla, handle_attention_nsa
  - `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +31/-2 (33 lines); hunks: -495,16 +495,37 @@ def mean_batch_invariant(input, dim, keepdim=False, dtype:...; -516,12 +537,20 @@ def enable_batch_invariant_mode():; symbols: mean_batch_invariant, bmm_batch_invariant, is_batch_invariant_mode_enabled, enable_batch_invariant_mode
  - `python/sglang/srt/server_args.py` modified +24/-2 (26 lines); hunks: -1532,13 +1532,30 @@ def _handle_deterministic_inference(self):; -1553,8 +1570,13 @@ def _handle_deterministic_inference(self):; symbols: _handle_deterministic_inference
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -350,7 +350,11 @@ def handle_attention_flashinfer(attn, forward_batch):
-    return _handle_attention_backend(attn, forward_batch, "fa3")
+    # when deterministic inference is enabled, use MLA
+    if get_global_server_args().enable_deterministic_inference:
+        return _dispatch_mla_subtype(attn, forward_batch)
+    else:
+        return _handle_attention_backend(attn, forward_batch, "fa3")
diff -- python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py
@@ -495,16 +495,37 @@ def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None =
+def bmm_batch_invariant(a, b, *, out=None):
+    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
+    # Process each batch separately with our persistent kernel
+    if a.ndim == 3 and b.ndim == 3:
+        results = []
+        for i in range(a.shape[0]):
diff -- python/sglang/srt/server_args.py
@@ -1532,13 +1532,30 @@ def _handle_deterministic_inference(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1; `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +31/-2; `python/sglang/srt/server_args.py` modified +24/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8464 - [2/N]Support DeepSeek-R1 w4a8 low latency deepep

- Link: https://github.com/sgl-project/sglang/pull/8464
- Status/date: merged / 2025-10-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +531/-9, 648 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[2/N]Support DeepSeek-R1 w4a8 low latency deepep"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; PR body summary: Follow https://github.com/sgl-project/sglang/pull/8247 https://github.com/sgl-project/sglang/pull/7762. Based on https://github.com/sgl-project/sglang/pull/8311. Support deepep....
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +194/-0 (194 lines); hunks: -1014,3 +1014,197 @@ def zero_experts_compute_triton(; symbols: zero_experts_compute_triton, compute_problem_sizes_w4a8_kernel, compute_problem_sizes_w4a8, deepep_ll_get_cutlass_w4a8_moe_mm_data, touching `zero_experts_compute_triton, compute_problem_sizes_w4a8_kernel, compute_problem_sizes_w4a8`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +138/-0 (138 lines); hunks: -11,12 +11,14; -396,3 +398,139 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, touching `cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +36/-0 (36 lines); hunks: -23,6 +23,7; -328,6 +329,41 @@ def apply(; symbols: apply, apply_deepep_ll, apply_deepep_normal, touching `apply, apply_deepep_ll, apply_deepep_normal`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +17/-0 (17 lines); hunks: -100,6 +100,7 @@ def __init__(; -199,6 +200,8 @@ def run_moe_core(; symbols: __init__, run_moe_core, forward_deepgemm_masked, forward_cutlass_w4afp8_masked, touching `__init__, run_moe_core, forward_deepgemm_masked`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +194/-0 (194 lines); hunks: -1014,3 +1014,197 @@ def zero_experts_compute_triton(; symbols: zero_experts_compute_triton, compute_problem_sizes_w4a8_kernel, compute_problem_sizes_w4a8, deepep_ll_get_cutlass_w4a8_moe_mm_data
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +138/-0 (138 lines); hunks: -11,12 +11,14; -396,3 +398,139 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +36/-0 (36 lines); hunks: -23,6 +23,7; -328,6 +329,41 @@ def apply(; symbols: apply, apply_deepep_ll, apply_deepep_normal
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +17/-0 (17 lines); hunks: -100,6 +100,7 @@ def __init__(; -199,6 +200,8 @@ def run_moe_core(; symbols: __init__, run_moe_core, forward_deepgemm_masked, forward_cutlass_w4afp8_masked
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` modified +72/-6 (78 lines); hunks: -34,6 +34,40 @@ __global__ void int4_fp8_get_group_gemm_starts(; -55,6 +89,28 @@ __global__ void int4_fp8_get_group_gemm_starts(
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -1014,3 +1014,197 @@ def zero_experts_compute_triton(
+@triton.jit
+def compute_problem_sizes_w4a8_kernel(
+    masked_m_ptr,
+    problem_sizes1_ptr,
+    problem_sizes2_ptr,
+    n,
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -11,12 +11,14 @@
+    deepep_ll_get_cutlass_w4a8_moe_mm_data,
+    silu_and_mul_masked_post_per_tensor_quant_fwd,
@@ -396,3 +398,139 @@ def cutlass_w4a8_moe_deepep_normal(
+def cutlass_w4a8_moe_deepep_ll(
+    a: torch.Tensor,
+    w1_q: torch.Tensor,
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -23,6 +23,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +194/-0; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +138/-0; `python/sglang/srt/layers/quantization/w4afp8.py` modified +36/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +17/-0
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` modified +72/-6; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` modified +4/-2
  - tests: `test/srt/quant/test_w4a8_deepseek_v3.py` modified +69/-0; `test/srt/run_suite.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/quant/test_w4a8_deepseek_v3.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11877 - [Doc] Add documentation for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11877
- Status/date: merged / 2025-10-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`; associated commits `729b242934cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +723/-3, 749 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Doc] Add documentation for DeepSeek V3.2"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`; PR body summary: Depends on #11844 and #11876 Also waiting for @whybeyoung on some PD updates.
- Key implementation: `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunks: -0,0 +1,570.
- Code diff details:
  - `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunks: -0,0 +1,570
- Key code excerpts:

```diff
diff -- docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md
@@ -0,0 +1,570 @@
+# DeepSeekV32-Exp RBG Based PD Deploy
+## 0. Prerequisites
+1. k8s >=1.26
+2. lws installed on k8s.
+3. rbg installed on k8s.
+For RBG installation, please refer to: https://github.com/sgl-project/rbg
```

- Reviewed files:
  - docs: `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0
- Risk and verification: This is mostly docs/examples in `docs/advanced_features/separate_reasoning.ipynb`, `docs/basic_usage/deepseek.md`, `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12057 - [doc] add example of using w4fp8 for Deepseek

- Link: https://github.com/sgl-project/sglang/pull/12057
- Status/date: merged / 2025-10-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-0, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[doc] add example of using w4fp8 for Deepseek"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `benchmark/deepseek_v3/README.md`; PR body summary: This PR adds a basic example of deploying w4fp8 Deepseek using SGLang. Below are the evaluation of accuracy. | Model | MMLU | |-------|------| | novita/Deepseek-V3.1-W4AFP8 | 0.....
- Key implementation: `benchmark/deepseek_v3/README.md` modified +15/-0 (15 lines); hunks: -368,6 +368,21 @@ edit your `config.json` and remove the `quantization_config....
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +15/-0 (15 lines); hunks: -368,6 +368,21 @@ edit your `config.json` and remove the `quantization_config...
- Key code excerpts:

```diff
diff -- benchmark/deepseek_v3/README.md
@@ -368,6 +368,21 @@ edit your `config.json` and remove the `quantization_config` block. For example:
+# Example: Serving with 4 H200 with w4fp8 Quantization
+There are mixed-precision quantization methods where MoE layers are computed using W4(int)A(FP)8 quantization while the dense layers remain in FP8 precision. Users can run these m
+'''bash
+python -m sglang.launch_server --model novita/Deepseek-V3-0324-W4AFP8 --mem-fraction-static 0.85 --disable-shared-experts-fusion --tp-size 4
+'''
+Other variants of pre-quantized DeepSeek models are also available:
```

- Reviewed files:
  - other: `benchmark/deepseek_v3/README.md` modified +15/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #11708 - Support running FP4 Deepseek on SM120.

- Link: https://github.com/sgl-project/sglang/pull/11708
- Status/date: merged / 2025-10-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +33/-35, 233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support running FP4 Deepseek on SM120."; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: sm120 is not runnable for deepseek because of sm check and invalid kernels. - add `is_blackwell_supported` as an extend to `is_sm100_supported` - bypass dsv3_fused_a_gemm on sm1....
- Key implementation: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +8/-7 (15 lines); hunks: -28,7 +28,7; -49,8 +49,10; symbols: apply, ModelOptNvFp4FusedMoEMethod, __init__, touching `apply, ModelOptNvFp4FusedMoEMethod, __init__`; `python/sglang/srt/models/gpt_oss.py` modified +1/-10 (11 lines); hunks: -70,18 +70,9; `python/sglang/srt/models/deepseek_v2.py` modified +1/-5 (6 lines); hunks: -131,13 +131,11; -197,8 +195,6; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-2 (4 lines); hunks: -26,8 +26,8; -229,7 +229,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +8/-7 (15 lines); hunks: -28,7 +28,7; -49,8 +49,10; symbols: apply, ModelOptNvFp4FusedMoEMethod, __init__
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-10 (11 lines); hunks: -70,18 +70,9
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-5 (6 lines); hunks: -131,13 +131,11; -197,8 +195,6; symbols: __init__
  - `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-2 (4 lines); hunks: -26,8 +26,8; -229,7 +229,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +2/-2 (4 lines); hunks: -25,8 +25,8; -243,7 +243,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -28,7 +28,7 @@
-    is_sm100_supported,
+    is_blackwell_supported,
@@ -49,8 +49,10 @@
-if is_cuda():
-    from sgl_kernel import scaled_fp4_quant
+try:
diff -- python/sglang/srt/models/gpt_oss.py
@@ -70,18 +70,9 @@
-from sglang.srt.utils import (
-    LazyValue,
-    add_prefix,
-    is_cuda,
-    is_flashinfer_available,
-    is_sm100_supported,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -131,13 +131,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +8/-7; `python/sglang/srt/models/gpt_oss.py` modified +1/-10; `python/sglang/srt/models/deepseek_v2.py` modified +1/-5; `python/sglang/srt/layers/attention/flashinfer_backend.py` modified +2/-2; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +2/-2; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_fp8_blockwise_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11655 - [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache

- Link: https://github.com/sgl-project/sglang/pull/11655
- Status/date: merged / 2025-10-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `81a632ace647`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +367/-44, 626 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`; PR body summary: Add logics to dequant the kvcache from fp8 to bf16 in a separate kernel and use `flashmla_prefill` kernel with fp8 kvcache. flashmla_decode (before) flashmla_prefill with no kvc....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +156/-20 (176 lines); hunks: -1,12 +1,14; -98,11 +100,27 @@ class NSAMetadata:; symbols: NSAMetadata, TopkTransformMethod, NSAIndexerMetadata, get_seqlens_int32, touching `NSAMetadata, TopkTransformMethod, NSAIndexerMetadata`; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +138/-6 (144 lines); hunks: -22,6 +22,10 @@ def _dequantize_k_cache_slow(; -45,16 +49,21 @@ def _dequantize_k_cache_slow(; symbols: _dequantize_k_cache_slow, _dequantize_k_cache_fast_wrapped, _dequantize_k_cache_fast, touching `_dequantize_k_cache_slow, _dequantize_k_cache_fast_wrapped, _dequantize_k_cache_fast`; `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +44/-12 (56 lines); hunks: -206,6 +206,8 @@ def _quantize_k_cache_fast_kernel(; -217,21 +219,9 @@ def _quantize_k_cache_fast_kernel(; symbols: _quantize_k_cache_fast_kernel, run_ans, touching `_quantize_k_cache_fast_kernel, run_ans`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +156/-20 (176 lines); hunks: -1,12 +1,14; -98,11 +100,27 @@ class NSAMetadata:; symbols: NSAMetadata, TopkTransformMethod, NSAIndexerMetadata, get_seqlens_int32
  - `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +138/-6 (144 lines); hunks: -22,6 +22,10 @@ def _dequantize_k_cache_slow(; -45,16 +49,21 @@ def _dequantize_k_cache_slow(; symbols: _dequantize_k_cache_slow, _dequantize_k_cache_fast_wrapped, _dequantize_k_cache_fast
  - `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +44/-12 (56 lines); hunks: -206,6 +206,8 @@ def _quantize_k_cache_fast_kernel(; -217,21 +219,9 @@ def _quantize_k_cache_fast_kernel(; symbols: _quantize_k_cache_fast_kernel, run_ans
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1,12 +1,14 @@
+from enum import IntEnum, auto
+from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
@@ -98,11 +100,27 @@ class NSAMetadata:
+    # The sum of sequence lengths for key, prefill only
+    seq_lens_sum: Optional[int] = None
+    # The flattened 1D page table with shape (seq_lens_sum,), prefill only
diff -- python/sglang/srt/layers/attention/nsa/dequant_k_cache.py
@@ -22,6 +22,10 @@ def _dequantize_k_cache_slow(
+    original_ndim = quant_k_cache.ndim
+    if original_ndim == 3:
+        # set block_size = 1
+        quant_k_cache = quant_k_cache.unsqueeze(1)
@@ -45,16 +49,21 @@ def _dequantize_k_cache_slow(
-    result = result.view(num_blocks, block_size, 1, d)
diff -- python/sglang/srt/layers/attention/nsa/quant_k_cache.py
@@ -206,6 +206,8 @@ def _quantize_k_cache_fast_kernel(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +156/-20; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +138/-6; `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +44/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12294 - [Deepseek V3.2] Enable flashmla_auto with MTP

- Link: https://github.com/sgl-project/sglang/pull/12294
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `42e1a72efb78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-3, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Enable flashmla_auto with MTP"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Follow up to https://github.com/sgl-project/sglang/pull/11655 With fp8 kvcache on B200, flashmla_sparse can be used in the extend (not draft_extend) phases in both normal and sp....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunks: -1222,11 +1222,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[Fo...; symbols: set_nsa_prefill_impl, touching `set_nsa_prefill_impl`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunks: -1222,11 +1222,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[Fo...; symbols: set_nsa_prefill_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1222,11 +1222,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None) ->
-                    # TODO(hlu1): enable MTP
-                    and forward_batch.forward_mode.is_extend()
-                    and forward_batch.spec_algorithm.is_none()
+                    and forward_batch.forward_mode == ForwardMode.EXTEND
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12095 - [2/2] Deepseek deterministic: support deepseek v3 deterministic inference on 8 x H200

- Link: https://github.com/sgl-project/sglang/pull/12095
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `e39628fd07b3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +19/-0, 54 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[2/2] Deepseek deterministic: support deepseek v3 deterministic inference on 8 x H200"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: part of roadmap Previous PR supported deepseek arch model's deterministic inference on a single Hopper GPU. This PR is to further support full deepseek v3 model's deterministic....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunks: -515,6 +515,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator:...; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunks: -515,6 +515,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator:...; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -515,6 +515,9 @@ def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = Non
+        if get_global_server_args().enable_deterministic_inference:
+            return F.linear(hidden_states, self.weight, None)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/srt/test_fused_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12094 - Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4

- Link: https://github.com/sgl-project/sglang/pull/12094
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `9ff9fa7f95be`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +110/-22, 231 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Since wk and weight_proj in Indexer have the same input, they can be fused into a single gemm by concatenating weights. For the base DeepSeekV3.2-Exp model, wk is FP8 and weight....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-22 (67 lines); hunks: -119,6 +119,7 @@ def __init__(; -129,6 +130,7 @@ def __init__(; symbols: __init__, _get_logits_head_gate, touching `__init__, _get_logits_head_gate`; `python/sglang/srt/models/deepseek_v2.py` modified +65/-0 (65 lines); hunks: -224,6 +224,17 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1143,6 +1154,9 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__, touching `add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-22 (67 lines); hunks: -119,6 +119,7 @@ def __init__(; -129,6 +130,7 @@ def __init__(; symbols: __init__, _get_logits_head_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +65/-0 (65 lines); hunks: -224,6 +224,17 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1143,6 +1154,9 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -119,6 +119,7 @@ def __init__(
+        fuse_wk_and_weights_proj: bool = False,
@@ -129,6 +130,7 @@ def __init__(
+        self.fuse_wk_and_weights_proj = fuse_wk_and_weights_proj
@@ -140,21 +142,29 @@ def __init__(
-        self.wk = ReplicatedLinear(
-            self.hidden_size,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -224,6 +224,17 @@ def add_forward_absorb_core_attention_backend(backend_name):
+def is_nsa_indexer_wk_and_weights_proj_fused(config, quant_config):
+    """
+    NSA Indexer wk and weights_proj can be fused in FP4 model because they are both in BF16
+    """
+    return (
+        is_deepseek_nsa(config)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-22; `python/sglang/srt/models/deepseek_v2.py` modified +65/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12300 - [DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type

- Link: https://github.com/sgl-project/sglang/pull/12300
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/transform_index.py`; associated commits `662725b936a2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/transform_index.py`; PR body summary: Running transform_index.py in NSA attention path fails at torch.gather(...) with a dtype mismatch: the index/out tensor is int (int32) while PyTorch requires Long (int64) for ga....
- Key implementation: `python/sglang/srt/layers/attention/nsa/transform_index.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def transform_index_page_table_decode_ref(; symbols: transform_index_page_table_decode_ref, touching `transform_index_page_table_decode_ref`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/transform_index.py` modified +1/-1 (2 lines); hunks: -103,7 +103,7 @@ def transform_index_page_table_decode_ref(; symbols: transform_index_page_table_decode_ref
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/transform_index.py
@@ -103,7 +103,7 @@ def transform_index_page_table_decode_ref(
-        page_table,
+        page_table.to(result.dtype),
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/transform_index.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/transform_index.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11892 - DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill

- Link: https://github.com/sgl-project/sglang/pull/11892
- Status/date: merged / 2025-11-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `f235498eca7a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +188/-4, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: For DeepSeek-V3.2 models, using MLA (Multi-Latent Attention) uniformly across all sequence lengths during prefill is suboptimal. For short sequences, the overhead of MLA's compr....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +84/-0 (84 lines); hunks: -261,6 +261,30 @@ def _get_q_k_bf16(; -394,6 +418,45 @@ def _get_topk_ragged(; symbols: _get_q_k_bf16, _get_k_bf16, _get_topk_paged, _get_topk_ragged, touching `_get_q_k_bf16, _get_k_bf16, _get_topk_paged`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +61/-2 (63 lines); hunks: -47,7 +47,7; -823,7 +823,23 @@ def forward_extend(; symbols: forward_extend, _forward_flashmla_kv, _forward_standard_mha, _forward_tilelang, touching `forward_extend, _forward_flashmla_kv, _forward_standard_mha`; `python/sglang/srt/models/deepseek_v2.py` modified +43/-2 (45 lines); hunks: -402,6 +402,34 @@ def handle_attention_aiter(attn, forward_batch):; -1478,8 +1506,21 @@ def forward_normal_prepare(; symbols: handle_attention_aiter, handle_attention_nsa, forward_normal_prepare, touching `handle_attention_aiter, handle_attention_nsa, forward_normal_prepare`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +84/-0 (84 lines); hunks: -261,6 +261,30 @@ def _get_q_k_bf16(; -394,6 +418,45 @@ def _get_topk_ragged(; symbols: _get_q_k_bf16, _get_k_bf16, _get_topk_paged, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +61/-2 (63 lines); hunks: -47,7 +47,7; -823,7 +823,23 @@ def forward_extend(; symbols: forward_extend, _forward_flashmla_kv, _forward_standard_mha, _forward_tilelang
  - `python/sglang/srt/models/deepseek_v2.py` modified +43/-2 (45 lines); hunks: -402,6 +402,34 @@ def handle_attention_aiter(attn, forward_batch):; -1478,8 +1506,21 @@ def forward_normal_prepare(; symbols: handle_attention_aiter, handle_attention_nsa, forward_normal_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -261,6 +261,30 @@ def _get_q_k_bf16(
+    def _get_k_bf16(
+        self,
+        x: torch.Tensor,
+        positions: torch.Tensor,
+        enable_dual_stream: bool,
+    ):
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -47,7 +47,7 @@
-    from sgl_kernel.flash_attn import flash_attn_with_kvcache
+    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
@@ -823,7 +823,23 @@ def forward_extend(
-        # Do absorbed multi-latent attention
+        # Detect MHA mode: multi KV heads (vs MLA with single KV head)
+        is_mha_mode = (layer.tp_k_head_num == layer.tp_q_head_num) and (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -402,6 +402,34 @@ def handle_attention_aiter(attn, forward_batch):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +84/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +61/-2; `python/sglang/srt/models/deepseek_v2.py` modified +43/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12778 - Update dsv3 quantization auto setting for sm100

- Link: https://github.com/sgl-project/sglang/pull/12778
- Status/date: merged / 2025-11-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-9, 42 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update dsv3 quantization auto setting for sm100"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/server_args.py`; PR body summary: update quantization based on model quant_method.
- Key implementation: `python/sglang/srt/server_args.py` modified +22/-9 (31 lines); hunks: -912,19 +912,32 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +22/-9 (31 lines); hunks: -912,19 +912,32 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -912,19 +912,32 @@ def _handle_model_specific_adjustments(self):
-                if self.moe_a2a_backend == "none" and self.moe_runner_backend == "auto":
-                    self.moe_runner_backend = "flashinfer_trtllm"
-                    logger.info(
-                        "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
-                    )
-                    if self.quantization is None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +22/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12788 - [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)

- Link: https://github.com/sgl-project/sglang/pull/12788
- Status/date: merged / 2025-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `7257525ccea6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-6, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Follow up this PR: DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill. Enable and optimize MHA on B200 (SM100) in the NSA backend by TRT-LLM ragged att....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +46/-2 (48 lines); hunks: -7,6 +7,7; -50,6 +51,10; symbols: NSAFlashMLAMetadata, __init__, get_device_int32_arange, _forward_standard_mha, touching `NSAFlashMLAMetadata, __init__, get_device_int32_arange`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunks: -415,11 +415,14 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, touching `handle_attention_nsa`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +46/-2 (48 lines); hunks: -7,6 +7,7; -50,6 +51,10; symbols: NSAFlashMLAMetadata, __init__, get_device_int32_arange, _forward_standard_mha
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunks: -415,11 +415,14 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -7,6 +7,7 @@
+from sglang.srt.environ import envs
@@ -50,6 +51,10 @@
+# Reuse this workspace buffer across all NSA backend instances
+global_workspace_buffer = None
@@ -231,6 +236,20 @@ def __init__(
+        # Allocate global workspace buffer for TRTLLm ragged attention kernel (SM100/B200)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -415,11 +415,14 @@ def handle_attention_nsa(attn, forward_batch):
-        # B200 (SM100) is temporarily disabled for MHA due to FA4 accuracy issues
-        # Currently only H200 (SM90) with FA3 is allowed to use MHA path
-        is_hopper = _device_sm == 90
+        # MHA path enabled for both H200 (SM90, FA3) and B200 (SM100, TRTLLm ragged)
+        # B200 uses trtllm_ragged_attention_deepseek kernel instead of FA4
+        supports_mha = _device_sm in [90, 100]
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +46/-2; `python/sglang/srt/models/deepseek_v2.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12520 - [Test] Add DeepSeekV3.2 NSA Indexer Test Suite

- Link: https://github.com/sgl-project/sglang/pull/12520
- Status/date: merged / 2025-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; associated commits `125f76ea44d8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +617/-1, 633 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] Add DeepSeekV3.2 NSA Indexer Test Suite"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; PR body summary: Address the issue https://github.com/sgl-project/sglang/issues/12509 Added 11 tests: 1. test_forward_decode_mode 2. test_forward_extend_mode 3. test_indexer_basic_creation 4. te....
- Key implementation: `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +12/-1 (13 lines); hunks: -277,7 +277,18 @@ def _set_k_and_s_triton(; symbols: _set_k_and_s_triton, touching `_set_k_and_s_triton`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +12/-1 (13 lines); hunks: -277,7 +277,18 @@ def _set_k_and_s_triton(; symbols: _set_k_and_s_triton
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -277,7 +277,18 @@ def _set_k_and_s_triton(
-    num_tokens_to_write__, scale_dim = index_k_scale.shape
+    # Handle both 1D (num_tokens,) and 2D (num_tokens, 1) shapes for index_k_scale
+    if index_k_scale.ndim == 1:
+        num_tokens_to_write__ = index_k_scale.shape[0]
+        scale_dim = 1
+    elif index_k_scale.ndim == 2:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +12/-1
- Risk and verification: The diff ships test coverage in `test/srt/layers/attention/nsa/test_nsa_indexer.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12816 - [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative

- Link: https://github.com/sgl-project/sglang/pull/12816
- Status/date: merged / 2025-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `bef37d6de86a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-18, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Fixed a bug in https://github.com/sgl-project/sglang/pull/12788 We should only skip Indexer logits computation when forward_mode.is_extend_without_speculative() returns true, me....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +6/-14 (20 lines); hunks: -308,14 +308,6 @@ def _get_sum_extend_prefix_lens(forward_batch):; -334,7 +326,7 @@ def _handle_attention_backend(; symbols: _get_sum_extend_prefix_lens, _is_extend_without_speculative, _support_mha_one_shot, _handle_attention_backend, touching `_get_sum_extend_prefix_lens, _is_extend_without_speculative, _support_mha_one_shot`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: -410,6 +410,8 @@ def _forward_cuda_k_only(; -555,14 +557,15 @@ def forward_cuda(; symbols: _forward_cuda_k_only, forward_cuda, touching `_forward_cuda_k_only, forward_cuda`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-14 (20 lines); hunks: -308,14 +308,6 @@ def _get_sum_extend_prefix_lens(forward_batch):; -334,7 +326,7 @@ def _handle_attention_backend(; symbols: _get_sum_extend_prefix_lens, _is_extend_without_speculative, _support_mha_one_shot, _handle_attention_backend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: -410,6 +410,8 @@ def _forward_cuda_k_only(; -555,14 +557,15 @@ def forward_cuda(; symbols: _forward_cuda_k_only, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -308,14 +308,6 @@ def _get_sum_extend_prefix_lens(forward_batch):
-def _is_extend_without_speculative(forward_batch):
-    return (
-        forward_batch.forward_mode.is_extend()
-        and not forward_batch.forward_mode.is_target_verify()
-        and not forward_batch.forward_mode.is_draft_extend()
-    )
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -410,6 +410,8 @@ def _forward_cuda_k_only(
+        assert forward_batch.forward_mode.is_extend_without_speculative()
@@ -555,14 +557,15 @@ def forward_cuda(
-        should_skip = False
-        if not forward_batch.forward_mode.is_decode_or_idle():
+        # We can only skip the logits computation if cuda graph is not involved
+        skip_logits_computation = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +6/-14; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12582 - [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel

- Link: https://github.com/sgl-project/sglang/pull/12582
- Status/date: merged / 2025-11-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +209/-61, 659 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel"; model line: DeepSeek V3/R1; category: bug fix; main diff: `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py`; PR body summary: Part1 of the fix for bug in https://github.com/sgl-project/sglang/issues/11629 In the topk kernel for prefill, the q and k inputs are both ragged. We need to pass the correct st....
- Key implementation: `sgl-kernel/tests/test_topk.py` modified +85/-24 (109 lines); hunks: -1,4 +1,4; -9,21 +9,36; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal, touching `_ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl`; `sgl-kernel/csrc/elementwise/topk.cu` modified +51/-24 (75 lines); hunks: -25,9 +25,10 @@ constexpr int kThreadsPerBlock = 1024;; -72,7 +73,7 @@ __device__ __forceinline__ auto convert_to_uint32(float x) ->...; `sgl-kernel/python/sgl_kernel/top_k.py` modified +61/-7 (68 lines); hunks: -1,3 +1,5; -11,13 +13,32 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_transform_fused, touching `fast_topk, fast_topk_v2, fast_topk_transform_fused`; `sgl-kernel/include/sgl_kernel_ops.h` modified +9/-3 (12 lines); hunks: -172,18 +172,24 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor....
- Code diff details:
  - `sgl-kernel/tests/test_topk.py` modified +85/-24 (109 lines); hunks: -1,4 +1,4; -9,21 +9,36; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +51/-24 (75 lines); hunks: -25,9 +25,10 @@ constexpr int kThreadsPerBlock = 1024;; -72,7 +73,7 @@ __device__ __forceinline__ auto convert_to_uint32(float x) ->...
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +61/-7 (68 lines); hunks: -1,3 +1,5; -11,13 +13,32 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +9/-3 (12 lines); hunks: -172,18 +172,24 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor...
  - `sgl-kernel/csrc/common_extension.cc` modified +3/-3 (6 lines); hunks: -107,15 +107,15 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/tests/test_topk.py
@@ -1,4 +1,4 @@
-from typing import Optional
+from typing import Any, Optional
@@ -9,21 +9,36 @@
-def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
+def _ref_torch_impl(
+    score: torch.Tensor,
diff -- sgl-kernel/csrc/elementwise/topk.cu
@@ -25,9 +25,10 @@ constexpr int kThreadsPerBlock = 1024;
-  const float* __restrict__ input;  // [B, input_stride]
-  int32_t* __restrict__ indices;    // [B, TopK]
-  int32_t* __restrict__ lengths;    // [B]
+  const float* __restrict__ input;         // [B, input_stride]
+  const int32_t* __restrict__ row_starts;  // [B]
+  int32_t* __restrict__ indices;           // [B, TopK]
diff -- sgl-kernel/python/sgl_kernel/top_k.py
@@ -1,3 +1,5 @@
```

- Reviewed files:
  - tests: `sgl-kernel/tests/test_topk.py` modified +85/-24
  - other: `sgl-kernel/csrc/elementwise/topk.cu` modified +51/-24; `sgl-kernel/python/sgl_kernel/top_k.py` modified +61/-7; `sgl-kernel/include/sgl_kernel_ops.h` modified +9/-3; `sgl-kernel/csrc/common_extension.cc` modified +3/-3
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_topk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12689 - [ROCM] Optimized deepseek-r1 model with rmsnorm + fp8 quant fusion

- Link: https://github.com/sgl-project/sglang/pull/12689
- Status/date: merged / 2025-11-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `4a78031a71dd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +331/-22, 474 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCM] Optimized deepseek-r1 model with rmsnorm + fp8 quant fusion"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Fusion of rmsnorm + group scale FP8 quantization (group_size=128) w/o fusion w/ fusion deepseek.py - LayerCommunicator (for hidden states) - DeepseekV2AttentionMLA (for q, k_) f....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +117/-15 (132 lines); hunks: -153,6 +153,8; -1507,13 +1509,14 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, forward_absorb_prepare, forward_npu_sparse_prepare, touching `forward_normal_prepare, forward_absorb_prepare, forward_npu_sparse_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +117/-15 (132 lines); hunks: -153,6 +153,8; -1507,13 +1509,14 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, forward_absorb_prepare, forward_npu_sparse_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -153,6 +153,8 @@
+    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
@@ -1507,13 +1509,14 @@ def forward_normal_prepare(
-            q_lora = self.q_a_layernorm(q)
-            q = self.q_b_proj(q_lora)[0].view(
-                -1, self.num_local_heads, self.qk_head_dim
-            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +117/-15
- Risk and verification: The diff ships test coverage in `test/srt/quant/test_fused_rms_fp8_group_quant.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12583 - [Deepseek V3.2] Fix accuracy bug in the Indexer

- Link: https://github.com/sgl-project/sglang/pull/12583
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `0d4a41842424`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +96/-17, 231 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Fix accuracy bug in the Indexer"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Must merge after https://github.com/sgl-project/sglang/pull/12582 Part 2 of the fix for https://github.com/sgl-project/sglang/issues/11629 This PR also fixed the mtp crash in ht....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +32/-8 (40 lines); hunks: -345,7 +345,10 @@ def _get_topk_ragged(; -368,35 +371,56 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, _forward_cuda_k_only, touching `_get_topk_ragged, _forward_cuda_k_only`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +6/-1 (7 lines); hunks: -140,6 +140,7 @@ def topk_transform(; -148,7 +149,9 @@ def topk_transform(; symbols: topk_transform, touching `topk_transform`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +32/-8 (40 lines); hunks: -345,7 +345,10 @@ def _get_topk_ragged(; -368,35 +371,56 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, _forward_cuda_k_only
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +6/-1 (7 lines); hunks: -140,6 +140,7 @@ def topk_transform(; -148,7 +149,9 @@ def topk_transform(; symbols: topk_transform
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -345,7 +345,10 @@ def _get_topk_ragged(
-        offset = 0
+        q_offset = 0
+        k_offset = 0
@@ -368,35 +371,56 @@ def _get_topk_ragged(
-            ks = torch.full((extend_seq_len,), offset, dtype=torch.int32, device="cuda")
-            ke = ks + seq_lens_expanded[offset : offset + extend_seq_len]
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -140,6 +140,7 @@ def topk_transform(
+        ks: Optional[torch.Tensor] = None,
@@ -148,7 +149,9 @@ def topk_transform(
-            return fast_topk_v2(logits, self.get_seqlens_expanded(), topk)
+            return fast_topk_v2(
+                logits, self.get_seqlens_expanded(), topk, row_starts=ks
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +32/-8; `python/sglang/srt/layers/attention/nsa_backend.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_deepseek_v32_basic.py`, `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_nsabackend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12215 - [DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat`

- Link: https://github.com/sgl-project/sglang/pull/12215
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `4eda9969e8b9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-6, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat`"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: https://github.com/sgl-project/sglang/issues/11989 `torch.cat([q_nope, q_rope], dim=-1)` is heavily used in `nsa_backend`, which is less efficient. Use existed kernel `_concat_m....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +7/-6 (13 lines); hunks: -21,6 +21,7; -911,7 +912,7 @@ def forward_extend(; symbols: forward_extend, forward_decode, touching `forward_extend, forward_decode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +7/-6 (13 lines); hunks: -21,6 +21,7; -911,7 +912,7 @@ def forward_extend(; symbols: forward_extend, forward_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -21,6 +21,7 @@
+from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general
@@ -911,7 +912,7 @@ def forward_extend(
-                q_all = torch.cat([q_nope, q_rope], dim=-1)
+                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
@@ -921,7 +922,7 @@ def forward_extend(
-                q_all = torch.cat([q_nope, q_rope], dim=-1)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +7/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11589 - [Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector

- Link: https://github.com/sgl-project/sglang/pull/11589
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `fc5da1e80b78`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-9, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Call] Steamline function arguments when tool_choice="auto" for deepseekv31_detector"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`; PR body summary: The current `deepseekv31_detector` only detects complete function arguments and immediately returns them while clearing the buffer. This approach can lead to a poor user experie....
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

### PR #13236 - [Deepseek V3.2] Clean up MTP

- Link: https://github.com/sgl-project/sglang/pull/13236
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `a7002e614bbd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +68/-76, 248 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Clean up MTP"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: - Fix an accuracy bug in MTP draft_extend stage. The `seqlens_int32_cpu` used in the calculations of `seqlens_expanded` is incorrect. We should use `seq_lens_cpu`, instead of `s....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +59/-73 (132 lines); hunks: -137,6 +137,9 @@ def get_page_table_64(self) -> torch.Tensor:; -304,8 +307,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: get_page_table_64, get_seqlens_expanded, get_cu_seqlens_k, topk_transform, touching `get_page_table_64, get_seqlens_expanded, get_cu_seqlens_k`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-1 (8 lines); hunks: -290,7 +290,10 @@ def _get_topk_paged(; -337,6 +340,8 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _get_topk_ragged, forward_cuda, touching `_get_topk_paged, _get_topk_ragged, forward_cuda`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +59/-73 (132 lines); hunks: -137,6 +137,9 @@ def get_page_table_64(self) -> torch.Tensor:; -304,8 +307,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: get_page_table_64, get_seqlens_expanded, get_cu_seqlens_k, topk_transform
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-1 (8 lines); hunks: -290,7 +290,10 @@ def _get_topk_paged(; -337,6 +340,8 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _get_topk_ragged, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -137,6 +137,9 @@ def get_page_table_64(self) -> torch.Tensor:
+    def get_cu_seqlens_k(self) -> torch.Tensor:
+        return self.attn_metadata.cu_seqlens_k
@@ -304,8 +307,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            max_seqlen_q = self.speculative_num_draft_tokens
-            nsa_max_seqlen_q = self.speculative_num_draft_tokens
+            max_seqlen_q = 1
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -290,7 +290,10 @@ def _get_topk_paged(
-        if forward_batch.forward_mode.is_target_verify():
+        if (
+            forward_batch.forward_mode.is_target_verify()
+            or forward_batch.forward_mode.is_draft_extend()
+        ):
@@ -337,6 +340,8 @@ def _get_topk_ragged(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +59/-73; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12065 - (1/n)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/12065
- Status/date: merged / 2025-11-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `d368c7451a48`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1247/-54, 1729 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "(1/n)support context parallel with deepseekv3.2-DSA"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. **Main design ideas：** Taking TP=EP=4 and....
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +305/-0 (305 lines); hunks: -1,4 +1,13; -21,3 +30,299 @@ def print_nsa_bool_env_vars():; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens, is_nsa_enable_prefill_cp, NSAContextParallelMetadata, touching `print_nsa_bool_env_vars, compute_nsa_seqlens, is_nsa_enable_prefill_cp`; `python/sglang/srt/layers/communicator_nsa_cp.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: nsa_enable_prefill_cp, NSACPLayerCommunicator, __init__, NSACPCommunicateSimpleFn, touching `nsa_enable_prefill_cp, NSACPLayerCommunicator, __init__`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +221/-8 (229 lines); hunks: -1,7 +1,7; -16,9 +16,18; symbols: __init__, _get_q_k_bf16, _forward_cuda_k_only, _get_topk_ragged_with_cp, touching `__init__, _get_q_k_bf16, _forward_cuda_k_only`; `python/sglang/srt/models/deepseek_v2.py` modified +134/-32 (166 lines); hunks: -54,13 +54,23; -412,7 +422,9 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, __init__, forward, touching `handle_attention_nsa, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +305/-0 (305 lines); hunks: -1,4 +1,13; -21,3 +30,299 @@ def print_nsa_bool_env_vars():; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens, is_nsa_enable_prefill_cp, NSAContextParallelMetadata
  - `python/sglang/srt/layers/communicator_nsa_cp.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: nsa_enable_prefill_cp, NSACPLayerCommunicator, __init__, NSACPCommunicateSimpleFn
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +221/-8 (229 lines); hunks: -1,7 +1,7; -16,9 +16,18; symbols: __init__, _get_q_k_bf16, _forward_cuda_k_only, _get_topk_ragged_with_cp
  - `python/sglang/srt/models/deepseek_v2.py` modified +134/-32 (166 lines); hunks: -54,13 +54,23; -412,7 +422,9 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, __init__, forward
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +28/-8 (36 lines); hunks: -145,32 +145,52 @@ def topk_transform(; symbols: topk_transform
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -1,4 +1,13 @@
+from dataclasses import dataclass
+from itertools import accumulate
+from typing import List
+import torch
+import torch.nn.functional as F
+from sglang.srt.layers.dp_attention import get_attention_tp_group
diff -- python/sglang/srt/layers/communicator_nsa_cp.py
@@ -0,0 +1,284 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,7 +1,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +305/-0; `python/sglang/srt/layers/communicator_nsa_cp.py` added +284/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +221/-8; `python/sglang/srt/models/deepseek_v2.py` modified +134/-32; `python/sglang/srt/layers/attention/nsa_backend.py` modified +28/-8
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_cp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13022 - [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa

- Link: https://github.com/sgl-project/sglang/pull/13022
- Status/date: merged / 2025-11-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `a8fcbf6fe3a2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-1, 37 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Use torch.compile to speed up torch.cat used in prefill. Replace the cat ops for k in `forward_extend` in the nsa attention with the torch.compiled version. before after.
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +22/-1 (23 lines); hunks: -123,6 +123,27 @@ class TopkTransformMethod(IntEnum):; -942,7 +963,7 @@ def forward_extend(; symbols: TopkTransformMethod, _compiled_cat, _cat, NSAIndexerMetadata, touching `TopkTransformMethod, _compiled_cat, _cat`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +22/-1 (23 lines); hunks: -123,6 +123,27 @@ class TopkTransformMethod(IntEnum):; -942,7 +963,7 @@ def forward_extend(; symbols: TopkTransformMethod, _compiled_cat, _cat, NSAIndexerMetadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -123,6 +123,27 @@ class TopkTransformMethod(IntEnum):
+@torch.compile
+def _compiled_cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
+    return torch.cat(tensors, dim=dim)
+def _cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
+    """
+    Concatenate two tensors along the last dimension.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +22/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13264 - [NVIDIA] Fix broken fp8 MoE of deepseek v3

- Link: https://github.com/sgl-project/sglang/pull/13264
- Status/date: merged / 2025-11-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `e389f91decda`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +117/-4, 171 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Fix broken fp8 MoE of deepseek v3"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This breaks the fp8 moe of the deepseek v3, causing: the root cause is now the routing_method_type becomes a default attribute of FusedMoE layer. So, we should explicitly set it....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -93,6 +93,7; -674,6 +675,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -93,6 +93,7; -674,6 +675,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -93,6 +93,7 @@
+from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -674,6 +675,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.DeepSeekV3,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_r1_fp8_trtllm_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13548 - [Fix] Fix DeepSeek V3 MTP on B200

- Link: https://github.com/sgl-project/sglang/pull/13548
- Status/date: merged / 2025-11-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Fix DeepSeek V3 MTP on B200"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_nextn.py`; PR body summary: Fix the bug here https://github.com/sgl-project/sglang/actions/runs/19449452397/job/55665254802 Test with GSM8K.
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +1/-0 (1 lines); hunks: -220,6 +220,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-0 (1 lines); hunks: -220,6 +220,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -220,6 +220,7 @@ def __init__(
+        self._executed_weight_requant_ue8m0 = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13617 - [ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel

- Link: https://github.com/sgl-project/sglang/pull/13617
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `c8ede0e93c3a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +61/-15, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCM] Optimized deepseek-r1 fp8 model with + triton_gemm_a8w8 + batch_gemm_a8w8 + fused set_mla_kv_buffer kernel"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Co-author : @yichiche To optimize the deepseek-r1 model performance on ROCM. This PR improves the performance of a8w8 GEMM and enable batched_gemm and mla_kv_buffer feature on M....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +57/-14 (71 lines); hunks: -167,6 +167,9; -1813,10 +1816,25 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, forward_npu_sparse_prepare, _set_mla_kv_buffer, touching `forward_absorb_prepare, forward_absorb_core, forward_npu_sparse_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +57/-14 (71 lines); hunks: -167,6 +167,9; -1813,10 +1816,25 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, forward_npu_sparse_prepare, _set_mla_kv_buffer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -167,6 +167,9 @@
+    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
+        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
+    )
@@ -1813,10 +1816,25 @@ def forward_absorb_prepare(
-                q_nope_out = torch.bmm(
-                    q_nope.to(torch.bfloat16).transpose(0, 1),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +57/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12964 - [DeepseekV3.2] Deepseek fp8 support for MHA path

- Link: https://github.com/sgl-project/sglang/pull/12964
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `fa924410276d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +55/-9, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepseekV3.2] Deepseek fp8 support for MHA path"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Enable FP8 KV cache support for MHA (Multi-Head Attention) path in DeepSeek NSA backend. Previously, MHA could only be used when KV cache dtype is BF16, limiting its usage with....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +46/-8 (54 lines); hunks: -53,6 +53,7; -418,8 +419,6 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, forward_normal_prepare, _get_mla_kv_buffer, _get_mla_kv_buffer_from_fp8, touching `handle_attention_nsa, forward_normal_prepare, _get_mla_kv_buffer`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-1 (10 lines); hunks: -455,7 +455,14 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -468,6 +475,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, touching `init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +46/-8 (54 lines); hunks: -53,6 +53,7; -418,8 +419,6 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, forward_normal_prepare, _get_mla_kv_buffer, _get_mla_kv_buffer_from_fp8
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-1 (10 lines); hunks: -455,7 +455,14 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -468,6 +475,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -53,6 +53,7 @@
+from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
@@ -418,8 +419,6 @@ def handle_attention_nsa(attn, forward_batch):
-    TODO: B200 (SM100) MHA path is temporarily disabled due to FA4 gpqa accuracy issues.
@@ -434,10 +433,17 @@ def handle_attention_nsa(attn, forward_batch):
-        # Check if kvcache dtype is bfloat16
-        kv_dtype_is_bf16 = forward_batch.token_to_kv_pool.dtype == torch.bfloat16
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -455,7 +455,14 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            if topk_transform_method == TopkTransformMethod.RAGGED:
+            # Generate page_table_1_flattened when needed:
+            mha_dequantize_needed = (
+                self.nsa_kv_cache_store_fp8 and max_seqlen_k <= self.nsa_index_topk
+            )
+            if (
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +46/-8; `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13459 - [Deepseek V3.2] Change indexer weights_proj to fp32

- Link: https://github.com/sgl-project/sglang/pull/13459
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `7291c72e575d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +92/-124, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Change indexer weights_proj to fp32"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Following https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/commit/8631a813356d39b09a5c1ee1cde8ed6015559eb3, change the indexer `weights_proj` precision from bf16 to fp32. https:....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunks: -109,7 +109,6 @@ def __init__(; -120,7 +119,6 @@ def __init__(; symbols: __init__, _get_logits_head_gate, touching `__init__, _get_logits_head_gate`; `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunks: -239,17 +239,6 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1226,9 +1215,6 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__, touching `add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunks: -109,7 +109,6 @@ def __init__(; -120,7 +119,6 @@ def __init__(; symbols: __init__, _get_logits_head_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunks: -239,17 +239,6 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1226,9 +1215,6 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -109,7 +109,6 @@ def __init__(
-        fuse_wk_and_weights_proj: bool = False,
@@ -120,7 +119,6 @@ def __init__(
-        self.fuse_wk_and_weights_proj = fuse_wk_and_weights_proj
@@ -139,28 +137,22 @@ def __init__(
-        if self.fuse_wk_and_weights_proj:
-            self.fused_wk_and_weights_proj = ReplicatedLinear(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -239,17 +239,6 @@ def add_forward_absorb_core_attention_backend(backend_name):
-def is_nsa_indexer_wk_and_weights_proj_fused(config, quant_config):
-    """
-    NSA Indexer wk and weights_proj can be fused in FP4 model because they are both in BF16
-    """
-    return (
-        is_deepseek_nsa(config)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53; `python/sglang/srt/models/deepseek_v2.py` modified +0/-71
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13705 - [AMD] Enable fused shared expert append and flatten quant for fp8 deepseekR1 model

- Link: https://github.com/sgl-project/sglang/pull/13705
- Status/date: merged / 2025-11-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `eff7df6d0a49`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +92/-26, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Enable fused shared expert append and flatten quant for fp8 deepseekR1 model"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Co-author : @yctseng0211 This PR introduces two performance improvements for DeepseekR1-0528 (fp8): 1. Support aiter fused_flatten_fp8_group_quant for fp8 DeepSeek model Adds a....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +10/-1 (11 lines); hunks: -168,10 +168,14; -2001,6 +2005,11 @@ def forward_absorb_core(; symbols: forward_absorb_core, touching `forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-1 (11 lines); hunks: -168,10 +168,14; -2001,6 +2005,11 @@ def forward_absorb_core(; symbols: forward_absorb_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -168,10 +168,14 @@
-    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
+    from aiter.ops.triton.fused_fp8_quant import (
+        fused_flatten_fp8_group_quant,
+        fused_rms_fp8_group_quant,
+    )
@@ -2001,6 +2005,11 @@ def forward_absorb_core(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +10/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10027 - [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels

- Link: https://github.com/sgl-project/sglang/pull/10027
- Status/date: merged / 2025-11-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +253/-77, 488 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Optimize DeepSeek-R1 w4afp8 glue kernels"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; PR body summary: This PR improves DeepSeek-R1 w4afp8 TP8/EP8 ITL performance. It is motivated by the following profiling trace, obtained by running DeepSeek-R1 w4afp8 TP8 with a concurrency leve....
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +227/-54 (281 lines); hunks: -16,6 +16,60; -142,25 +196,17 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg...; symbols: _get_launch_config_1d, get_num_blocks, _get_launch_config_2d, deepep_permute_triton_kernel, touching `_get_launch_config_1d, get_num_blocks, _get_launch_config_2d`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +21/-17 (38 lines); hunks: -10,15 +10,17; -44,6 +46,7 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe, touching `cutlass_w4a8_moe`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +5/-6 (11 lines); hunks: -270,17 +270,17 @@ def process_weights_after_loading(self, layer: Module) ->...; -324,9 +324,8 @@ def apply(; symbols: process_weights_after_loading, apply, apply_deepep_ll, touching `process_weights_after_loading, apply, apply_deepep_ll`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +227/-54 (281 lines); hunks: -16,6 +16,60; -142,25 +196,17 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg...; symbols: _get_launch_config_1d, get_num_blocks, _get_launch_config_2d, deepep_permute_triton_kernel
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +21/-17 (38 lines); hunks: -10,15 +10,17; -44,6 +46,7 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +5/-6 (11 lines); hunks: -270,17 +270,17 @@ def process_weights_after_loading(self, layer: Module) ->...; -324,9 +324,8 @@ def apply(; symbols: process_weights_after_loading, apply, apply_deepep_ll
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -16,6 +16,60 @@
+def _get_launch_config_1d(device, numel):
+    MAX_THREADS_PER_BLOCK = 1024
+    MIN_THREADS_PER_BLOCK = 512
+    MAX_WAVES = 8  # empirical numbers
+    props = torch.cuda.get_device_properties(device)
+    sm_count = props.multi_processor_count
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -10,15 +10,17 @@
+from sglang.srt.distributed import get_moe_expert_parallel_world_size
+    cutlass_w4_run_moe_ep_preproess,
-    post_reorder_triton_kernel_for_cutlass_moe,
-    pre_reorder_triton_kernel_for_cutlass_moe,
-    run_moe_ep_preproess,
+    post_reorder_for_cutlass_moe,
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -270,17 +270,17 @@ def process_weights_after_loading(self, layer: Module) -> None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +227/-54; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +21/-17; `python/sglang/srt/layers/quantization/w4afp8.py` modified +5/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13544 - [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend

- Link: https://github.com/sgl-project/sglang/pull/13544
- Status/date: merged / 2025-11-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `5eed5fc0b091`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +74/-78, 342 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: NativeSparseAttnBackend currently spreads dispatch logic for NSA prefill/decode implementations and MHA vs. MLA selection across multiple places: - Global `NSA_PREFILL_IMPL` / `....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +69/-42 (111 lines); hunks: -20,6 +20,7; -228,9 +229,6 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, init_forward_metadata, touching `compute_cu_seqlens, NativeSparseAttnBackend, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +5/-36 (41 lines); hunks: -417,43 +417,12 @@ def handle_attention_aiter(attn, forward_batch):; symbols: handle_attention_aiter, handle_attention_nsa, touching `handle_attention_aiter, handle_attention_nsa`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +69/-42 (111 lines); hunks: -20,6 +20,7; -228,9 +229,6 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, init_forward_metadata
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-36 (41 lines); hunks: -417,43 +417,12 @@ def handle_attention_aiter(attn, forward_batch):; symbols: handle_attention_aiter, handle_attention_nsa
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -20,6 +20,7 @@
+    is_nsa_enable_prefill_cp,
@@ -228,9 +229,6 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
-NSA_PREFILL_IMPL: _NSA_IMPL_T
-NSA_DECODE_IMPL: _NSA_IMPL_T
@@ -264,10 +262,12 @@ def __init__(
-        global NSA_PREFILL_IMPL, NSA_DECODE_IMPL
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -417,43 +417,12 @@ def handle_attention_aiter(attn, forward_batch):
-    Select MHA or MLA based on sequence length for optimal performance.
-    - Decode: MLA (avoids per-token decompression)
-    - Prefill <= 2048: MHA (topk ineffective, MHA has lower FLOPs)
-    - Prefill > 2048: MLA (topk filtering reduces computation significantly)
+    Dispatch logic is centralized in NativeSparseAttnBackend.set_nsa_prefill_impl and executed
+    in init_forward_metadata. Read the decision from backend.use_mha.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +69/-42; `python/sglang/srt/models/deepseek_v2.py` modified +5/-36
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13954 - Fix Deepseek v3.1 loading issue

- Link: https://github.com/sgl-project/sglang/pull/13954
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `13e5beeab499`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Deepseek v3.1 loading issue"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Fix nightly test error in https://github.com/sgl-project/sglang/actions/runs/19689716521/job/56402800532.
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

### PR #13646 - [DeepSeekV3.2] Enable pure TP & Partial DP Attention

- Link: https://github.com/sgl-project/sglang/pull/13646
- Status/date: merged / 2025-11-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `decb48965dd1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +286/-24, 460 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeekV3.2] Enable pure TP & Partial DP Attention"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: DeepSeekV3.2 NSA currently has rough edges when running in **pure TP mode** (`dp_size < tp_size`): - FlashMLA sparse can see an invalid `num_heads` per rank after TP sharding. -....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunks: -330,6 +330,25 @@ def _get_topk_paged(; -409,24 +428,86 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _forward_cuda_k_only, touching `_get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunks: -170,14 +170,18 @@ def topk_transform(; -286,9 +290,11 @@ def __init__(; symbols: topk_transform, __init__, forward_extend, forward_decode, touching `topk_transform, __init__, forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunks: -330,6 +330,25 @@ def _get_topk_paged(; -409,24 +428,86 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _forward_cuda_k_only
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunks: -170,14 +170,18 @@ def topk_transform(; -286,9 +290,11 @@ def __init__(; symbols: topk_transform, __init__, forward_extend, forward_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -330,6 +330,25 @@ def _get_topk_paged(
+    def _should_chunk_mqa_logits(
+        self, num_q: int, num_k: int, device: torch.device
+    ) -> Tuple[bool, int]:
+        """
+        Detect whether we need to chunk the MQA logits computation to avoid OOM
+        Return: (need_chunk, free_mem)
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -170,14 +170,18 @@ def topk_transform(
+        topk_indices_offset_override: Optional[torch.Tensor] = None,
-        if cu_seqlens_q is not None:
+        if topk_indices_offset_override is not None:
+            cu_topk_indices_offset = topk_indices_offset_override
+            cu_seqlens_q_topk = None
+        elif cu_seqlens_q is not None:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14; `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_deepseek_v32_perf.py`, `test/nightly/test_deepseek_v32_nsabackend.py`, `test/nightly/test_deepseek_v32_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14194 - [feature] implement dcp for deepseek_v2

- Link: https://github.com/sgl-project/sglang/pull/14194
- Status/date: open / 2025-12-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +1363/-86, 2098 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feature] implement dcp for deepseek_v2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; PR body summary: Here's the first step to fully implement #12196 to support much longer context with TP 8 under 8xH20. Currently, it only works with attention backend flashinfer. It's compatible....
- Key implementation: `python/sglang/srt/layers/attention/utils.py` modified +254/-0 (254 lines); hunks: -1,7 +1,13; -1391,3 +1397,251 @@ def fused_qk_rope_reshape_and_cache(; symbols: fused_qk_rope_reshape_and_cache, _correct_attn_cp_out_kernel, CPTritonContext, __init__, touching `fused_qk_rope_reshape_and_cache, _correct_attn_cp_out_kernel, CPTritonContext`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +136/-0 (136 lines); hunks: -39,6 +39,8; -54,6 +56,7; symbols: ForwardBatch, init_new, _clamp_position_native, create_dcp_kv_indices, touching `ForwardBatch, init_new, _clamp_position_native`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +93/-14 (107 lines); hunks: -1,12 +1,21; -28,6 +37,8; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +83/-7 (90 lines); hunks: -16,11 +16,13; -334,6 +336,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph, forward_extend, forward_decode, touching `init_forward_metadata, init_forward_metadata_replay_cuda_graph, forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/utils.py` modified +254/-0 (254 lines); hunks: -1,7 +1,13; -1391,3 +1397,251 @@ def fused_qk_rope_reshape_and_cache(; symbols: fused_qk_rope_reshape_and_cache, _correct_attn_cp_out_kernel, CPTritonContext, __init__
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +136/-0 (136 lines); hunks: -39,6 +39,8; -54,6 +56,7; symbols: ForwardBatch, init_new, _clamp_position_native, create_dcp_kv_indices
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +93/-14 (107 lines); hunks: -1,12 +1,21; -28,6 +37,8; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +83/-7 (90 lines); hunks: -16,11 +16,13; -334,6 +336,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph, forward_extend, forward_decode
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +67/-5 (72 lines); hunks: -4,6 +4,11; -233,11 +238,46 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, _chunked_prefix_attn_mha, _all_gather_dcp_kv_cache, _set_mla_kv_buffer
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/utils.py
@@ -1,7 +1,13 @@
+from typing import Optional
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
+)
+from sglang.srt.distributed.parallel_state import GroupCoordinator
@@ -1391,3 +1397,251 @@ def fused_qk_rope_reshape_and_cache(
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -39,6 +39,8 @@
+    get_dcp_rank,
+    get_dcp_world_size,
@@ -54,6 +56,7 @@
+    create_chunked_prefix_cache_kv_indices,
@@ -438,6 +441,12 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):
+    # For decode context parallel
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -1,12 +1,21 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/utils.py` modified +254/-0; `python/sglang/srt/model_executor/forward_batch_info.py` modified +136/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +93/-14; `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +83/-7; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +67/-5; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +48/-19
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/device_communicators/pynccl_allocator.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/entrypoints/engine.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14325 - [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200

- Link: https://github.com/sgl-project/sglang/pull/14325
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `7dfcc78155b6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +195/-63, 300 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: In `extend` mode with NSA indexer enabled, H200 setups that select **PAGED** top-k transform (`flashmla_kv` + FP8 KV cache) may trigger chunked execution in `_get_topk_ragged` w....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-5 (31 lines); hunks: -370,6 +370,8 @@ def _get_topk_ragged(; -401,6 +403,7 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, touching `_get_topk_ragged`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-5 (31 lines); hunks: -370,6 +370,8 @@ def _get_topk_ragged(; -401,6 +403,7 @@ def _get_topk_ragged(; symbols: _get_topk_ragged
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -370,6 +370,8 @@ def _get_topk_ragged(
+        # Token-to-batch mapping for PAGED chunk alignment
+        token_to_batch_idx: List[int] = []
@@ -401,6 +403,7 @@ def _get_topk_ragged(
+            token_to_batch_idx.extend([i] * extend_seq_len)
@@ -473,6 +476,13 @@ def _get_topk_ragged(
+        # Only materialize batch index tensor when PAGED path needs it
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-5
- Risk and verification: The diff ships test coverage in `test/nightly/test_deepseek_v32_nsabackend.py`, `test/nightly/test_deepseek_v32_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13115 - support mtp with deepseek r1 nvfp4 model

- Link: https://github.com/sgl-project/sglang/pull/13115
- Status/date: merged / 2025-12-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `ea177372bd8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +123/-59, 459 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support mtp with deepseek r1 nvfp4 model"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: collabrate with @trevor-m Support large scale EP deployment for the DS R1 fp4 model with eagle spec decoding. - add the custom moe a2a backend for speculative decoding - fix the....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +3/-1 (4 lines); hunks: -2502,7 +2502,9 @@ def forward_normal_chunked_kv_prepare(; symbols: forward_normal_chunked_kv_prepare, forward_normal_chunked_kv_core, touching `forward_normal_chunked_kv_prepare, forward_normal_chunked_kv_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-1 (4 lines); hunks: -2502,7 +2502,9 @@ def forward_normal_chunked_kv_prepare(; symbols: forward_normal_chunked_kv_prepare, forward_normal_chunked_kv_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2502,7 +2502,9 @@ def forward_normal_chunked_kv_prepare(
-        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
+        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
+            forward_batch.extend_prefix_lens_cpu
+        )
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14573 - [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode

- Link: https://github.com/sgl-project/sglang/pull/14573
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `b7b7524e9560`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +144/-7, 165 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`; PR body summary: When using `--tool-call-parser deepseekv32` in streaming mode, tool calls for functions with no parameters (e.g., `get_date()`) were being silently skipped, while non-streaming....
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-7 (9 lines); hunks: -241,13 +241,8 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-7 (9 lines); hunks: -241,13 +241,8 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -241,13 +241,8 @@ def parse_streaming_increment(
-                    # Check if invoke_content is empty or whitespace only
-                    # If so, skip this tool call entirely (it's likely incomplete or malformed)
-                    if not invoke_content.strip():
-                        # Remove the incomplete tool call from buffer
-                        self._buffer = current_text[invoke_match.end() :]
-                        current_text = self._buffer
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-7
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14837 - [Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)

- Link: https://github.com/sgl-project/sglang/pull/14837
- Status/date: merged / 2025-12-10
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv31.jinja`; associated commits `ef1ab2302ab2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Auto Sync] Update tool_chat_template_deepseekv31.jinja (20251210)"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `examples/chat_template/tool_chat_template_deepseekv31.jinja`; PR body summary: Sync changes from commit `abd3e13d`. **Files Changed:** - examples/chat_template/tool_chat_template_deepseekv31.jinja Author: Jue Wang *This is an automated PR created by script....
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

### PR #14541 - [NPU]dsv3.2 cp for npu

- Link: https://github.com/sgl-project/sglang/pull/14541
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`; associated commits `388018a5bd41`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +281/-134, 587 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]dsv3.2 cp for npu"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`; PR body summary: Now, sglang already has the --enable-nsa-prefill-context-parallel option proposed to support CP parallelism, (1/n)support context parallel with deepseekv3.2-DSA , but NPU not su....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +117/-94 (211 lines); hunks: -23,11 +23,7; -965,56 +961,25 @@ def forward_npu(; symbols: forward_npu, do_npu_cp_balance_indexer, touching `forward_npu, do_npu_cp_balance_indexer`; `python/sglang/srt/layers/attention/nsa/utils.py` modified +25/-4 (29 lines); hunks: -49,6 +49,10 @@ class NSAContextParallelMetadata:; -312,17 +316,34 @@ def prepare_input_dp_with_cp_dsa(; symbols: NSAContextParallelMetadata, prepare_input_dp_with_cp_dsa, touching `NSAContextParallelMetadata, prepare_input_dp_with_cp_dsa`; `python/sglang/srt/layers/communicator_nsa_cp.py` modified +7/-8 (15 lines); hunks: -176,14 +176,13 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual, _scatter_hidden_states_and_residual, touching `_gather_hidden_states_and_residual, _scatter_hidden_states_and_residual`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +9/-0 (9 lines); hunks: -311,8 +311,17 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, touching `forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +117/-94 (211 lines); hunks: -23,11 +23,7; -965,56 +961,25 @@ def forward_npu(; symbols: forward_npu, do_npu_cp_balance_indexer
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +25/-4 (29 lines); hunks: -49,6 +49,10 @@ class NSAContextParallelMetadata:; -312,17 +316,34 @@ def prepare_input_dp_with_cp_dsa(; symbols: NSAContextParallelMetadata, prepare_input_dp_with_cp_dsa
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +7/-8 (15 lines); hunks: -176,14 +176,13 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual, _scatter_hidden_states_and_residual
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +9/-0 (9 lines); hunks: -311,8 +311,17 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
  - `python/sglang/srt/hardware_backend/npu/utils.py` modified +8/-0 (8 lines); hunks: -68,6 +68,14 @@ def init_npu_backend():; symbols: init_npu_backend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -23,11 +23,7 @@
-from sglang.srt.layers.dp_attention import (
-    get_attention_tp_group,
-    get_attention_tp_rank,
-    get_attention_tp_size,
-)
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -49,6 +49,10 @@ class NSAContextParallelMetadata:
+    kv_len_prev_tensor: torch.Tensor = None
+    kv_len_next_tensor: torch.Tensor = None
+    actual_seq_q_prev_tensor: torch.Tensor = None
+    actual_seq_q_next_tensor: torch.Tensor = None
@@ -312,17 +316,34 @@ def prepare_input_dp_with_cp_dsa(
+    kv_len_prev = prefix_sum_list[cp_rank]
diff -- python/sglang/srt/layers/communicator_nsa_cp.py
@@ -176,14 +176,13 @@ def _gather_hidden_states_and_residual(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +117/-94; `python/sglang/srt/layers/attention/nsa/utils.py` modified +25/-4; `python/sglang/srt/layers/communicator_nsa_cp.py` modified +7/-8; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +9/-0; `python/sglang/srt/hardware_backend/npu/utils.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `test/srt/ascend/test_ascend_tp4_bf16.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14897 - Fix dsv3 dp accuracy issue when using bf16-kv

- Link: https://github.com/sgl-project/sglang/pull/14897
- Status/date: merged / 2025-12-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-2, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix dsv3 dp accuracy issue when using bf16-kv"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`; PR body summary: cherry-pick "Fix dp issue dsv3 bf16 kv #14895" to `amd_mori` branch.
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +8/-2 (10 lines); hunks: -178,12 +178,18 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +8/-2 (10 lines); hunks: -178,12 +178,18 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -178,12 +178,18 @@ def __init__(
+            global _use_mla_ps_kernel, fast_mode, intra_batch_mode
+            self.enable_dp_attention = is_dp_attention_enabled()
+            if self.kv_cache_dtype is not fp8_dtype and self.enable_dp_attention:
+                _use_mla_ps_kernel = False
+                fast_mode = False
+                intra_batch_mode = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14572 - [NPU] optimization for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/14572
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `c05d3afb5d8b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +141/-68, 393 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] optimization for dsv3.2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; PR body summary: Co-author: @jiaming1130 In this PR, we optimized the performance of the DeepSeek-V3.2 on NPU to improve the performance of the network in the decode phase, and also adapted the....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +51/-15 (66 lines); hunks: -10,12 +10,18; -980,19 +986,47 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`; `python/sglang/srt/models/deepseek_v2.py` modified +25/-4 (29 lines); hunks: -700,6 +700,7 @@ def __init__(; -738,7 +739,11 @@ def __init__(; symbols: __init__, forward_deepep, _post_combine_hook, touching `__init__, forward_deepep, _post_combine_hook`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-18 (52 lines); hunks: -273,39 +273,51 @@ def forward_dsa_prepare_npu(; -367,7 +379,11 @@ def forward_dsa_core_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_core_npu, touching `forward_dsa_prepare_npu, forward_dsa_core_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +51/-15 (66 lines); hunks: -10,12 +10,18; -980,19 +986,47 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +25/-4 (29 lines); hunks: -700,6 +700,7 @@ def __init__(; -738,7 +739,11 @@ def __init__(; symbols: __init__, forward_deepep, _post_combine_hook
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-18 (52 lines); hunks: -273,39 +273,51 @@ def forward_dsa_prepare_npu(; -367,7 +379,11 @@ def forward_dsa_core_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_core_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -10,12 +10,18 @@
+global _use_multi_stream
+if is_npu():
+    import custom_ops  # noqa: F401
+    import torch_npu
+    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream
@@ -980,19 +986,47 @@ def forward_npu(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -700,6 +700,7 @@ def __init__(
+                    or get_moe_a2a_backend().is_ascend_fuseep()
@@ -738,7 +739,11 @@ def __init__(
-        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
+        if (
+            get_moe_a2a_backend().is_deepep()
+            or get_moe_a2a_backend().is_mooncake()
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -273,39 +273,51 @@ def forward_dsa_prepare_npu(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +51/-15; `python/sglang/srt/models/deepseek_v2.py` modified +25/-4; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-18
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15086 - [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache

- Link: https://github.com/sgl-project/sglang/pull/15086
- Status/date: merged / 2025-12-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `c96903074c4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-5, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`; PR body summary: Fix NSA backend assertion error when running DeepSeek-V3.2 triggered when running DeepSeek-V3.2 PP with radix-cache. The assertion `num_tokens <= total_num_tokens` in `dequant_k....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +17/-2 (19 lines); hunks: -463,14 +463,16 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -486,6 +488,19 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, touching `init_forward_metadata`; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-3 (5 lines); hunks: -189,10 +189,9 @@ def dequantize_k_cache_paged(; symbols: dequantize_k_cache_paged, touching `dequantize_k_cache_paged`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +17/-2 (19 lines); hunks: -463,14 +463,16 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -486,6 +488,19 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata
  - `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-3 (5 lines); hunks: -189,10 +189,9 @@ def dequantize_k_cache_paged(; symbols: dequantize_k_cache_paged
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -463,14 +463,16 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            # Check if MHA with FP8 needs page_table_1_flattened for dequantization
+            # Check if MHA FP8 dequantization is needed
-            if (
+            # page_table_1_flattened is only used when prefix sharing is enabled:
+            has_prefix_sharing = any(forward_batch.extend_prefix_lens_cpu)
+            if has_prefix_sharing and (
diff -- python/sglang/srt/layers/attention/nsa/dequant_k_cache.py
@@ -189,10 +189,9 @@ def dequantize_k_cache_paged(
-    total_num_tokens, _ = quant_k_cache.shape
+    # num_tokens can exceed kv_cache_size due to prefix sharing (multiple seqs share same KV slots)
+    # Index bounds validated in nsa_backend.init_forward_metadata
-    assert num_tokens <= total_num_tokens
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +17/-2; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14975 - [AMD] Support fused_rms_mxfp4_quant in the prefill stage for DeepSeek-R1-MXFP4

- Link: https://github.com/sgl-project/sglang/pull/14975
- Status/date: merged / 2025-12-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `8ac350f335c6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-1, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Support fused_rms_mxfp4_quant in the prefill stage for DeepSeek-R1-MXFP4"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Adopted the fused kernel fused_rms_mxfp4_quant to replace the existing RMSNorm + quant path, enabling more efficient preparation of activations for quantized projections. While....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -1722,7 +1722,17 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, touching `forward_normal_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -1722,7 +1722,17 @@ def forward_normal_prepare(; symbols: forward_normal_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1722,7 +1722,17 @@ def forward_normal_prepare(
+            elif _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
+                # MXFP4: fused RMSNorm + quant
+                q, _, _, _ = fused_rms_mxfp4_quant(
+                    q,
+                    self.q_a_layernorm.weight,
+                    self.q_a_layernorm.variance_epsilon,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15315 - [Performance] Optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe

- Link: https://github.com/sgl-project/sglang/pull/15315
- Status/date: open / 2025-12-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +58/-11, 205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] Optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu`; PR body summary: While deploying and optimizing the w4a8 moe group gemm for DeepSeek-R1-W4AFP8, I found that when w4a8 is in deepep low_latency mode, only the final else branch of the groupgemm....
- Key implementation: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +6/-0 (6 lines); hunks: -170,6 +170,7 @@ def cutlass_w4a8_moe(; -193,6 +194,7 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, touching `cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll`; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +38/-6 (44 lines); hunks: -79,10 +79,12 @@ void dispatch_w4a8_moe_mm_sm90(; -114,6 +116,20 @@ void dispatch_w4a8_moe_mm_sm90(; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3 (9 lines); hunks: -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(; -49,7 +50,8 @@ void cutlass_w4a8_moe_mm(; `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0 (4 lines); hunks: -57,6 +57,7 @@ def cutlass_w4a8_moe_mm(; -83,6 +84,8 @@ def cutlass_w4a8_moe_mm(; symbols: cutlass_w4a8_moe_mm, touching `cutlass_w4a8_moe_mm`.
- Code diff details:
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +6/-0 (6 lines); hunks: -170,6 +170,7 @@ def cutlass_w4a8_moe(; -193,6 +194,7 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +38/-6 (44 lines); hunks: -79,10 +79,12 @@ void dispatch_w4a8_moe_mm_sm90(; -114,6 +116,20 @@ void dispatch_w4a8_moe_mm_sm90(
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3 (9 lines); hunks: -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(; -49,7 +50,8 @@ void cutlass_w4a8_moe_mm(
  - `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0 (4 lines); hunks: -57,6 +57,7 @@ def cutlass_w4a8_moe_mm(; -83,6 +84,8 @@ def cutlass_w4a8_moe_mm(; symbols: cutlass_w4a8_moe_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: -459,7 +459,8 @@ void cutlass_w4a8_moe_mm(
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -170,6 +170,7 @@ def cutlass_w4a8_moe(
+        num_local_experts,
@@ -193,6 +194,7 @@ def cutlass_w4a8_moe(
+        num_local_experts,
@@ -360,6 +362,7 @@ def cutlass_w4a8_moe_deepep_normal(
+        num_experts,
@@ -383,6 +386,7 @@ def cutlass_w4a8_moe_deepep_normal(
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu
@@ -79,10 +79,12 @@ void dispatch_w4a8_moe_mm_sm90(
-    int64_t topk) {
-  uint32_t const m = a_tensors.size(0) / topk;
-  uint32_t const n = d_tensors.size(1);
-  uint32_t const k = a_tensors.size(1);
+    int64_t topk,
+    int64_t num_experts) {
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu
@@ -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +6/-0
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +38/-6; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3; `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1; `sgl-kernel/csrc/common_extension.cc` modified +1/-1
  - tests: `python/sglang/test/test_cutlass_w4a8_moe.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_cutlass_w4a8_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15304 - Fix the accuracy issue when running mxfp4 dsv3 model and enable ep

- Link: https://github.com/sgl-project/sglang/pull/15304
- Status/date: merged / 2025-12-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +2/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix the accuracy issue when running mxfp4 dsv3 model and enable ep"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py`; PR body summary: Fix the accuracy problem Add expert mask in moe function call `export SGLANG_AITER_MLA_PERSIST=0 python3 -m sglang.launch_server \ --model-path $MODEL \ --tp-size 8 \ --trust-re....
- Key implementation: `python/sglang/srt/layers/quantization/mxfp4.py` modified +1/-0 (1 lines); hunks: -866,5 +866,6 @@ def apply(; symbols: apply, touching `apply`; `python/sglang/srt/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -245,6 +245,7 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +1/-0 (1 lines); hunks: -866,5 +866,6 @@ def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: -245,6 +245,7 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/mxfp4.py
@@ -866,5 +866,6 @@ def apply(
+            expert_mask=layer.expert_mask_gpu,
diff -- python/sglang/srt/layers/quantization/quark/quark_moe.py
@@ -245,6 +245,7 @@ def apply(
+            expert_mask=layer.expert_mask_gpu,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/mxfp4.py` modified +1/-0; `python/sglang/srt/layers/quantization/quark/quark_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15307 - [Deepseek V3.2] Support Overlap Spec + NSA

- Link: https://github.com/sgl-project/sglang/pull/15307
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `d20699a33c50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +25/-8, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Support Overlap Spec + NSA"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Part of V3.2 Roadmap https://github.com/sgl-project/sglang/issues/15025 Enable overlap spec and EAGLE + NSA backend. In EAGLE V1, we had (with `python3 -m sglang.test.send_one -....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunks: -389,7 +389,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -422,9 +422,20 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph, touching `init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunks: -295,7 +295,7 @@ def _get_topk_paged(; -900,7 +900,7 @@ def forward_cuda(; symbols: _get_topk_paged, forward_cuda, touching `_get_topk_paged, forward_cuda`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunks: -389,7 +389,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -422,9 +422,20 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunks: -295,7 +295,7 @@ def _get_topk_paged(; -900,7 +900,7 @@ def forward_cuda(; symbols: _get_topk_paged, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -389,7 +389,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-        elif forward_batch.forward_mode.is_draft_extend():
+        elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
@@ -422,9 +422,20 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            page_table = torch.repeat_interleave(
-                page_table, repeats=forward_batch.extend_seq_lens, dim=0
-            )
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -295,7 +295,7 @@ def _get_topk_paged(
-            or forward_batch.forward_mode.is_draft_extend()
+            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
@@ -900,7 +900,7 @@ def forward_cuda(
-                or forward_batch.forward_mode.is_draft_extend()
+                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15278 - feat: DeepSeek-V3.2 Streaming tool call output

- Link: https://github.com/sgl-project/sglang/pull/15278
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `41683536d394`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +111/-69, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: DeepSeek-V3.2 Streaming tool call output"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`; PR body summary: FIxes #14711.
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +95/-65 (160 lines); hunks: -1,7 +1,6; -11,6 +10,7; symbols: __init__, has_tool_call, _parse_parameters_from_xml, touching `__init__, has_tool_call, _parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +95/-65 (160 lines); hunks: -1,7 +1,6; -11,6 +10,7; symbols: __init__, has_tool_call, _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -1,7 +1,6 @@
-from typing import List
@@ -11,6 +10,7 @@
+from sglang.srt.function_call.utils import _find_common_prefix
@@ -71,17 +71,26 @@ def __init__(self):
-        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
-        self._last_arguments = ""
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +95/-65
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15380 - [Performance] sgl-kernel, optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe

- Link: https://github.com/sgl-project/sglang/pull/15380
- Status/date: open / 2025-12-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +79/-56, 233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Performance] sgl-kernel, optimize group gemm in DeepSeek-R1-W4AFP8 w4a8 moe"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu`, `sgl-kernel/python/sgl_kernel/cutlass_moe.py`; PR body summary: https://github.com/sgl-project/sglang/pull/15315 While deploying and optimizing the w4a8 moe group gemm for DeepSeek-R1-W4AFP8, I found that when w4a8 is in deepep low_latency m....
- Key implementation: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +66/-51 (117 lines); hunks: -98,17 +98,22 @@ void dispatch_w4a8_moe_mm_sm90(; -124,62 +129,70 @@ void dispatch_w4a8_moe_mm_sm90(; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3 (9 lines); hunks: -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(; -49,7 +50,8 @@ void cutlass_w4a8_moe_mm(; `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0 (4 lines); hunks: -57,6 +57,7 @@ def cutlass_w4a8_moe_mm(; -83,6 +84,8 @@ def cutlass_w4a8_moe_mm(; symbols: cutlass_w4a8_moe_mm, touching `cutlass_w4a8_moe_mm`; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: -383,7 +383,8 @@ void cutlass_w4a8_moe_mm(.
- Code diff details:
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +66/-51 (117 lines); hunks: -98,17 +98,22 @@ void dispatch_w4a8_moe_mm_sm90(; -124,62 +129,70 @@ void dispatch_w4a8_moe_mm_sm90(
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3 (9 lines); hunks: -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(; -49,7 +50,8 @@ void cutlass_w4a8_moe_mm(
  - `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0 (4 lines); hunks: -57,6 +57,7 @@ def cutlass_w4a8_moe_mm(; -83,6 +84,8 @@ def cutlass_w4a8_moe_mm(; symbols: cutlass_w4a8_moe_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: -383,7 +383,8 @@ void cutlass_w4a8_moe_mm(
  - `sgl-kernel/csrc/common_extension.cc` modified +1/-1 (2 lines); hunks: -245,7 +245,7 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu
@@ -98,17 +98,22 @@ void dispatch_w4a8_moe_mm_sm90(
-    int64_t topk) {
-  uint32_t const m = a_tensors.size(0) / topk;
-  uint32_t const n = d_tensors.size(1);
-  uint32_t const k = a_tensors.size(1);
+    int64_t topk,
+    int64_t expected_m_per_group) {
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu
@@ -23,7 +23,8 @@ void cutlass_w4a8_moe_mm_sm90(
-    int64_t topk);
+    int64_t topk,
+    int64_t expected_m_per_group);
@@ -49,7 +50,8 @@ void cutlass_w4a8_moe_mm(
-    int64_t topk) {
+    int64_t topk,
diff -- sgl-kernel/python/sgl_kernel/cutlass_moe.py
@@ -57,6 +57,7 @@ def cutlass_w4a8_moe_mm(
```

- Reviewed files:
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +66/-51; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/scaled_mm_entry.cu` modified +6/-3; `sgl-kernel/python/sgl_kernel/cutlass_moe.py` modified +4/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1; `sgl-kernel/csrc/common_extension.cc` modified +1/-1
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #12921 - [perf]optimize w4afp8 kernel on deepseek-v3-0324

- Link: https://github.com/sgl-project/sglang/pull/12921
- Status/date: merged / 2025-12-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +160/-264, 516 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[perf]optimize w4afp8 kernel on deepseek-v3-0324"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`; PR body summary: we use w4afp8 deepseekv3-0324 online, and we find its performance is not good enough when decode batch size < 32 fine-grained tiling config and based on https://github.com/sgl-p....
- Key implementation: `python/sglang/srt/layers/quantization/w4afp8.py` modified +0/-1 (1 lines); hunks: -300,7 +300,6 @@ def apply(; symbols: apply, touching `apply`; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +64/-236 (300 lines); hunks: -66,6 +66,25 @@ inline void invoke_gemm(; -87,268 +106,77 @@ void dispatch_w4a8_moe_mm_sm90(; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` modified +96/-27 (123 lines); hunks: -2,29 +2,48; -34,20 +53,69 @@ __global__ void compute_problem_sizes_w4a8(.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +0/-1 (1 lines); hunks: -300,7 +300,6 @@ def apply(; symbols: apply
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +64/-236 (300 lines); hunks: -66,6 +66,25 @@ inline void invoke_gemm(; -87,268 +106,77 @@ void dispatch_w4a8_moe_mm_sm90(
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` modified +96/-27 (123 lines); hunks: -2,29 +2,48; -34,20 +53,69 @@ __global__ void compute_problem_sizes_w4a8(
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -300,7 +300,6 @@ def apply(
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu
@@ -66,6 +66,25 @@ inline void invoke_gemm(
+// Helper macro to reduce code duplication
+// Note: Config must be wrapped in parentheses when it contains commas (e.g., template parameters)
+// This uses a helper macro to strip the parentheses from the template parameter
+#define INVOKE_GEMM_WITH_CONFIG_HELPER(...) \
+  invoke_gemm<__VA_ARGS__>(                 \
+      d_tensors,                            \
diff -- sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu
@@ -2,29 +2,48 @@
-#include <iostream>
-constexpr uint64_t THREADS_PER_EXPERT = 512;
+#include <cub/block/block_reduce.cuh>
+#include <cub/block/block_scan.cuh>
+template <int BLOCK_SIZE>
-    int32_t* atomic_buffer,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` modified +0/-1
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +64/-236; `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` modified +96/-27
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/w4afp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15429 - [Deepseek V3.2] Fix Deepseek MTP in V1 mode

- Link: https://github.com/sgl-project/sglang/pull/15429
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `e88e75a9dfdd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Deepseek V3.2] Fix Deepseek MTP in V1 mode"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Restore the earlier behaviour of Deepseek V3 + MTP. (Accidental change) Fix https://github.com/sgl-project/sglang/issues/15428.
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: -435,7 +435,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, touching `init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: -435,7 +435,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -435,7 +435,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-                    page_table, repeats=extend_seq_lens_cpu, dim=0
+                    page_table, repeats=forward_batch.extend_seq_lens, dim=0
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15040 - [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata

- Link: https://github.com/sgl-project/sglang/pull/15040
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `6afc5d497bb3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +91/-5, 166 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: https://github.com/sgl-project/sglang/issues/15025 Changes - Add `paged_mqa_schedule_metadata` to `NSAMetadata` (batch-level caching). - Compute once in `init_forward_metadata()....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +84/-1 (85 lines); hunks: -31,7 +31,7; -113,6 +113,9 @@ class NSAMetadata:; symbols: NSAMetadata, _cat, NSAIndexerMetadata, get_seqlens_int32, touching `NSAMetadata, _cat, NSAIndexerMetadata`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: -300,10 +300,13 @@ def _get_topk_paged(; symbols: _get_topk_paged, touching `_get_topk_paged`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +84/-1 (85 lines); hunks: -31,7 +31,7; -113,6 +113,9 @@ class NSAMetadata:; symbols: NSAMetadata, _cat, NSAIndexerMetadata, get_seqlens_int32
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunks: -300,10 +300,13 @@ def _get_topk_paged(; symbols: _get_topk_paged
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -31,7 +31,7 @@
-from sglang.srt.utils import is_hip
+from sglang.srt.utils import is_cuda, is_hip
@@ -113,6 +113,9 @@ class NSAMetadata:
+    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
+    # Precomputed once per forward batch and reused across layers.
+    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -300,10 +300,13 @@ def _get_topk_paged(
-        # NOTE(dark): 132 is SM count on H200/B200, not magic number
-        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
-            seqlens_32, blocksize, self.sm_count
-        )
+        # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
+        # otherwise fall back to computing it here.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +84/-1; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15531 - Support piecewise cuda graph for dsv3 fp4

- Link: https://github.com/sgl-project/sglang/pull/15531
- Status/date: merged / 2025-12-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `8fe3e3746832`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +148/-16, 261 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support piecewise cuda graph for dsv3 fp4"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: https://github.com/sgl-project/sglang/issues/11490.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -20,6 +20,7; -400,6 +401,9 @@ def handle_attention_fa4(attn, forward_batch):; symbols: handle_attention_fa4, handle_attention_trtllm_mla, forward, touching `handle_attention_fa4, handle_attention_trtllm_mla, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: -20,6 +20,7; -400,6 +401,9 @@ def handle_attention_fa4(attn, forward_batch):; symbols: handle_attention_fa4, handle_attention_trtllm_mla, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -20,6 +20,7 @@
+from contextlib import nullcontext
@@ -400,6 +401,9 @@ def handle_attention_fa4(attn, forward_batch):
+    if is_in_piecewise_cuda_graph():
+        return AttnForwardMethod.MLA
@@ -3188,7 +3192,13 @@ def forward(
-            with get_global_expert_distribution_recorder().with_current_layer(i):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +11/-1
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14750 - [Tool Call][DSV32] Streamline function call parameters

- Link: https://github.com/sgl-project/sglang/pull/14750
- Status/date: merged / 2025-12-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `01bd0d3e8b8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-29, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Call][DSV32] Streamline function call parameters"; model line: DeepSeek V3/R1; category: docs/tests/CI; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`; PR body summary: As titled and Issue #14711. **IMPORTANT: This is an initial version aimed at streamlining function call parameters for `DeepSeek-V3.2`. I have tested it, and it passes all tests....
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +23/-15 (38 lines); hunks: -2,6 +2,8; -10,7 +12,7; symbols: __init__, has_tool_call, _parse_parameters_from_xml, parse_streaming_increment, touching `__init__, has_tool_call, _parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +23/-15 (38 lines); hunks: -2,6 +2,8; -10,7 +12,7; symbols: __init__, has_tool_call, _parse_parameters_from_xml, parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -2,6 +2,8 @@
+from partial_json_parser.core.options import Allow
@@ -10,7 +12,7 @@
-from sglang.srt.function_call.utils import _find_common_prefix
+from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads
@@ -82,11 +84,12 @@ def __init__(self):
+        self.prefix_parameter_end_call = ["</", "｜DSML｜", "parameter"]
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +23/-15
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14280 - feat PD: add eagle3 support for DeepSeek V3 in EP mode

- Link: https://github.com/sgl-project/sglang/pull/14280
- Status/date: merged / 2025-12-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `2ec6fa3c54a8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-1, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat PD: add eagle3 support for DeepSeek V3 in EP mode"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: When try to enable EP MOE for DeepSeek V3 with EAGLE3, SGL_ENABLE_JIT_DEEPGEMM_BMM=1 SGL_ENABLE_JIT_DEEPGEMM=1 \ TORCHINDUCTOR_CACHE_DIR=/home/admin/inductor_root_cache /opt/con....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +12/-1 (13 lines); hunks: -46,6 +46,7; -3117,6 +3118,10 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-1 (13 lines); hunks: -46,6 +46,7; -3117,6 +3118,10 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -46,6 +46,7 @@
+    tensor_model_parallel_all_gather,
@@ -3117,6 +3118,10 @@ def __init__(
+        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
+            self.enable_a2a_moe = True
+        else:
+            self.enable_a2a_moe = False
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +12/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13394 - Fix DeepSeekV31's structural tag trigger

- Link: https://github.com/sgl-project/sglang/pull/13394
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv31_detector.py`; associated commits `2667c857a78f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 7 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix DeepSeekV31's structural tag trigger"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv31_detector.py`; PR body summary: As titled. To prevent LLMs generating tool calls with wrong name Reference (can be seen in other models in the SGLang codebase as well):.
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

### PR #13959 - [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache

- Link: https://github.com/sgl-project/sglang/pull/13959
- Status/date: merged / 2026-01-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/communicator_nsa_cp.py` and 6 files; associated commits `0d244116d28a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +603/-264, 1414 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: The original default token splitting scheme of cp does not support prefill multi-batch. A new token splitting method is introduced to enable multi-batch support, fused MoE compa....
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +209/-5 (214 lines); hunks: -1,15 +1,26; -41,6 +52,75 @@ def is_nsa_enable_prefill_cp():; symbols: is_nsa_enable_prefill_cp, is_nsa_prefill_cp_in_seq_split, is_nsa_prefill_cp_round_robin_split, can_nsa_prefill_cp_round_robin_split, touching `is_nsa_enable_prefill_cp, is_nsa_prefill_cp_in_seq_split, is_nsa_prefill_cp_round_robin_split`; `python/sglang/srt/layers/communicator_nsa_cp.py` modified +60/-133 (193 lines); hunks: -18,7 +18,10; -28,6 +31,11; symbols: __init__, _post_init_communicate, get_fn, _scattered_to_tp_attn_full, touching `__init__, _post_init_communicate, get_fn`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +149/-20 (169 lines); hunks: -2,7 +2,7; -25,8 +25,12; symbols: NSAMetadata, TopkTransformMethod, get_seqlens_expanded, get_cu_seqlens_k, touching `NSAMetadata, TopkTransformMethod, get_seqlens_expanded`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-68 (113 lines); hunks: -28,6 +28,7; -63,6 +64,21 @@ def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx, touching `get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +209/-5 (214 lines); hunks: -1,15 +1,26; -41,6 +52,75 @@ def is_nsa_enable_prefill_cp():; symbols: is_nsa_enable_prefill_cp, is_nsa_prefill_cp_in_seq_split, is_nsa_prefill_cp_round_robin_split, can_nsa_prefill_cp_round_robin_split
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +60/-133 (193 lines); hunks: -18,7 +18,10; -28,6 +31,11; symbols: __init__, _post_init_communicate, get_fn, _scattered_to_tp_attn_full
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +149/-20 (169 lines); hunks: -2,7 +2,7; -25,8 +25,12; symbols: NSAMetadata, TopkTransformMethod, get_seqlens_expanded, get_cu_seqlens_k
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-68 (113 lines); hunks: -28,6 +28,7; -63,6 +64,21 @@ def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-16 (26 lines); hunks: -64,8 +64,8; -578,9 +578,7 @@ def forward(; symbols: forward, forward_absorb_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -1,15 +1,26 @@
-from typing import List
+from typing import TYPE_CHECKING, List, Tuple, Union
-from sglang.srt.layers.dp_attention import get_attention_tp_group
+import triton
+import triton.language as tl
+from sglang.srt.layers.dp_attention import (
diff -- python/sglang/srt/layers/communicator_nsa_cp.py
@@ -18,7 +18,10 @@
-from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
+from sglang.srt.layers.attention.nsa.utils import (
+    is_nsa_enable_prefill_cp,
+    nsa_use_prefill_cp,
+)
@@ -28,6 +31,11 @@
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -2,7 +2,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +209/-5; `python/sglang/srt/layers/communicator_nsa_cp.py` modified +60/-133; `python/sglang/srt/layers/attention/nsa_backend.py` modified +149/-20; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-68; `python/sglang/srt/models/deepseek_v2.py` modified +10/-16; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +5/-5
- Risk and verification: The diff ships test coverage in `test/manual/test_deepseek_v32_cp_single_node.py`, `test/srt/test_prefill_adder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15938 - Clean Some Environment Variables for DeepSeek V32

- Link: https://github.com/sgl-project/sglang/pull/15938
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `7d757d6f17ef`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +39/-108, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Clean Some Environment Variables for DeepSeek V32"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; PR body summary: - Move `SGLANG_NSA_FUSE_TOPK` and `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` to environ.py - Deprecate `SGLANG_NSA_DUAL_STREAM`, `SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8....
- Key implementation: `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +6/-42 (48 lines); hunks: -2,15 +2,9; -56,43 +50,13 @@ def quantize_k_cache_separate(; symbols: quantize_k_cache, quantize_k_cache_separate, _quantize_k_cache_slow, _quantize_k_cache_ref, touching `quantize_k_cache, quantize_k_cache_separate, _quantize_k_cache_slow`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-19 (27 lines); hunks: -22,9 +22,6; -230,7 +227,7 @@ def topk_transform(; symbols: topk_transform, forward_extend, forward_decode, touching `topk_transform, forward_extend, forward_decode`; `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-24 (24 lines); hunks: -15,35 +15,11; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens, touching `print_nsa_bool_env_vars, compute_nsa_seqlens`; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-7 (9 lines); hunks: -2,17 +2,12; symbols: dequantize_k_cache, _dequantize_k_cache_slow, _dequantize_k_cache_ref, touching `dequantize_k_cache, _dequantize_k_cache_slow, _dequantize_k_cache_ref`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +6/-42 (48 lines); hunks: -2,15 +2,9; -56,43 +50,13 @@ def quantize_k_cache_separate(; symbols: quantize_k_cache, quantize_k_cache_separate, _quantize_k_cache_slow, _quantize_k_cache_ref
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-19 (27 lines); hunks: -22,9 +22,6; -230,7 +227,7 @@ def topk_transform(; symbols: topk_transform, forward_extend, forward_decode
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-24 (24 lines); hunks: -15,35 +15,11; symbols: print_nsa_bool_env_vars, compute_nsa_seqlens
  - `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-7 (9 lines); hunks: -2,17 +2,12; symbols: dequantize_k_cache, _dequantize_k_cache_slow, _dequantize_k_cache_ref
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-3 (4 lines); hunks: -25,7 +25,6; -802,8 +801,7 @@ def forward_cuda(; symbols: forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/quant_k_cache.py
@@ -2,15 +2,9 @@
-from sglang.srt.layers.attention.nsa.utils import NSA_QUANT_K_CACHE_FAST
-    # TODO upstream can skip concat([k_nope, k_pe]) since we split them here
-    if NSA_QUANT_K_CACHE_FAST:
-        return _quantize_k_cache_fast_wrapped(cache_k)
-    else:
-        return _quantize_k_cache_slow(cache_k)
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -22,9 +22,6 @@
-    NSA_ENABLE_MTP_PRECOMPUTE_METADATA,
-    NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,
-    NSA_FUSE_TOPK,
@@ -230,7 +227,7 @@ def topk_transform(
-        if not NSA_FUSE_TOPK:
+        if not envs.SGLANG_NSA_FUSE_TOPK.get():
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -15,35 +15,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +6/-42; `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-19; `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-24; `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-7; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16637 - [DSv32] Overlap indexer weights_proj during dual_stream decode

- Link: https://github.com/sgl-project/sglang/pull/16637
- Status/date: merged / 2026-01-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `20abaee26cd8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +64/-22, 134 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32] Overlap indexer weights_proj during dual_stream decode"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: The weights_proj in the DSA indexer uses float type, which is slow on modern GPUs. Profiler traces show this accounts for ~20% of decode layer runtime. The traces also reveal th....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +34/-11 (45 lines); hunks: -1666,6 +1666,7 @@ def forward_absorb_prepare(; -1722,8 +1723,39 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, touching `forward_absorb_prepare`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-11 (41 lines); hunks: -205,6 +205,12 @@ def _with_real_sm_count(self):; -856,21 +862,36 @@ def forward_cuda(; symbols: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, forward_cuda, touching `_with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-11 (45 lines); hunks: -1666,6 +1666,7 @@ def forward_absorb_prepare(; -1722,8 +1723,39 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-11 (41 lines); hunks: -205,6 +205,12 @@ def _with_real_sm_count(self):; -856,21 +862,36 @@ def forward_cuda(; symbols: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1666,6 +1666,7 @@ def forward_absorb_prepare(
+        topk_indices = None
@@ -1722,8 +1723,39 @@ def forward_absorb_prepare(
-            k_nope = k_nope.unsqueeze(1)
-            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
+            # overlap q_b_proj and indexer during decode
+            if (
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -205,6 +205,12 @@ def _with_real_sm_count(self):
+    @torch.compile(dynamic=True)
+    def _project_and_scale_head_gates(self, x: torch.Tensor):
+        weights, _ = self.weights_proj(x.float())
+        weights = weights * self.n_heads**-0.5
+        return weights
@@ -856,21 +862,36 @@ def forward_cuda(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +34/-11; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- Link: https://github.com/sgl-project/sglang/pull/17178
- Status/date: merged / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/test/run_eval.py`; PR body summary: deepseek-r1 is a pure reasoning model with no way to enable/disable reasoning, unless you modify the chat template directly. Starting from v3.1, deepseek introduced hybrid reaso....
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
- Motivation: Title: "[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; PR body summary: 1. Performance tuning based on the code after fused moe swapab #16723. The optimal configuration of fused MoE changes when swapab is taken into consideration. 2. Optimize the tu....
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

### PR #16649 - [Refactor] Split out deepseek v2 weight loader function into mixin

- Link: https://github.com/sgl-project/sglang/pull/16649
- Status/date: merged / 2026-01-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +721/-600, 1455 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Split out deepseek v2 weight loader function into mixin"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`; PR body summary: DeepseekV2 code has been developed fast and a lot of historical code and be more orgranized, including the weight loading part. Issue related: https://github.com/sgl-project/sgl....
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` added +657/-0 (657 lines); hunks: -0,0 +1,657; symbols: DeepseekV2WeightLoaderMixin, do_load_weights, post_load_weights, _maybe_quant_weights_to_fp8_ue8m0, touching `DeepseekV2WeightLoaderMixin, do_load_weights, post_load_weights`; `python/sglang/srt/models/deepseek_v2.py` modified +9/-594 (603 lines); hunks: -17,15 +17,13; -111,44 +109,29; symbols: enable_nextn_moe_bf16_cast_to_fp8, forward, DeepseekV2ForCausalLM, __init__, touching `enable_nextn_moe_bf16_cast_to_fp8, forward, DeepseekV2ForCausalLM`; `python/sglang/srt/models/deepseek_common/utils.py` modified +53/-1 (54 lines); hunks: -1,3 +1,19; -19,5 +35,41; symbols: awq_dequantize_func, enable_nextn_moe_bf16_cast_to_fp8, touching `awq_dequantize_func, enable_nextn_moe_bf16_cast_to_fp8`; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-5 (7 lines); hunks: -46,11 +46,8.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` added +657/-0 (657 lines); hunks: -0,0 +1,657; symbols: DeepseekV2WeightLoaderMixin, do_load_weights, post_load_weights, _maybe_quant_weights_to_fp8_ue8m0
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-594 (603 lines); hunks: -17,15 +17,13; -111,44 +109,29; symbols: enable_nextn_moe_bf16_cast_to_fp8, forward, DeepseekV2ForCausalLM, __init__
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +53/-1 (54 lines); hunks: -1,3 +1,19; -19,5 +35,41; symbols: awq_dequantize_func, enable_nextn_moe_bf16_cast_to_fp8
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-5 (7 lines); hunks: -46,11 +46,8
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -0,0 +1,657 @@
+# Copyright 2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -17,15 +17,13 @@
-import concurrent.futures
-import tqdm
@@ -111,44 +109,29 @@
-from sglang.srt.layers.quantization.fp8_utils import (
-    block_quant_dequant,
-    block_quant_to_tensor_quant,
diff -- python/sglang/srt/models/deepseek_common/utils.py
@@ -1,3 +1,19 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` added +657/-0; `python/sglang/srt/models/deepseek_v2.py` modified +9/-594; `python/sglang/srt/models/deepseek_common/utils.py` modified +53/-1; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`, `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15347 - Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

- Link: https://github.com/sgl-project/sglang/pull/15347
- Status/date: merged / 2026-01-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +165/-12, 215 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`; PR body summary: flashinfer has an optimized routing kernel for DeepSeek V3: https://github.com/flashinfer-ai/flashinfer/pull/2099 The API was renamed to `fused_topk_deepseek` here: https://gith....
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, touching `biased_grouped_topk_gpu`; `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek, touching `test_fused_topk_deepseek`; `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed, touching `test_a_gsm8k, test_bs_1_speed`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: -75,6 +75,11; -732,12 +737,68 @@ def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
  - `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: test_fused_topk_deepseek
  - `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: -82,10 +82,7 @@ def test_a_gsm8k(; -99,10 +96,7 @@ def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -75,6 +75,11 @@
+    try:
+        from flashinfer.fused_moe import fused_topk_deepseek
+    except ImportError:
+        fused_topk_deepseek = None
@@ -732,12 +737,68 @@ def biased_grouped_topk_gpu(
-    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
diff -- test/registered/kernels/test_fused_topk_deepseek.py
@@ -0,0 +1,97 @@
+import pytest
+import torch
+from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu, biased_grouped_topk_impl
+from sglang.test.ci.ci_register import register_cuda_ci
+register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)
+@pytest.mark.parametrize(
diff -- test/srt/test_deepseek_v3_mtp.py
@@ -82,10 +82,7 @@ def test_a_gsm8k(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +66/-4
  - tests: `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0; `test/srt/test_deepseek_v3_mtp.py` modified +2/-8
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_fused_topk_deepseek.py`, `test/srt/test_deepseek_v3_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16961 - [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation

- Link: https://github.com/sgl-project/sglang/pull/16961
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `d2105d4abda6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-12, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: 1、`draft_extend` and `target_verify` use `nsa_decode_backend` as backend implementation instead of `nsa_prefill_backend` In the MTP scenario, the NSA attention implementation se....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +14/-5 (19 lines); hunks: -1263,7 +1263,16 @@ def forward_extend(; -1273,7 +1282,7 @@ def forward_extend(; symbols: forward_extend, touching `forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +14/-5 (19 lines); hunks: -1263,7 +1263,16 @@ def forward_extend(; -1273,7 +1282,7 @@ def forward_extend(; symbols: forward_extend
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1263,7 +1263,16 @@ def forward_extend(
-        if self.nsa_prefill_impl == "tilelang":
+        nsa_impl = (
+            self.nsa_decode_impl
+            if (
+                forward_batch.forward_mode.is_target_verify()
+                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +14/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17179 - [AMD] Add DeepSeek-V3.2 and VLMs model in nightly tests

- Link: https://github.com/sgl-project/sglang/pull/17179
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`; associated commits `a3addd6203ca`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 18 files, +1140/-99, 1539 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add DeepSeek-V3.2 and VLMs model in nightly tests"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py`; PR body summary: Add 5 unique test in AMD coverage: https://github.com/sgl-project/sglang/actions/runs/21153414554/attempts/1#summary-60833755716 - Add accuracy and perf tests for the DeepSeek-V....
- Key implementation: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +5/-30 (35 lines); hunks: -1,7 +1,7; -32,9 +32,9; symbols: get_mxfp4_models, touching `get_mxfp4_models`; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +13/-3 (16 lines); hunks: -30,7 +30,10; -43,7 +46,14 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, setUpClass, touching `generate_simple_markdown_report, setUpClass`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +5/-30 (35 lines); hunks: -1,7 +1,7; -32,9 +32,9; symbols: get_mxfp4_models
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +13/-3 (16 lines); hunks: -30,7 +30,10; -43,7 +46,14 @@ def generate_simple_markdown_report(results: List[BenchmarkRe...; symbols: generate_simple_markdown_report, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py
@@ -1,7 +1,7 @@
-Tests DeepSeek-R1-MXFP4 quantized model with multiple configurations
-(basic, MTP, DP, TC) using few-shot completion benchmark on MI35x.
+Tests DeepSeek-R1-MXFP4 quantized model with basic configuration
+using few-shot completion benchmark on MI35x.
@@ -32,9 +32,9 @@
-# Register for AMD CI - MI35x DeepSeek-R1-MXFP4 accuracy tests (~120 min)
diff -- test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py
@@ -30,7 +30,10 @@
-    """Generate a simplified markdown report without traces and cost columns."""
+    """Generate a simplified markdown report without traces and cost columns.
+    Skips the first result if it's a warmup run (duplicate batch_size).
+    """
@@ -43,7 +46,14 @@ def generate_simple_markdown_report(results: List[BenchmarkResult]) -> str:
-    for result in results:
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +5/-30; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py` modified +13/-3
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/accuracy/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/test_vlms_mmmu_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17205 - [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance

- Link: https://github.com/sgl-project/sglang/pull/17205
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `612026ad2c82`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: From these pr[https://github.com/sgl-project/sglang/pull/16637 and https://github.com/sgl-project/sglang/issues/16861 and https://github.com/sgl-project/sglang/pull/13459/files]....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-4 (9 lines); hunks: -182,12 +182,11 @@ def __init__(; -221,13 +220,15 @@ def _with_real_sm_count(self):; symbols: __init__, _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, touching `__init__, _with_real_sm_count, _project_and_scale_head_gates`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-4 (9 lines); hunks: -182,12 +182,11 @@ def __init__(; -221,13 +220,15 @@ def _with_real_sm_count(self):; symbols: __init__, _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -182,12 +182,11 @@ def __init__(
-        # NOTE: weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenience
-            params_dtype=torch.float32,
+            params_dtype=torch.bfloat16,
@@ -221,13 +220,15 @@ def _with_real_sm_count(self):
-        weights, _ = self.weights_proj(x.float())
+        weights, _ = self.weights_proj(x)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17007 - [NPU]bugfix: fix for dsv3.2 and dsvl2

- Link: https://github.com/sgl-project/sglang/pull/17007
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `c0b5a180febe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +129/-46, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]bugfix: fix for dsv3.2 and dsvl2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; PR body summary: 1、There are some bugs for DS-Vl2 on rotary_embedding 2、DSV32 is not compatible with the scenario where m.alt_stream is not None. 1、Fix the ds-v12 rotary_embedding bug and add a....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -1220,6 +1220,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +80/-46 (126 lines); hunks: -64,36 +64,44 @@ def forward_mha_prepare_npu(; -288,10 +296,30 @@ def forward_dsa_prepare_npu(; symbols: forward_mha_prepare_npu, forward_dsa_prepare_npu, touching `forward_mha_prepare_npu, forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -1220,6 +1220,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +80/-46 (126 lines); hunks: -64,36 +64,44 @@ def forward_mha_prepare_npu(; -288,10 +296,30 @@ def forward_dsa_prepare_npu(; symbols: forward_mha_prepare_npu, forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1220,6 +1220,7 @@ def __init__(
+        self.use_deepseek_yarn_rope = rope_scaling is not None
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -64,36 +64,44 @@ def forward_mha_prepare_npu(
-    B, S = q.shape[0], 1
-    cos, sin = m.rotary_emb.get_cos_sin_cache(
-        positions, hidden_states.dtype, offsets=None
-    )
-    q_pe = torch_npu.npu_interleave_rope(
-        q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +80/-46
- Risk and verification: The diff ships test coverage in `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16758 - [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache

- Link: https://github.com/sgl-project/sglang/pull/16758
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `2fb328109fb9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +118/-31, 228 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Enables NSA backend with trtllm kernels for sparse attention. This can be more efficient than FlashMLA when the head size isn't a multiple of 64 and hence requires padding. This....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +54/-3 (57 lines); hunks: -254,7 +254,9 @@ def topk_transform(; -287,6 +289,9 @@ def __init__(; symbols: topk_transform, NativeSparseAttnBackend, __init__, forward_decode, touching `topk_transform, NativeSparseAttnBackend, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +54/-3 (57 lines); hunks: -254,7 +254,9 @@ def topk_transform(; -287,6 +289,9 @@ def __init__(; symbols: topk_transform, NativeSparseAttnBackend, __init__, forward_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -254,7 +254,9 @@ def topk_transform(
-_NSA_IMPL_T: TypeAlias = Literal["flashmla_sparse", "flashmla_kv", "fa3", "tilelang"]
+_NSA_IMPL_T: TypeAlias = Literal[
+    "flashmla_sparse", "flashmla_kv", "fa3", "tilelang", "trtllm"
+]
@@ -287,6 +289,9 @@ def __init__(
+        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +54/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17662 - [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend

- Link: https://github.com/sgl-project/sglang/pull/17662
- Status/date: merged / 2026-01-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `1674b9ef4494`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-1, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: Quick follow‑up to #16758: speculative decoding still fails with `--nsa-decode-backend trtllm`. - Add TRT‑LLM handling in the extend path for speculative modes and guard it with....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +18/-1 (19 lines); hunks: -1339,6 +1339,21 @@ def forward_extend(; -1468,6 +1483,7 @@ def forward_decode(; symbols: forward_extend, forward_decode, _forward_trtllm, touching `forward_extend, forward_decode, _forward_trtllm`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +18/-1 (19 lines); hunks: -1339,6 +1339,21 @@ def forward_extend(; -1468,6 +1483,7 @@ def forward_decode(; symbols: forward_extend, forward_decode, _forward_trtllm
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1339,6 +1339,21 @@ def forward_extend(
+        elif nsa_impl == "trtllm":
+            assert forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend(
+                include_v2=True
+            ), "TRT-LLM NSA only supports target_verify/draft_extend; normal extend untested."
+            if q_rope is not None:
+                q_all = _concat_mla_absorb_q_general(q_nope, q_rope)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +18/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15381 - [NPU]DeepSeek-V3.2 support npu mlaprolog

- Link: https://github.com/sgl-project/sglang/pull/15381
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `b56366f8275a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +195/-61, 364 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]DeepSeek-V3.2 support npu mlaprolog"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; PR body summary: we enabled the DSV3.2 to support npu mla_prolog_v3 to improve the performance - Added support for the mla_prolog_v3 custom operator, which includes operations such as Q/KV LoRA,....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-0 (5 lines); hunks: -1113,6 +1113,7 @@ def forward_npu(; -1136,6 +1137,9 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +116/-55 (171 lines); hunks: -1,3 +1,4; -281,61 +282,25 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_core_npu, npu_mla_preprocess, touching `forward_dsa_prepare_npu, forward_dsa_core_npu, npu_mla_preprocess`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-0 (5 lines); hunks: -1113,6 +1113,7 @@ def forward_npu(; -1136,6 +1137,9 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +116/-55 (171 lines); hunks: -1,3 +1,4; -281,61 +282,25 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, forward_dsa_core_npu, npu_mla_preprocess
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1113,6 +1113,7 @@ def forward_npu(
+        dynamic_scale: torch.Tensor = None,
@@ -1136,6 +1137,9 @@ def forward_npu(
+                q_lora = (
+                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
+                )
@@ -1154,6 +1158,7 @@ def forward_npu(
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -1,3 +1,4 @@
+import re
@@ -281,61 +282,25 @@ def forward_dsa_prepare_npu(
+    dynamic_scale = None
-        if not hasattr(m, "mla_preprocess"):
-            m.mla_preprocess = NPUFusedMLAPreprocess(
-                m.fused_qkv_a_proj_with_mqa,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-0; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +116/-55
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17688 - [DSv32] Overlap indexer qk projection and activation quant

- Link: https://github.com/sgl-project/sglang/pull/17688
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `a8dda2aa5727`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32] Overlap indexer qk projection and activation quant"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: @humansand After https://github.com/sgl-project/sglang/pull/17205, the indexer weight projection is fully hidden and no longer exposes latency. Per layer, there is a (52.5-39.5)....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-4 (8 lines); hunks: -957,11 +957,11 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-4 (8 lines); hunks: -957,11 +957,11 @@ def forward_cuda(; symbols: forward_cuda
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -957,11 +957,11 @@ def forward_cuda(
+            query, key = self._get_q_k_bf16(
+                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
+            )
+            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
-                query, key = self._get_q_k_bf16(
-                    q_lora, x, positions, False, forward_batch=forward_batch
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17523
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`; associated commits `f8636fbb253a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +1540/-43, 1823 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI"; model line: DeepSeek V3/R1; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`; PR body summary: 1. Add **Kimi-K2**, **DeepSeek-V3.2** accuracy and performance tests for MI325 (MI30x) platform, update Mi35x tests, consolidate test jobs, and fix various CI failures. 2. Total....
- Key implementation: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -214,6 +214,9 @@ def test_deepseek_r1_accuracy(self):; symbols: test_deepseek_r1_accuracy, touching `test_deepseek_r1_accuracy`; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -239,6 +239,9 @@ def test_deepseek_r1_mxfp4_accuracy(self):; symbols: test_deepseek_r1_mxfp4_accuracy, touching `test_deepseek_r1_mxfp4_accuracy`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -214,6 +214,9 @@ def test_deepseek_r1_accuracy(self):; symbols: test_deepseek_r1_accuracy
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -239,6 +239,9 @@ def test_deepseek_r1_mxfp4_accuracy(self):; symbols: test_deepseek_r1_mxfp4_accuracy
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py
@@ -214,6 +214,9 @@ def test_deepseek_r1_accuracy(self):
+                        print(
+                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
+                        )
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py
@@ -239,6 +239,9 @@ def test_deepseek_r1_mxfp4_accuracy(self):
+                        print(
+                            f"  accuracy={acc:.3f} threshold={config.accuracy_threshold} {status}"
+                        )
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py` modified +3/-0; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17744 - Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization

- Link: https://github.com/sgl-project/sglang/pull/17744
- Status/date: merged / 2026-01-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-12, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; PR body summary: - Fix OOM when loading DeepSeek models with streaming/remote weight loading - Defer `dict(weights)` call until quantization is actually needed - Return original weights iterator....
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +16/-12 (28 lines); hunks: -623,9 +623,9 @@ def _maybe_quant_weights_to_fp8_ue8m0(; -655,16 +655,20 @@ def _maybe_quant_weights_to_fp8_ue8m0(; symbols: _maybe_quant_weights_to_fp8_ue8m0, touching `_maybe_quant_weights_to_fp8_ue8m0`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +16/-12 (28 lines); hunks: -623,9 +623,9 @@ def _maybe_quant_weights_to_fp8_ue8m0(; -655,16 +655,20 @@ def _maybe_quant_weights_to_fp8_ue8m0(; symbols: _maybe_quant_weights_to_fp8_ue8m0
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -623,9 +623,9 @@ def _maybe_quant_weights_to_fp8_ue8m0(
-            List of (name, tensor) pairs with quantized weights
+            Original weights iterator if no quantization needed,
+            otherwise list of (name, tensor) pairs with quantized weights
-        weights_dict = dict(weights)
@@ -655,16 +655,20 @@ def _maybe_quant_weights_to_fp8_ue8m0(
-        if partial_names:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +16/-12
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17076 - [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph

- Link: https://github.com/sgl-project/sglang/pull/17076
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `677f3c49dabf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +58/-7, 135 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: This PR #15227 only resolves the issue occurring during `forward_idle`. If running in Eagle mode (MTP Spec2, batch size exceeding the CUDA maximum batch size, or with `disable-c....
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +28/-4 (32 lines); hunks: -9,12 +9,15; -81,12 +84,33 @@ def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor...; symbols: nsa_cp_round_robin_split_data, pad_nsa_cache_seqlens, cal_padded_tokens, touching `nsa_cp_round_robin_split_data, pad_nsa_cache_seqlens, cal_padded_tokens`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-2 (22 lines); hunks: -86,6 +86,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; -390,6 +395,9 @@ def _get_topk_paged(; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, _get_topk_paged, touching `get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunks: -189,6 +189,9 @@ def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, t...; symbols: get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, touching `get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_nsa_extend_len_cpu`; `test/registered/kernels/test_nsa_indexer.py` modified +7/-1 (8 lines); hunks: -1,5 +1,5; -132,6 +132,12 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, touching `get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +28/-4 (32 lines); hunks: -9,12 +9,15; -81,12 +84,33 @@ def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor...; symbols: nsa_cp_round_robin_split_data, pad_nsa_cache_seqlens, cal_padded_tokens
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-2 (22 lines); hunks: -86,6 +86,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; -390,6 +395,9 @@ def _get_topk_paged(; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, _get_topk_paged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunks: -189,6 +189,9 @@ def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, t...; symbols: get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
  - `test/registered/kernels/test_nsa_indexer.py` modified +7/-1 (8 lines); hunks: -1,5 +1,5; -132,6 +132,12 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -9,12 +9,15 @@
+    DpPaddingMode,
+    get_attention_dp_rank,
+from sglang.srt.utils.common import ceil_align, ceil_div
@@ -81,12 +84,33 @@ def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
-def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):
+def cal_padded_tokens(forward_batch: "ForwardBatch"):
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -86,6 +86,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:
+    def get_nsa_extend_len_cpu(self) -> List[int]:
+        """
+        Return: extend seq lens for each batch.
+        """
@@ -390,6 +395,9 @@ def _get_topk_paged(
+        # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -189,6 +189,9 @@ def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +28/-4; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-2; `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +7/-1
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17964 - [NPU] support dsv32 radixcache on ascend

- Link: https://github.com/sgl-project/sglang/pull/17964
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `b0a6d5244cef`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +16/-2, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] support dsv32 radixcache on ascend"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Ascend currently does not support the radixcache feature on the DeepSeek-3.2 model 1. We have modified the inputs of `npu_sparse_flash_attention` and `npu_lightning_indexer` to....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -1226,7 +1226,7 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -1226,7 +1226,7 @@ def forward_npu(; symbols: forward_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1226,7 +1226,7 @@ def forward_npu(
-                actual_seq_lengths_q = forward_batch.seq_lens.cumsum(dim=0)
+                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18297 - Deepseekv32 compatibility with transformers v5

- Link: https://github.com/sgl-project/sglang/pull/18297
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `e8a2c133807c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +33/-19, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Deepseekv32 compatibility with transformers v5"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: 1. Enhance Indexer and DeepseekV2AttentionMLA to support dynamic is_neox_style parameter. 2. Refactor rope parameter handling in DeepseekV2DecoderLayer for improved configuratio....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -1150,6 +1150,7 @@ def __init__(; -1162,6 +1163,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-0 (4 lines); hunks: -300,6 +300,8 @@ def __init__(; -1837,6 +1839,8 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[For...; symbols: __init__, set_nsa_prefill_impl, touching `__init__, set_nsa_prefill_impl`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-1 (3 lines); hunks: -145,6 +145,7 @@ def __init__(; -202,7 +203,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -1150,6 +1150,7 @@ def __init__(; -1162,6 +1163,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-0 (4 lines); hunks: -300,6 +300,8 @@ def __init__(; -1837,6 +1839,8 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[For...; symbols: __init__, set_nsa_prefill_impl
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-1 (3 lines); hunks: -145,6 +145,7 @@ def __init__(; -202,7 +203,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1150,6 +1150,7 @@ def __init__(
+            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
@@ -1162,6 +1163,7 @@ def __init__(
+                is_neox_style=is_neox_style,
@@ -1191,13 +1193,14 @@ def __init__(
+            is_neox_style = not getattr(config, "rope_interleave", True)
-                is_neox_style=False,
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -300,6 +300,8 @@ def __init__(
+        # Force NSA prefill to use MLA (i.e. disable MHA_ONE_SHOT), controlled by env var.
+        self._force_attn_forward_mla: bool = envs.SGLANG_NSA_FORCE_MLA.get()
@@ -1837,6 +1839,8 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
+        if self._force_attn_forward_mla:
+            self.use_mha = False
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -145,6 +145,7 @@ def __init__(
+        is_neox_style: bool = True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-4; `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18488 - Tilelang sparse decode fwd for dsv32 mi355

- Link: https://github.com/sgl-project/sglang/pull/18488
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; associated commits `4262f5259b94`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +257/-0, 271 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Tilelang sparse decode fwd for dsv32 mi355"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; PR body summary: cc @HaiShaw, @kkHuang-amd, @hubertlu-tw The current dsv32 attention implementation uses a single TileLang kernel for both prefill and decode. This kernel is optimized for prefil....
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +257/-0 (257 lines); hunks: -773,6 +773,242 @@ def main(; -788,6 +1024,27 @@ def tilelang_sparse_fwd(; symbols: main, sparse_mla_fwd_decode_partial, sparse_mla_fwd_decode_combine, touching `main, sparse_mla_fwd_decode_partial, sparse_mla_fwd_decode_combine`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +257/-0 (257 lines); hunks: -773,6 +773,242 @@ def main(; -788,6 +1024,27 @@ def tilelang_sparse_fwd(; symbols: main, sparse_mla_fwd_decode_partial, sparse_mla_fwd_decode_combine
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -773,6 +773,242 @@ def main(
+@tilelang.jit(
+    out_idx=[-2, -1],
+    pass_configs={
+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
+    },
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +257/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18553 - Fix Bug on dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/18553
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `2cc235e7952e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-8, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Bug on dsv3.2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: Fix two bugs on dsv32 when infering on npu 1. fix a bug: forward_npu method of nas_Indexer has a dual stream feature that can not be closed. May cause problem in some cases. 2.....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-6 (18 lines); hunks: -7,6 +7,7; -1190,13 +1191,17 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-6 (18 lines); hunks: -7,6 +7,7; -1190,13 +1191,17 @@ def forward_npu(; symbols: forward_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -7,6 +7,7 @@
+from sglang.srt.environ import envs
@@ -1190,13 +1191,17 @@ def forward_npu(
-        indexer_weight_stream = get_indexer_weight_stream()
-        indexer_weight_stream.wait_stream(torch.npu.current_stream())
-        with torch.npu.stream(indexer_weight_stream):
+        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18607 - [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp

- Link: https://github.com/sgl-project/sglang/pull/18607
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +9/-5, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix accuracy issue when running TP4 dsv3 model with mtp"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile`; PR body summary: cc @HaiShaw, @kkHuang-amd Fixed dpsk-r1-fp4 accuracy issue on tp4 with mtp. - aiter backend change in sglang. - aiter update for tp4 fp8 prefill attn kernel (head=32). (merged,....
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +7/-3 (10 lines); hunks: -195,11 +195,15 @@ def __init__(; -301,7 +305,7 @@ def make_mla_meta_data(; symbols: __init__, make_mla_meta_data, touching `__init__, make_mla_meta_data`; `docker/rocm.Dockerfile` modified +2/-2 (4 lines); hunks: -21,7 +21,7 @@ ENV BUILD_TRITON="0"; -31,7 +31,7 @@ ENV BUILD_TRITON="0".
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +7/-3 (10 lines); hunks: -195,11 +195,15 @@ def __init__(; -301,7 +305,7 @@ def make_mla_meta_data(; symbols: __init__, make_mla_meta_data
  - `docker/rocm.Dockerfile` modified +2/-2 (4 lines); hunks: -21,7 +21,7 @@ ENV BUILD_TRITON="0"; -31,7 +31,7 @@ ENV BUILD_TRITON="0"
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -195,11 +195,15 @@ def __init__(
+            if self.num_head == 32:
+                fast_mode = True
+                intra_batch_mode = False
-            # for non-fp8 kv_cache, use non-persist kernel to avoid performance degradation
-            if self.kv_cache_dtype is not fp8_dtype:
+            # for non-fp8 kv_cache on tp8, use non-persist kernel to avoid performance degradation
diff -- docker/rocm.Dockerfile
@@ -21,7 +21,7 @@ ENV BUILD_TRITON="0"
-ENV AITER_COMMIT="v0.1.10.post2"
+ENV AITER_COMMIT="v0.1.10.post3"
@@ -31,7 +31,7 @@ ENV BUILD_TRITON="0"
-ENV AITER_COMMIT="v0.1.10.post2"
+ENV AITER_COMMIT="v0.1.10.post3"
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +7/-3
  - other: `docker/rocm.Dockerfile` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16907 - Fix model loading for DeepSeek-V3.2-AWQ

- Link: https://github.com/sgl-project/sglang/pull/16907
- Status/date: merged / 2026-02-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `190fa8246fbc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-4, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix model loading for DeepSeek-V3.2-AWQ"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Got the following error when using DeepSeek-V3.2-AWQ (https://huggingface.co/QuantTrio/DeepSeek-V3.2-AWQ/) Add check of whether `weight_packed` exists before using Tested on gsm....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +8/-4 (12 lines); hunks: -245,10 +245,14 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-4 (12 lines); hunks: -245,10 +245,14 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -245,10 +245,14 @@ def __init__(
-        if not hasattr(self.gate_up_proj, "weight"):
-            self.gate_up_proj.weight = getattr(self.gate_up_proj, "weight_packed")
-        if not hasattr(self.down_proj, "weight"):
-            self.down_proj.weight = getattr(self.down_proj, "weight_packed")
+        if not hasattr(self.gate_up_proj, "weight") and hasattr(
+            self.gate_up_proj, "weight_packed"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +8/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18389 - Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4

- Link: https://github.com/sgl-project/sglang/pull/18389
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `0ffd0a3995e5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +352/-183, 914 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/kernels/test_nsa_indexer.py`; PR body summary: 17655 - support Deepseek v3.2 NVFP4 with trtllm mla sparse fp8 attention backend - update the nsa backend to support trtllm sparse fp8 attention backend - update the deepseek v2....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunks: -33,7 +33,10; -340,6 +343,7 @@ def __init__(; symbols: __init__, forward_extend, touching `__init__, forward_extend`; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -1488,6 +1488,12 @@ def _fuse_rope_for_trtllm_mla(self, forward_batch: Forwar...; symbols: _fuse_rope_for_trtllm_mla, touching `_fuse_rope_for_trtllm_mla`; `test/registered/kernels/test_nsa_indexer.py` modified +1/-0 (1 lines); hunks: -232,6 +232,7 @@ def __init__(self, config=None):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunks: -33,7 +33,10; -340,6 +343,7 @@ def __init__(; symbols: __init__, forward_extend
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -1488,6 +1488,12 @@ def _fuse_rope_for_trtllm_mla(self, forward_batch: Forwar...; symbols: _fuse_rope_for_trtllm_mla
  - `test/registered/kernels/test_nsa_indexer.py` modified +1/-0 (1 lines); hunks: -232,6 +232,7 @@ def __init__(self, config=None):; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -33,7 +33,10 @@
-from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general
+from sglang.srt.layers.attention.utils import (
+    concat_mla_absorb_q_general,
+    mla_quantize_and_rope_for_fp8,
+)
@@ -340,6 +343,7 @@ def __init__(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1488,6 +1488,12 @@ def _fuse_rope_for_trtllm_mla(self, forward_batch: ForwardBatch) -> bool:
+        if self.current_attention_backend == "nsa":
+            return (
+                get_global_server_args().nsa_decode_backend == "trtllm"
+                or get_global_server_args().nsa_prefill_backend == "trtllm"
+            ) and forward_batch.attn_backend.kv_cache_dtype == torch.float8_e4m3fn
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -232,6 +232,7 @@ def __init__(self, config=None):
+            kv_cache_dim=self.config["kv_lora_rank"] + self.config["qk_rope_head_dim"],
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/hicache/test_nsa_pool_host_unit.py`, `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18892 - [Kernel] Add JIT support for DeepSeek V3 GEMM

- Link: https://github.com/sgl-project/sglang/pull/18892
- Status/date: open / 2026-02-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +1284/-0, 1299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Add JIT support for DeepSeek V3 GEMM"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh`, `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`; PR body summary: This PR migrates DeepSeek V3 GEMM kernels to the JIT (Just-In-Time) compilation framework, including: - `dsv3_fused_a_gemm` - `dsv3_router_gemm` related roadmap: https://github.....
- Key implementation: `python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh` added +602/-0 (602 lines); hunks: -0,0 +1,602; symbols: Type, touching `Type`; `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +176/-0 (176 lines); hunks: -0,0 +1,176; `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: benchmark_bf16, tflops, benchmark_fp32, touching `benchmark_bf16, tflops, benchmark_fp32`; `python/sglang/jit_kernel/benchmark/bench_dsv3_fused_a_gemm.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _make_inputs, benchmark, tflops, touching `_make_inputs, benchmark, tflops`.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh` added +602/-0 (602 lines); hunks: -0,0 +1,602; symbols: Type
  - `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +176/-0 (176 lines); hunks: -0,0 +1,176
  - `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +118/-0 (118 lines); hunks: -0,0 +1,118; symbols: benchmark_bf16, tflops, benchmark_fp32
  - `python/sglang/jit_kernel/benchmark/bench_dsv3_fused_a_gemm.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _make_inputs, benchmark, tflops
  - `python/sglang/jit_kernel/dsv3_fused_a_gemm.py` added +82/-0 (82 lines); hunks: -0,0 +1,82; symbols: _jit_dsv3_fused_a_gemm_module, jit_dsv3_fused_a_gemm, dsv3_fused_a_gemm
- Key code excerpts:

```diff
diff -- python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh
@@ -0,0 +1,602 @@
+/*
+ * Adapted from
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/619709fc33bd5dc268f19d6a741fe7ed51c0f8f5/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3FusedAGemm.cu
+ *
+ * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
+ * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
diff -- python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh
@@ -0,0 +1,176 @@
+/*
+ * Adapted from
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
+ * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
+ *
+ * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
diff -- python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py
@@ -0,0 +1,118 @@
```

- Reviewed files:
  - runtime: `python/sglang/jit_kernel/csrc/gemm/dsv3_fused_a_gemm.cuh` added +602/-0; `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +176/-0; `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +118/-0; `python/sglang/jit_kernel/benchmark/bench_dsv3_fused_a_gemm.py` added +83/-0; `python/sglang/jit_kernel/dsv3_fused_a_gemm.py` added +82/-0; `python/sglang/jit_kernel/dsv3_router_gemm.py` added +76/-0
  - tests: `python/sglang/jit_kernel/tests/test_dsv3_fused_a_gemm.py` added +69/-0; `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py` added +61/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_dsv3_fused_a_gemm.py`, `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18978 - [AMD] Fix mi35x dsv32 mtp nightly

- Link: https://github.com/sgl-project/sglang/pull/18978
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `462267982bd8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix mi35x dsv32 mtp nightly"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa_backend.py`; PR body summary: PR #17554 introduced `fused_metadata_copy`, a JIT-compiled CUDA kernel enabled by default. This kernel cannot run on ROCm/HIP — it fails on every CUDA graph replay iteration, pr....
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: -72,7 +72,7.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunks: -72,7 +72,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -72,7 +72,7 @@
-_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get()
+_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get() and not _is_hip
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19041 - [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`

- Link: https://github.com/sgl-project/sglang/pull/19041
- Status/date: merged / 2026-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `eddf193292d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +48/-9, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`; PR body summary: @humansand From recent GLM-5 tech report: > Compared with the non-deterministic CUDA-based top‑k implementation used in SGLang’s DSA Indexer, directly using the naive torch.topk....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +17/-7 (24 lines); hunks: -229,21 +229,31 @@ def _with_real_sm_count(self):; symbols: _with_real_sm_count, _project_and_scale_head_gates, _weights_proj_bf16_in_fp32_out, _get_logits_head_gate, touching `_with_real_sm_count, _project_and_scale_head_gates, _weights_proj_bf16_in_fp32_out`; `test/registered/kernels/test_nsa_indexer.py` modified +2/-2 (4 lines); hunks: -24,7 +24,7; -34,7 +34,7.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +17/-7 (24 lines); hunks: -229,21 +229,31 @@ def _with_real_sm_count(self):; symbols: _with_real_sm_count, _project_and_scale_head_gates, _weights_proj_bf16_in_fp32_out, _get_logits_head_gate
  - `test/registered/kernels/test_nsa_indexer.py` modified +2/-2 (4 lines); hunks: -24,7 +24,7; -34,7 +34,7
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -229,21 +229,31 @@ def _with_real_sm_count(self):
-    @torch.compile(dynamic=True) if not _is_hip else lambda f: f
-    def _project_and_scale_head_gates(self, x: torch.Tensor):
+    def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor) -> torch.Tensor:
+        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
+            weight = self.weights_proj.weight
+            out = torch.empty(
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -24,7 +24,7 @@
-register_cuda_ci(est_time=2, suite="stage-b-test-small-1-gpu")
+register_cuda_ci(est_time=2, suite="stage-b-test-large-1-gpu")
@@ -34,7 +34,7 @@
-    "index_n_heads": 1,
+    "index_n_heads": 32,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +17/-7
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18624 - [AMD] DSR1/V3 use fp8 bmm in MLA for MI300X

- Link: https://github.com/sgl-project/sglang/pull/18624
- Status/date: merged / 2026-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `c193a52fa263`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-9, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] DSR1/V3 use fp8 bmm in MLA for MI300X"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Improve TPOT by using use fp8 bmm in MLA on MI300X for DSR1/V3 bs=16, ISL 8k OSL 1K Before **q@w_kc: bmm** **attn_output@w_vc: bmm** After **q@w_kc: bmm** **attn_output@w_vc: bmm**.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-9 (22 lines); hunks: -119,6 +119,7; -158,11 +159,11; symbols: forward, forward_absorb_prepare, forward_absorb_core, touching `forward, forward_absorb_prepare, forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-9 (22 lines); hunks: -119,6 +119,7; -158,11 +159,11; symbols: forward, forward_absorb_prepare, forward_absorb_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -119,6 +119,7 @@
+from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
@@ -158,11 +159,11 @@
-if _use_aiter_gfx95:
+if _use_aiter:
+if _use_aiter_gfx95:
@@ -564,8 +565,6 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18242 - [ROCm] Optimize Deepseek R1 on MI300X

- Link: https://github.com/sgl-project/sglang/pull/18242
- Status/date: merged / 2026-02-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `b9cf1563de97`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +7/-2, 44 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Optimize Deepseek R1 on MI300X"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: The DeepSeek-V3/R1 model on AMD MI300X (gfx942) was previously limited to fallback code paths for several MLA attention operations because the optimized kernels were gated behin....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -180,6 +180,9; -1724,7 +1727,7 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: -180,6 +180,9; -1724,7 +1727,7 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -180,6 +180,9 @@
+if _use_aiter:
+    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla
@@ -1724,7 +1727,7 @@ def forward_absorb_prepare(
-            and (not _use_aiter or not _is_gfx95_supported or self.use_nsa)
+            and (not _use_aiter or not _is_hip or self.use_nsa)
@@ -1780,7 +1783,7 @@ def forward_absorb_core(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19148 - [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache

- Link: https://github.com/sgl-project/sglang/pull/19148
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `4e843f121657`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +307/-21, 386 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`; PR body summary: In DeepSeek v3.2, after the Indexer produces key in bf16 (roughly (N, 128)), it needs to populate NSA’s index_k_with_scale_buffer. The previous implementation used two steps: 1.....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +79/-21 (100 lines); hunks: -7,6 +7,10; -670,15 +674,15 @@ def _forward_cuda_k_only(; symbols: _forward_cuda_k_only, forward_indexer, _store_index_k_cache, forward_cuda, touching `_forward_cuda_k_only, forward_indexer, _store_index_k_cache`; `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` added +124/-0 (124 lines); hunks: -0,0 +1,124.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +79/-21 (100 lines); hunks: -7,6 +7,10; -670,15 +674,15 @@ def _forward_cuda_k_only(; symbols: _forward_cuda_k_only, forward_indexer, _store_index_k_cache, forward_cuda
  - `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` added +124/-0 (124 lines); hunks: -0,0 +1,124
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -7,6 +7,10 @@
+from sglang.jit_kernel.fused_store_index_cache import (
+    can_use_nsa_fused_store,
+    fused_store_index_k_cache,
+)
@@ -670,15 +674,15 @@ def _forward_cuda_k_only(
-        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
diff -- python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh
@@ -0,0 +1,124 @@
+#include <sgl_kernel/tensor.h>
+#include <sgl_kernel/utils.h>
+#include <sgl_kernel/math.cuh>
+#include <sgl_kernel/type.cuh>
+#include <sgl_kernel/utils.cuh>
+#include <sgl_kernel/vec.cuh>
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +79/-21; `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` added +124/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/jit_kernel/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19425 - [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4

- Link: https://github.com/sgl-project/sglang/pull/19425
- Status/date: merged / 2026-02-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-2, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; PR body summary: > Co-authored-by: Haoyang Li > Co-authored-by: billishyahao This patch is to fix tensor shape mismatch issue for amd dsr1 0528 mxfp4 https://huggingface.co/amd/DeepSeek-R1-0528-....
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +11/-0 (11 lines); hunks: -49,6 +49,7; -60,6 +61,7; symbols: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN, touching `DeepseekModelNextN, __init__, forward`; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-2 (9 lines); hunks: -52,6 +52,7 @@ def __init__(; -69,13 +70,17 @@ def get_min_capability(cls) -> int:; symbols: __init__, get_min_capability, get_name, apply_weight_name_mapper, touching `__init__, get_min_capability, get_name`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +11/-0 (11 lines); hunks: -49,6 +49,7; -60,6 +61,7; symbols: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-2 (9 lines); hunks: -52,6 +52,7 @@ def __init__(; -69,13 +70,17 @@ def get_min_capability(cls) -> int:; symbols: __init__, get_min_capability, get_name, apply_weight_name_mapper
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -49,6 +49,7 @@
+from sglang.srt.models.utils import WeightsMapper
@@ -60,6 +61,7 @@
@@ -190,6 +192,15 @@ def forward(
+    # Support amd/DeepSeek-R1-0528-MXFP4 renaming: model.layers.61*.
+    # Ref: HF config.json for amd/DeepSeek-R1-0528-MXFP4
+    # https://huggingface.co/amd/DeepSeek-R1-0528-MXFP4/blob/main/config.json
diff -- python/sglang/srt/layers/quantization/quark/quark.py
@@ -52,6 +52,7 @@ def __init__(
+        self.exclude_layers = cast(list[str], self.quant_config.get("exclude", []))
@@ -69,13 +70,17 @@ def get_min_capability(cls) -> int:
+    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
+        self.exclude_layers = hf_to_sglang_mapper.apply_list(self.exclude_layers)
-        exclude_layers = cast(list[str], self.quant_config.get("exclude"))
-            prefix, ignore=exclude_layers, fused_mapping=self.packed_modules_mapping
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +11/-0; `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/quark/quark.py`, `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19122 - [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py

- Link: https://github.com/sgl-project/sglang/pull/19122
- Status/date: merged / 2026-02-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +906/-818, 1896 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; PR body summary: Part of #16255 Split different MLA forward methods into separate files, including: - `forward_mla.py` for Cuda/HIP mla aborption - `forward_mla_fused_rope_cpu.py‎` for cpu optim....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +22/-811 (833 lines); hunks: -19,7 +19,6; -33,7 +32,6; symbols: DeepseekV2MLP, __init__, op_output, DeepseekV2AttentionMLA, touching `DeepseekV2MLP, __init__, op_output`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0 (492 lines); hunks: -0,0 +1,492; symbols: DeepseekMLAForwardMixin, init_mla_forward, forward_absorb_prepare, forward_absorb_core, touching `DeepseekMLAForwardMixin, init_mla_forward, forward_absorb_prepare`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0 (227 lines); hunks: -0,0 +1,227; symbols: DeepseekMLARocmForwardMixin, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare, forward_absorb_fused_mla_rope_core, touching `DeepseekMLARocmForwardMixin, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: DeepseekMLACpuForwardMixin, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare, forward_absorb_fused_mla_rope_cpu_core, touching `DeepseekMLACpuForwardMixin, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +22/-811 (833 lines); hunks: -19,7 +19,6; -33,7 +32,6; symbols: DeepseekV2MLP, __init__, op_output, DeepseekV2AttentionMLA
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0 (492 lines); hunks: -0,0 +1,492; symbols: DeepseekMLAForwardMixin, init_mla_forward, forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0 (227 lines); hunks: -0,0 +1,227; symbols: DeepseekMLARocmForwardMixin, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare, forward_absorb_fused_mla_rope_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0 (152 lines); hunks: -0,0 +1,152; symbols: DeepseekMLACpuForwardMixin, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare, forward_absorb_fused_mla_rope_cpu_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` modified +6/-0 (6 lines); hunks: -1,7 +1,13
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -19,7 +19,6 @@
-import os
@@ -33,7 +32,6 @@
-from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
@@ -107,11 +105,6 @@
-from sglang.srt.layers.quantization.fp8_kernel import (
-    fp8_dtype,
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -0,0 +1,492 @@
+from __future__ import annotations
+from typing import TYPE_CHECKING, Optional
+import torch
+from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
+from sglang.srt.layers import deep_gemm_wrapper
+from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py
@@ -0,0 +1,227 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +22/-811; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` modified +6/-0; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +1/-1
  - tests: `test/srt/cpu/test_qkv_proj_with_rope.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/test_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19428 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/19428
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `b3718982a1b4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +101/-9, 316 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] add feature mla_ag_after_qlora for dsv3.2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; PR body summary: In DeepSeek-V3.2, due to the large hidden dimension of the model, performing all-gather on hidden_states before attention introduces a high-dimensional communication operation.....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +33/-2 (35 lines); hunks: -12,6 +12,7; -43,13 +44,15; symbols: forward_npu, do_npu_cp_balance_indexer, touching `forward_npu, do_npu_cp_balance_indexer`; `python/sglang/srt/models/deepseek_v2.py` modified +26/-3 (29 lines); hunks: -1301,13 +1301,15 @@ def forward(; -1318,6 +1320,7 @@ def forward_prepare(; symbols: forward, forward_prepare, __init__, touching `forward, forward_prepare, __init__`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +35/-3 (38 lines); hunks: -4,21 +4,24; -28,6 +31,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu, forward_mla_prepare_npu, forward_dsa_prepare_npu, touching `forward_mha_prepare_npu, forward_mla_prepare_npu, forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +33/-2 (35 lines); hunks: -12,6 +12,7; -43,13 +44,15; symbols: forward_npu, do_npu_cp_balance_indexer
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-3 (29 lines); hunks: -1301,13 +1301,15 @@ def forward(; -1318,6 +1320,7 @@ def forward_prepare(; symbols: forward, forward_prepare, __init__
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +35/-3 (38 lines); hunks: -4,21 +4,24; -28,6 +31,7 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu, forward_mla_prepare_npu, forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -12,6 +12,7 @@
+from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
@@ -43,13 +44,15 @@
+from sglang.srt.layers.communicator import ScatterMode
+_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
@@ -1203,6 +1206,7 @@ def forward_npu(
+        layer_scatter_modes=None,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1301,13 +1301,15 @@ def forward(
+        layer_scatter_modes: LayerScatterModes = None,
+            layer_scatter_modes=layer_scatter_modes,
@@ -1318,6 +1320,7 @@ def forward_prepare(
+        layer_scatter_modes: LayerScatterModes = None,
@@ -1370,15 +1373,30 @@ def forward_prepare(
-                self, positions, hidden_states, forward_batch, zero_allocator
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -4,21 +4,24 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +33/-2; `python/sglang/srt/models/deepseek_v2.py` modified +26/-3; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +35/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18174 - [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON

- Link: https://github.com/sgl-project/sglang/pull/18174
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `6af0448cc9bf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-3, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`; PR body summary: LLMs may generate tool call content that are not correct json format, we must catch this error for normal streaming output..
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +6/-3 (9 lines); hunks: -158,9 +158,12 @@ def _parse_parameters_from_xml(; symbols: _parse_parameters_from_xml, touching `_parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +6/-3 (9 lines); hunks: -158,9 +158,12 @@ def _parse_parameters_from_xml(; symbols: _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -158,9 +158,12 @@ def _parse_parameters_from_xml(
-                    parameters[param_name] = _partial_json_loads(
-                        param_value, Allow.ALL
-                    )[0]
+                    try:
+                        parameters[param_name] = _partial_json_loads(
+                            param_value, Allow.ALL
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +6/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16091 - [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format.

- Link: https://github.com/sgl-project/sglang/pull/16091
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `666caaf9ce76`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-22, 151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format."; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/function_call/deepseekv32_detector.py`; PR body summary: The previous implementation only supported streaming function call parameters in XML format. 1. Convert `_parse_parameters_from_xml` to return `str`, unifying the handling of JS....
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +17/-21 (38 lines); hunks: -85,6 +85,7 @@ def __init__(self):; -93,27 +94,24 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _parse_parameters_from_xml, touching `__init__, has_tool_call, _parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +17/-21 (38 lines); hunks: -85,6 +85,7 @@ def __init__(self):; -93,27 +94,24 @@ def has_tool_call(self, text: str) -> bool:; symbols: __init__, has_tool_call, _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -85,6 +85,7 @@ def __init__(self):
+        self.prefix_invoke_end_call = ["</", "｜DSML｜", "inv", "oke"]
@@ -93,27 +94,24 @@ def has_tool_call(self, text: str) -> bool:
-    ) -> dict:
+    ) -> str:
-        Parse parameters from either XML-like format or JSON format to dict.
+        Parse parameters from either XML-like format or JSON format to str.
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +17/-21
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19834 - [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek)

- Link: https://github.com/sgl-project/sglang/pull/19834
- Status/date: merged / 2026-03-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1614/-177, 2530 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek)"; model line: DeepSeek V3/R1; category: bug fix; main diff: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; PR body summary: Track accuracy and performance regression for two new DeepSeek-R1-MXFP4 server configurations on MI35x: - `--kv-cache-dtype fp8_e4m3` - `--enable-aiter-allreduce-fusion` Update....
- Key implementation: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name, touching `get_model_path, ModelConfig, __post_init__`; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: get_model_path, ModelConfig, __post_init__, get_display_name, touching `get_model_path, ModelConfig, __post_init__`; `.github/workflows/nightly-test-amd-rocm720.yml` modified +153/-68 (221 lines); hunks: -21,46 +21,10 @@ on:; -98,7 +62,7 @@ jobs:; `.github/workflows/nightly-test-amd.yml` modified +153/-68 (221 lines); hunks: -21,46 +21,10 @@ on:; -98,7 +62,7 @@ jobs:.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` added +281/-0 (281 lines); hunks: -0,0 +1,281; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` added +280/-0 (280 lines); hunks: -0,0 +1,280; symbols: get_model_path, ModelConfig, __post_init__, get_display_name
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +153/-68 (221 lines); hunks: -21,46 +21,10 @@ on:; -98,7 +62,7 @@ jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +153/-68 (221 lines); hunks: -21,46 +21,10 @@ on:; -98,7 +62,7 @@ jobs:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` added +178/-0 (178 lines); hunks: -0,0 +1,178; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py
@@ -0,0 +1,281 @@
+"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with KV Cache FP8 (8-GPU)
+Tests DeepSeek-R1-MXFP4 quantized model with --kv-cache-dtype fp8_e4m3
+using few-shot completion benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-kv-fp8 suite
+"""
+import ast
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py
@@ -0,0 +1,280 @@
+"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with AIter AllReduce Fusion (8-GPU)
+Tests DeepSeek-R1-MXFP4 quantized model with --enable-aiter-allreduce-fusion
+using few-shot completion benchmark on MI35x.
+Registry: nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4-ar-fusion suite
+"""
+import ast
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -21,46 +21,10 @@ on:
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` added +281/-0; `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` added +280/-0; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` added +178/-0; `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py` added +177/-0; `test/registered/amd/test_deepseek_v3_mtp_kv_fp8.py` added +116/-0
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +153/-68; `.github/workflows/nightly-test-amd.yml` modified +153/-68; `.github/workflows/pr-test-amd-rocm720.yml` modified +129/-16
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_ar_fusion_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19843 - [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3

- Link: https://github.com/sgl-project/sglang/pull/19843
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `84aaa69795f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +12/-7, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: In the AITER (AMD MI355X) path with FP8 quantization, was initialized in float32, but at runtime in biased_grouped_topk it gets cast to match gating_output.dtype: correction_bia....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +12/-7 (19 lines); hunks: -271,13 +271,18 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-7 (19 lines); hunks: -271,13 +271,18 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -271,13 +271,18 @@ def __init__(
-            correction_bias_dtype = (
-                torch.bfloat16
-                if quant_config is not None
-                and quant_config.get_name() == "modelopt_fp4"
-                and get_moe_runner_backend().is_flashinfer_trtllm()
-                else torch.float32
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +12/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19319 - [deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug

- Link: https://github.com/sgl-project/sglang/pull/19319
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; associated commits `006bd44cf920`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +380/-81, 740 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug"; model line: DeepSeek V3/R1; category: bug fix; main diff: `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; PR body summary: fix get_k_and_s_triton kenel for 128K seqlen case bug.
- Key implementation: `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` added +191/-0 (191 lines); hunks: -0,0 +1,191; symbols: golden_torch_gen, get_k_and_s_triton, touching `golden_torch_gen, get_k_and_s_triton`; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +105/-48 (153 lines); hunks: -167,19 +167,30 @@ def execute(cls, *args, **kwargs):; -599,7 +610,9 @@ def _get_s_triton_kernel(; symbols: execute, triton, _get_s_triton_kernel, _get_k_and_s_triton, touching `execute, triton, _get_s_triton_kernel`; `test/manual/layers/attention/nsa/test_index_buf_accessor.py` modified +46/-9 (55 lines); hunks: -264,6 +264,7 @@ def test_get_k_and_s_correctness(; -283,13 +284,16 @@ def test_get_k_and_s_correctness(; symbols: test_get_k_and_s_correctness, test_get_k_and_s_sequential_pages, test_get_k_and_s_repeated_pages, touching `test_get_k_and_s_correctness, test_get_k_and_s_sequential_pages, test_get_k_and_s_repeated_pages`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-22 (51 lines); hunks: -491,6 +491,7 @@ def _should_chunk_mqa_logits(; -509,9 +510,11 @@ def _get_topk_ragged(; symbols: _should_chunk_mqa_logits, _get_topk_ragged, forward_cuda, touching `_should_chunk_mqa_logits, _get_topk_ragged, forward_cuda`.
- Code diff details:
  - `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` added +191/-0 (191 lines); hunks: -0,0 +1,191; symbols: golden_torch_gen, get_k_and_s_triton
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +105/-48 (153 lines); hunks: -167,19 +167,30 @@ def execute(cls, *args, **kwargs):; -599,7 +610,9 @@ def _get_s_triton_kernel(; symbols: execute, triton, _get_s_triton_kernel, _get_k_and_s_triton
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` modified +46/-9 (55 lines); hunks: -264,6 +264,7 @@ def test_get_k_and_s_correctness(; -283,13 +284,16 @@ def test_get_k_and_s_correctness(; symbols: test_get_k_and_s_correctness, test_get_k_and_s_sequential_pages, test_get_k_and_s_repeated_pages
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-22 (51 lines); hunks: -491,6 +491,7 @@ def _should_chunk_mqa_logits(; -509,9 +510,11 @@ def _get_topk_ragged(; symbols: _should_chunk_mqa_logits, _get_topk_ragged, forward_cuda
- Key code excerpts:

```diff
diff -- test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py
@@ -0,0 +1,191 @@
+import torch
+from sglang.srt.layers.attention.nsa.index_buf_accessor import (
+    _get_k_and_s_triton_kernel,
+)
+def golden_torch_gen(
+    seq_len_tensor: torch.Tensor,
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -167,19 +167,30 @@ def execute(cls, *args, **kwargs):
-        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
+        cls,
+        pool: "NSATokenToKVPool",
+        buf: torch.Tensor,
+        page_indices: torch.Tensor,
+        seq_len_tensor: torch.Tensor,
diff -- test/manual/layers/attention/nsa/test_index_buf_accessor.py
@@ -264,6 +264,7 @@ def test_get_k_and_s_correctness(
```

- Reviewed files:
  - tests: `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` added +191/-0; `test/manual/layers/attention/nsa/test_index_buf_accessor.py` modified +46/-9
  - runtime: `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +105/-48; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-22
- Risk and verification: The diff ships test coverage in `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18280 - [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp

- Link: https://github.com/sgl-project/sglang/pull/18280
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `17031120b8f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-4, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; PR body summary: Fix PR #16043 #19319 Prefill CP scenario bug. In the DeepSeek V3.2 prefill with CP (context parallel) scenario, `forward_batch.seq_lens` and `forward_batch.seq_lens_cpu` are glo....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +9/-3 (12 lines); hunks: -97,6 +97,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; -538,11 +543,12 @@ def _get_topk_ragged(; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu, _get_topk_ragged, touching `get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-0 (8 lines); hunks: -138,6 +138,8 @@ class NSAMetadata:; -194,6 +196,9 @@ def get_cu_seqlens_k(self) -> torch.Tensor:; symbols: NSAMetadata, get_cu_seqlens_k, get_indexer_kvcache_range, get_indexer_seq_len, touching `NSAMetadata, get_cu_seqlens_k, get_indexer_kvcache_range`; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +1/-1 (2 lines); hunks: -624,7 +624,7 @@ def _get_k_and_s_triton(; symbols: _get_k_and_s_triton, touching `_get_k_and_s_triton`; `test/registered/kernels/test_nsa_indexer.py` modified +4/-0 (4 lines); hunks: -132,6 +132,10 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu, touching `get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +9/-3 (12 lines); hunks: -97,6 +97,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; -538,11 +543,12 @@ def _get_topk_ragged(; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-0 (8 lines); hunks: -138,6 +138,8 @@ class NSAMetadata:; -194,6 +196,9 @@ def get_cu_seqlens_k(self) -> torch.Tensor:; symbols: NSAMetadata, get_cu_seqlens_k, get_indexer_kvcache_range, get_indexer_seq_len
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +1/-1 (2 lines); hunks: -624,7 +624,7 @@ def _get_k_and_s_triton(; symbols: _get_k_and_s_triton
  - `test/registered/kernels/test_nsa_indexer.py` modified +4/-0 (4 lines); hunks: -132,6 +132,10 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:; symbols: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -97,6 +97,11 @@ def get_indexer_seq_len_cpu(self) -> torch.Tensor:
+    def get_indexer_seq_len(self) -> torch.Tensor:
+        """
+        Return: seq lens for each batch.
+        """
@@ -538,11 +543,12 @@ def _get_topk_ragged(
-        seq_len_sum = forward_batch.seq_lens_sum
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -138,6 +138,8 @@ class NSAMetadata:
+    # seq lens for each batch.
+    indexer_seq_lens: Optional[torch.Tensor] = None
@@ -194,6 +196,9 @@ def get_cu_seqlens_k(self) -> torch.Tensor:
+    def get_indexer_seq_len(self) -> torch.Tensor:
+        return self.attn_metadata.indexer_seq_lens
@@ -404,6 +409,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -624,7 +624,7 @@ def _get_k_and_s_triton(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +9/-3; `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-0; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +1/-1
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9744 - [CPU] Add FP8 Bmm support

- Link: https://github.com/sgl-project/sglang/pull/9744
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +585/-84, 1014 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add FP8 Bmm support"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/longcat_flash.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; PR body summary: This PR is a follow-up on https://github.com/sgl-project/sglang/issues/8281 to add FP8 BMM support in MLA weight absorb. - The main change is the C++ kernels for fp8 bmm on CPU:....
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, touching `forward_absorb_prepare, forward_absorb_core`; `python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights, touching `post_load_weights`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28 (72 lines); hunks: -16,6 +16,7; -268,18 +269,24 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/longcat_flash.py` modified +0/-12 (12 lines); hunks: -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10 (10 lines); hunks: -46,8 +46,6; -583,14 +581,6 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8 (8 lines); hunks: -1208,14 +1208,6 @@ def post_load_weights(self, is_nextn=False, weight_names=...; symbols: post_load_weights
  - `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4 (4 lines); hunks: -426,10 +426,6 @@ def post_load_weights(self):; symbols: post_load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -16,6 +16,7 @@
+    _is_cpu,
@@ -268,18 +269,24 @@ def forward_absorb_prepare(
-            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
-            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
-                q_nope.transpose(0, 1),
-                (
diff -- python/sglang/srt/models/longcat_flash.py
@@ -760,18 +760,6 @@ def post_load_weights(self, weight_names=None):
-                    # TODO: remove this after adding FP8 support in bmm cpu kernel
-                    if (
-                        _is_cpu
-                        and _is_cpu_amx_available
-                        and w.dtype == torch.float8_e4m3fn
-                    ):
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -46,8 +46,6 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +44/-28; `python/sglang/srt/models/longcat_flash.py` modified +0/-12; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +0/-10; `python/sglang/srt/models/bailing_moe_linear.py` modified +0/-8; `python/sglang/srt/models/longcat_flash_nextn.py` modified +0/-4; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +2/-1
  - other: `sgl-kernel/csrc/cpu/gemm_fp8.cpp` modified +310/-1; `sgl-kernel/csrc/cpu/bmm.cpp` modified +93/-11
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_bmm.py`, `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18451 - [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256.

- Link: https://github.com/sgl-project/sglang/pull/18451
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `85fe8c6793a0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256."; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: This change modified the router kernel selection with ROCm backend for models using DeepSeek model architecture. The Aiter kernel aiter_dsv3_router_gemm has been optimized for t....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -325,7 +325,11 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: -325,7 +325,11 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -325,7 +325,11 @@ def forward(
-            elif _use_aiter_gfx95 and hidden_states.shape[0] <= 256:
+            elif (
+                _use_aiter_gfx95
+                and hidden_states.shape[0] <= 256
+                and self.weight.shape[0] <= 256
+            ):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20841 - Fix gpu-fault issue when run deepseek-r1 and enable dp

- Link: https://github.com/sgl-project/sglang/pull/20841
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix gpu-fault issue when run deepseek-r1 and enable dp"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/layers/attention/aiter_backend.py`; PR body summary: Fix the gpu cored-dump. Root cause: During CUDA graph capture, the captured data was fed with a KV sequence length smaller than the query (Q) length, which caused an illegal mem....
- Key implementation: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -1774,7 +1774,7 @@ def init_forward_metadata_replay_cuda_graph(; symbols: init_forward_metadata_replay_cuda_graph, get_cuda_graph_seq_len_fill_value, update_verify_buffers_to_fill_after_draft, touching `init_forward_metadata_replay_cuda_graph, get_cuda_graph_seq_len_fill_value, update_verify_buffers_to_fill_after_draft`.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: -1774,7 +1774,7 @@ def init_forward_metadata_replay_cuda_graph(; symbols: init_forward_metadata_replay_cuda_graph, get_cuda_graph_seq_len_fill_value, update_verify_buffers_to_fill_after_draft
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -1774,7 +1774,7 @@ def init_forward_metadata_replay_cuda_graph(
-        return 1
+        return 1 if self.num_draft_tokens is None else self.num_draft_tokens
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/aiter_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20438 - [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/20438
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `649172879778`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; PR body summary: In the current Native Sparse Attention (NSA) implementation, specifically when Context Parallelism (CP) is enabled via `--enable-nsa-prefill-context-parallel`, the key_all_gathe....
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-0 (19 lines); hunks: -329,6 +329,25 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16, touching `_get_q_k_bf16`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-0 (19 lines); hunks: -329,6 +329,25 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -329,6 +329,25 @@ def _get_q_k_bf16(
+        elif (
+            self.alt_stream is not None
+            and forward_batch.nsa_cp_metadata is not None
+            and self.nsa_enable_prefill_cp
+        ):
+            key = rotate_activation(key)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19945 - [AMD] Tilelang sparse fwd for dsv32 mi355/mi300

- Link: https://github.com/sgl-project/sglang/pull/19945
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; associated commits `855d15adf657`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +141/-95, 314 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Tilelang sparse fwd for dsv32 mi355/mi300"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; PR body summary: - Enable the faster/new tilelang kernel on MI300. - Improve longer-context kernel performance on MI355. - Enable the new tilelang partial+combine path with MI300 config: block_I....
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +141/-95 (236 lines); hunks: -790,16 +790,22 @@ def sparse_mla_fwd_decode_partial(; -815,20 +821,22 @@ def sparse_mla_fwd_decode_partial(; symbols: sparse_mla_fwd_decode_partial, main, tilelang_sparse_fwd, touching `sparse_mla_fwd_decode_partial, main, tilelang_sparse_fwd`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +141/-95 (236 lines); hunks: -790,16 +790,22 @@ def sparse_mla_fwd_decode_partial(; -815,20 +821,22 @@ def sparse_mla_fwd_decode_partial(; symbols: sparse_mla_fwd_decode_partial, main, tilelang_sparse_fwd
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -790,16 +790,22 @@ def sparse_mla_fwd_decode_partial(
+    inner_iter=1,
+    num_stages=1,
-    grid: (seq_len * REPLICATE_H, top_k_blocks).
-    Each block does one topk block, writes partial_o, partial_lse.
+    grid: (seq_len * REPLICATE_H, top_k / block_I / inner_iter)
+    Each GPU block processes `inner_iter` consecutive KV tiles and writes one (partial_o, partial_lse) entry.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +141/-95
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21526 - Fix aiter router GEMM regression for non-DSR1 MoE models on ROCm gfx95

- Link: https://github.com/sgl-project/sglang/pull/21526
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix aiter router GEMM regression for non-DSR1 MoE models on ROCm gfx95"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: aiter_dsv3_router_gemm uses a Triton atomic kernel (_gemm_a16_w16_atomic) that is specifically tuned for DeepSeek-R1/V3's hidden_size=7168. On models with different hidden dimen....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunks: -330,6 +330,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -330,6 +330,7 @@ def forward(
+                and hidden_states.shape[1] == 7168  # tuned for DSR1/V3
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21529 - Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm

- Link: https://github.com/sgl-project/sglang/pull/21529
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +308/-126, 644 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Enable MXFP4 (including Quark W4A4) quantized models (e.g. GLM-5-MXFP4-Q8, Kimi-K2.5-MXFP4) to load and run on AMD MI355X with ROCm. These models use Quark's W4A4 MXFP4 scheme w....
- Key implementation: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +85/-44 (129 lines); hunks: -33,8 +33,7; -52,7 +51,7 @@ def __init__(self, weight_config: dict[str, Any], input_config...; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner, touching `__init__, create_weights, process_weights_after_loading`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +28/-14 (42 lines); hunks: -297,6 +297,21 @@ def __init__(; -444,8 +459,11 @@ def _load_w13(; symbols: __init__, _load_w13, _load_w2, weight_loader, touching `__init__, _load_w13, _load_w2`; `python/sglang/srt/models/deepseek_v2.py` modified +20/-2 (22 lines); hunks: -2057,8 +2057,10 @@ def forward(; -2160,6 +2162,22 @@ def determine_num_fused_shared_experts(; symbols: forward, DeepseekV2ForCausalLM, __init__, determine_num_fused_shared_experts, touching `forward, DeepseekV2ForCausalLM, __init__`; `python/sglang/srt/layers/quantization/mxfp4.py` modified +7/-1 (8 lines); hunks: -298,7 +298,7 @@ def get_quant_method(; -332,6 +332,12 @@ def create_weights(; symbols: get_quant_method, get_scaled_act_names, create_weights, touching `get_quant_method, get_scaled_act_names, create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +85/-44 (129 lines); hunks: -33,8 +33,7; -52,7 +51,7 @@ def __init__(self, weight_config: dict[str, Any], input_config...; symbols: __init__, create_weights, process_weights_after_loading, create_moe_runner
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +28/-14 (42 lines); hunks: -297,6 +297,21 @@ def __init__(; -444,8 +459,11 @@ def _load_w13(; symbols: __init__, _load_w13, _load_w2, weight_loader
  - `python/sglang/srt/models/deepseek_v2.py` modified +20/-2 (22 lines); hunks: -2057,8 +2057,10 @@ def forward(; -2160,6 +2162,22 @@ def determine_num_fused_shared_experts(; symbols: forward, DeepseekV2ForCausalLM, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +7/-1 (8 lines); hunks: -298,7 +298,7 @@ def get_quant_method(; -332,6 +332,12 @@ def create_weights(; symbols: get_quant_method, get_scaled_act_names, create_weights
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +6/-2 (8 lines); hunks: -85,7 +85,9 @@ def get_quant_method(; -94,7 +96,9 @@ def get_quant_method(; symbols: get_quant_method
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py
@@ -33,8 +33,7 @@
-    from aiter.ops.shuffle import shuffle_weight
-    from aiter.utility.fp4_utils import e8m0_shuffle
+    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
@@ -52,7 +51,7 @@ def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):
-            )  # noqa E501
+            )
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -297,6 +297,21 @@ def __init__(
+        # Pre-compute expert_mask for CUDA graph compatibility (EP mode)
+        if (
+            _use_aiter
+            and getattr(self.dispatcher, "local_expert_mapping", None) is not None
+        ):
+            expert_mask = (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2057,8 +2057,10 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +85/-44; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +28/-14; `python/sglang/srt/models/deepseek_v2.py` modified +20/-2; `python/sglang/srt/layers/quantization/mxfp4.py` modified +7/-1; `python/sglang/srt/layers/quantization/quark/quark.py` modified +6/-2; `python/sglang/srt/layers/quantization/quark/utils.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21530 - [ROCm] Fix fused MLA decode rope path for Kimi K2.5 and DeepSeek-variant models

- Link: https://github.com/sgl-project/sglang/pull/21530
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +103/-33, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm] Fix fused MLA decode rope path for Kimi K2.5 and DeepSeek-variant models"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`, `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`; PR body summary: The SGLANG_ROCM_FUSED_DECODE_MLA=1 path had several issues preventing it from working with models like Kimi K2.5 that use DeepseekScalingRotaryEmbedding or different ForwardMeta....
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` modified +68/-19 (87 lines); hunks: -25,6 +25,53; -42,9 +89,6 @@ def forward_absorb_fused_mla_rope_prepare(; symbols: _get_cos_sin_cache, _ensure_cuda_2d, _bmm_absorb, DeepseekMLARocmForwardMixin, touching `_get_cos_sin_cache, _ensure_cuda_2d, _bmm_absorb`; `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` modified +35/-14 (49 lines); hunks: -20,6 +20,7; -115,7 +116,7 @@ def _fwd_grouped_kernel_stage1_rope(; symbols: _fwd_grouped_kernel_stage1_rope, decode_attention_fwd_grouped_rope, touching `_fwd_grouped_kernel_stage1_rope, decode_attention_fwd_grouped_rope`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` modified +68/-19 (87 lines); hunks: -25,6 +25,53; -42,9 +89,6 @@ def forward_absorb_fused_mla_rope_prepare(; symbols: _get_cos_sin_cache, _ensure_cuda_2d, _bmm_absorb, DeepseekMLARocmForwardMixin
  - `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` modified +35/-14 (49 lines); hunks: -20,6 +20,7; -115,7 +116,7 @@ def _fwd_grouped_kernel_stage1_rope(; symbols: _fwd_grouped_kernel_stage1_rope, decode_attention_fwd_grouped_rope
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py
@@ -25,6 +25,53 @@
+def _get_cos_sin_cache(rotary_emb):
+    """Extract a (max_seq_len, rotary_dim) cos_sin_cache tensor.
+    The Triton kernel expects layout [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]
+    per position.  Different RotaryEmbedding subclasses store the cache under
+    different attribute names and shapes; this helper normalises them all.
+    """
diff -- python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py
@@ -20,6 +20,7 @@
+import torch
@@ -115,7 +116,7 @@ def _fwd_grouped_kernel_stage1_rope(
-    k_pe_last_token = tl.zeros([BLOCK_R], dtype=q.dtype)
+    k_pe_last_token = tl.zeros([BLOCK_R], dtype=tl.float32)
@@ -183,18 +184,15 @@ def _fwd_grouped_kernel_stage1_rope(
-            # debug assert
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` modified +68/-19; `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` modified +35/-14
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21531 - [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel

- Link: https://github.com/sgl-project/sglang/pull/21531
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +450/-39, 560 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe_lite.py`, `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`; PR body summary: dsv3_router_gemm_bf16_out.cu → kOutFloat=false path dsv3_router_gemm_float_out.cu → kOutFloat=true path dsv3_router_gemm_entry.cu → RouterGemmDispatcher replace LoopUnroller htt....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-6 (19 lines); hunks: -162,7 +162,14; -317,13 +324,13 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/glm4_moe_lite.py` modified +10/-5 (15 lines); hunks: -181,13 +181,18 @@ def forward(self, hidden_states, gemm_output_zero_allocato...; symbols: forward, touching `forward`; `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +184/-0 (184 lines); hunks: -0,0 +1,184; `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: benchmark_bf16_output, benchmark_float32_output, touching `benchmark_bf16_output, benchmark_float32_output`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-6 (19 lines); hunks: -162,7 +162,14; -317,13 +324,13 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +10/-5 (15 lines); hunks: -181,13 +181,18 @@ def forward(self, hidden_states, gemm_output_zero_allocato...; symbols: forward
  - `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +184/-0 (184 lines); hunks: -0,0 +1,184
  - `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: benchmark_bf16_output, benchmark_float32_output
  - `python/sglang/jit_kernel/dsv3_router_gemm.py` added +103/-0 (103 lines); hunks: -0,0 +1,103; symbols: _jit_dsv3_router_gemm_module, can_use_dsv3_router_gemm, dsv3_router_gemm
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -162,7 +162,14 @@
-    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
+    from sgl_kernel import dsv3_fused_a_gemm
+    from sglang.jit_kernel.dsv3_router_gemm import (
+        can_use_dsv3_router_gemm as _can_use_dsv3_router_gemm,
+    )
+    from sglang.jit_kernel.dsv3_router_gemm import (
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -181,13 +181,18 @@ def forward(self, hidden_states, gemm_output_zero_allocator: BumpAllocator = Non
-            and _device_sm >= 90
-            from sgl_kernel import dsv3_router_gemm
-            logits = dsv3_router_gemm(hidden_states, self.weight).to(
-                hidden_states.dtype
+            from sglang.jit_kernel.dsv3_router_gemm import (
+                can_use_dsv3_router_gemm,
diff -- python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh
@@ -0,0 +1,184 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-6; `python/sglang/srt/models/glm4_moe_lite.py` modified +10/-5; `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +184/-0; `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +104/-0; `python/sglang/jit_kernel/dsv3_router_gemm.py` added +103/-0
  - tests: `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py` added +36/-0
  - other: `sgl-kernel/python/sgl_kernel/gemm.py` modified +0/-19; `sgl-kernel/CMakeLists.txt` modified +0/-3
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18461 - [Intel GPU] Enable DeepSeek R1 inference on XPU

- Link: https://github.com/sgl-project/sglang/pull/18461
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `f0303fd07eb0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +46/-28, 245 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Intel GPU] Enable DeepSeek R1 inference on XPU"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Enable DeepSeek-R1 model inference for XPU use FP8 precision through triton 1. run bmm in bf16 for absorb 2. Add XPU support for benchmarking 3. Add EP support Not Applicable No....
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -50,6 +50,7; -497,7 +498,7 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7; -40,6 +41,7; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -137,6 +137,7; -677,6 +678,7 @@ def _post_combine_hook(; symbols: _post_combine_hook, touching `_post_combine_hook`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: -50,6 +50,7; -497,7 +498,7 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0 (2 lines); hunks: -31,6 +31,7; -40,6 +41,7
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: -137,6 +137,7; -677,6 +678,7 @@ def _post_combine_hook(; symbols: _post_combine_hook
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -50,6 +50,7 @@
+    _is_xpu,
@@ -497,7 +498,7 @@ def post_load_weights(
-                        _is_cuda
+                        (_is_cuda or _is_xpu)
diff -- python/sglang/srt/models/deepseek_common/utils.py
@@ -31,6 +31,7 @@
+    is_xpu,
@@ -40,6 +41,7 @@
+_is_xpu = is_xpu()
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -137,6 +137,7 @@
+    _is_xpu,
@@ -677,6 +678,7 @@ def _post_combine_hook(
+            and not _is_xpu
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1; `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0; `python/sglang/srt/models/deepseek_v2.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/token_dispatcher/standard.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14162 - DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- Link: https://github.com/sgl-project/sglang/pull/14162
- Status/date: merged / 2026-03-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +94/-12, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; PR body summary: profiling: deepseek-R1-0528: deepseek-R1-0528-w4afp8: 1.When DeepEP is enabled, the communication latency of the DeepSeek-R1-0508-W4AFP8 model is twice that of the DeepSeek-R1-0....
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton, touching `silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: -25,6 +25,7; -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, touching `cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: -331,6 +331,9 @@ def forward_cutlass_w4afp8(; -342,9 +345,6 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked, touching `forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: -334,10 +334,11 @@ def apply_deepep_ll(; symbols: apply_deepep_ll, touching `apply_deepep_ll`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: -25,6 +25,7; -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: -331,6 +331,9 @@ def forward_cutlass_w4afp8(; -342,9 +345,6 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: -334,10 +334,11 @@ def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(
+@triton.jit
+def _fp8_per_token_quant_to_per_tensor_quant_kernel(
+    x_ptr,
+    x_scale_ptr,
+    x_scale_stride0,
+    x_scale_stride1,
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -25,6 +25,7 @@
+    fp8_per_token_to_per_tensor_quant_triton,
@@ -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(
-    a: torch.Tensor,
+    a_states: torch.Tensor,
+    a_scales: torch.Tensor,
@@ -473,7 +475,7 @@ def cutlass_w4a8_moe_deepep_ll(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -331,6 +331,9 @@ def forward_cutlass_w4afp8(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3; `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21719 - Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication"

- Link: https://github.com/sgl-project/sglang/pull/21719
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +12/-94, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication""; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; PR body summary: Reverts sgl-project/sglang#14162.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +0/-73 (73 lines); hunks: -1381,76 +1381,3 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton, touching `silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +7/-15 (22 lines); hunks: -25,7 +25,6; -412,8 +411,7 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, touching `cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: -331,9 +331,6 @@ def forward_cutlass_w4afp8(; -345,6 +342,9 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked, touching `forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-2 (3 lines); hunks: -334,11 +334,10 @@ def apply_deepep_ll(; symbols: apply_deepep_ll, touching `apply_deepep_ll`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +0/-73 (73 lines); hunks: -1381,76 +1381,3 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +7/-15 (22 lines); hunks: -25,7 +25,6; -412,8 +411,7 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: -331,9 +331,6 @@ def forward_cutlass_w4afp8(; -345,6 +342,9 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-2 (3 lines); hunks: -334,11 +334,10 @@ def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -1381,76 +1381,3 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(
-@triton.jit
-def _fp8_per_token_quant_to_per_tensor_quant_kernel(
-    x_ptr,
-    x_scale_ptr,
-    x_scale_stride0,
-    x_scale_stride1,
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -25,7 +25,6 @@
-    fp8_per_token_to_per_tensor_quant_triton,
@@ -412,8 +411,7 @@ def cutlass_w4a8_moe_deepep_normal(
-    a_states: torch.Tensor,
-    a_scales: torch.Tensor,
+    a: torch.Tensor,
@@ -475,7 +473,7 @@ def cutlass_w4a8_moe_deepep_ll(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -331,9 +331,6 @@ def forward_cutlass_w4afp8(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +0/-73; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +7/-15; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3; `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-2; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21657 - [AMD] Use tgemm.mm for MoEGate router gemm in deepseek_v2.py

- Link: https://github.com/sgl-project/sglang/pull/21657
- Status/date: merged / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-32, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Use tgemm.mm for MoEGate router gemm in deepseek_v2.py"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/rocm_linear_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: When running the GLM-5-fp8 model, the MoEGate router GEMM is a major bottleneck, taking about **~30%** of a single layer's time. Following the ATOM approach, this change switche....
- Key implementation: `python/sglang/srt/layers/rocm_linear_utils.py` modified +3/-23 (26 lines); hunks: -1,37 +1,17; symbols: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size, touching `aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size`; `python/sglang/srt/models/deepseek_v2.py` modified +5/-9 (14 lines); hunks: -153,9 +153,11; -327,14 +329,8 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/rocm_linear_utils.py` modified +3/-23 (26 lines); hunks: -1,37 +1,17; symbols: aiter_dsv3_router_gemm, get_dsv3_gemm_output_zero_allocator_size
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-9 (14 lines); hunks: -153,9 +153,11; -327,14 +329,8 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/rocm_linear_utils.py
@@ -1,37 +1,17 @@
-from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
-from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
-from sglang.srt.utils import BumpAllocator
+from aiter.tuned_gemm import tgemm
-    gemm_output_zero_allocator: BumpAllocator = None,
-    M = hidden_states.shape[0]
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -153,9 +153,11 @@
+if _use_aiter:
+    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm
-        aiter_dsv3_router_gemm,
@@ -327,14 +329,8 @@ def forward(
-            elif (
-                _use_aiter_gfx95
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rocm_linear_utils.py` modified +3/-23; `python/sglang/srt/models/deepseek_v2.py` modified +5/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rocm_linear_utils.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21280 - [RL] Support mxfp8 DeepSeek V3

- Link: https://github.com/sgl-project/sglang/pull/21280
- Status/date: merged / 2026-04-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +105/-45, 204 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[RL] Support mxfp8 DeepSeek V3"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; PR body summary: @humansand Support Blackwell mxfp8 DeepSeek RL. Since the `kv_b_proj` can have different contraction axis in absorbed vs non-absorbed MLA mode while mxfp8 is 1d quantization, fo....
- Key implementation: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +86/-38 (124 lines); hunks: -47,6 +47,10; -126,10 +130,13 @@ def align_fp8_moe_weights_for_flashinfer_trtllm(; symbols: align_fp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm, touching `align_fp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-7 (19 lines); hunks: -291,13 +291,18 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/quantization/fp8.py` modified +7/-0 (7 lines); hunks: -93,6 +93,7; -241,6 +242,12 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, apply_weight_name_mapper, Fp8LinearMethod, touching `get_quant_method, get_scaled_act_names, apply_weight_name_mapper`.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +86/-38 (124 lines); hunks: -47,6 +47,10; -126,10 +130,13 @@ def align_fp8_moe_weights_for_flashinfer_trtllm(; symbols: align_fp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-7 (19 lines); hunks: -291,13 +291,18 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/quantization/fp8.py` modified +7/-0 (7 lines); hunks: -93,6 +93,7; -241,6 +242,12 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, apply_weight_name_mapper, Fp8LinearMethod
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py
@@ -47,6 +47,10 @@
+_flashinfer_trtllm_shuffle_row_indices_cache_mxfp8: dict[
+    tuple, dict[str, torch.Tensor]
+] = {}
@@ -126,10 +130,13 @@ def align_fp8_moe_weights_for_flashinfer_trtllm(
-    from flashinfer import (
-        reorder_rows_for_gated_act_gemm,
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -291,13 +291,18 @@ def __init__(
-        self.should_fuse_routed_scaling_factor_in_topk = isinstance(
-            self.quant_method, ModelOptNvFp4FusedMoEMethod
-        ) or (
-            isinstance(self.quant_method, Fp8MoEMethod)
-            and (
-                get_moe_runner_backend().is_cutlass()
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -93,6 +93,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +86/-38; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-7; `python/sglang/srt/layers/quantization/fp8.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17707 - Add dsv3 router gemm benchmark on blackwell

- Link: https://github.com/sgl-project/sglang/pull/17707
- Status/date: merged / 2026-04-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `9fa12d605af2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +284/-4, 310 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add dsv3 router gemm benchmark on blackwell"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: * Comparing dsv3_router_gemm performance between flashinfer and the current sglang kernel to optimize performance (https://github.com/sgl-project/sglang/issues/14453) * Add accu....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +34/-4 (38 lines); hunks: -165,7 +165,10; -324,11 +327,20 @@ def forward(; symbols: forward, DeepseekV32ForCausalLM, flashinfer_dsv3_router_gemm, touching `forward, DeepseekV32ForCausalLM, flashinfer_dsv3_router_gemm`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-4 (38 lines); hunks: -165,7 +165,10; -324,11 +327,20 @@ def forward(; symbols: forward, DeepseekV32ForCausalLM, flashinfer_dsv3_router_gemm
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -165,7 +165,10 @@
+    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
+    from sglang.srt.utils.custom_op import register_custom_op
@@ -324,11 +327,20 @@ def forward(
+                if _device_sm >= 100 and self.weight.shape[0] == 256:
+                    # router gemm output float32
+                    logits = torch.empty(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +34/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21405 - Enable IndexCache for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/21405
- Status/date: merged / 2026-04-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `5a3531641735`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +196/-20, 358 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable IndexCache for DeepSeek V3.2"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; PR body summary: fix https://github.com/sgl-project/sglang/issues/21286 * Port https://github.com/THUDM/IndexCache * add ut for deepseek-v3.2 gsm8k with DeepSeek-V3.2-Exp with this PR: main Thro....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -1106,6 +1106,7 @@ def __init__(; -1175,6 +1176,8 @@ def __init__(; symbols: __init__, op_prepare, op_core, touching `__init__, op_prepare, op_core`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, touching `forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -1106,6 +1106,7 @@ def __init__(; -1175,6 +1176,8 @@ def __init__(; symbols: __init__, op_prepare, op_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1106,6 +1106,7 @@ def __init__(
+        is_nextn: bool = False,
@@ -1175,6 +1176,8 @@ def __init__(
+        self.skip_topk = None
+        self.next_skip_topk = None
@@ -1195,6 +1198,26 @@ def __init__(
+            # Refer: https://arxiv.org/abs/2603.12201 for more details.
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -91,6 +91,7 @@ def forward_absorb_prepare(
+        prev_topk_indices: Optional[torch.Tensor] = None,
@@ -182,25 +183,31 @@ def forward_absorb_prepare(
-                topk_indices = self.indexer(
-                    x=hidden_states,
-                    q_lora=q_lora,
-                    positions=positions,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22268 - [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3

- Link: https://github.com/sgl-project/sglang/pull/22268
- Status/date: open / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`; PR body summary: When LoRA adapters are applied to `fused_qkv_a_proj_with_mqa` in DeepSeek V2/V3 models, the fused GEMM optimization path in `prepare_qkv_latent` reads `.weight` directly from th....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -1562,11 +1562,16 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent, touching `prepare_qkv_latent`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -1562,11 +1562,16 @@ def prepare_qkv_latent(; symbols: prepare_qkv_latent
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1562,11 +1562,16 @@ def prepare_qkv_latent(
+        # When LoRA adapters wrap the projection, the fused GEMM path reads
+        # .weight directly and would bypass the LoRA delta.  Detect this by
+        # checking for the ``base_layer`` attribute that all LoRA wrappers add.
+        is_lora_wrapped = hasattr(self.fused_qkv_a_proj_with_mqa, "base_layer")
+            and not is_lora_wrapped
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20089 - feat: [1/2] [DeepEP] Fuse shared expert into MoE dispatch under EP

- Link: https://github.com/sgl-project/sglang/pull/20089
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +199/-49, 448 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: [1/2] [DeepEP] Fuse shared expert into MoE dispatch under EP"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; PR body summary: In DeepSeek V3/R1 with expert parallelism (EP), each rank currently computes the shared expert **separately** from routed experts. This means shared expert is not part of the De....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +84/-25 (109 lines); hunks: -103,6 +103,9; -364,11 +367,34 @@ def __init__(; symbols: __init__, forward_deepep, touching `__init__, forward_deepep`; `python/sglang/srt/layers/moe/topk.py` modified +82/-8 (90 lines); hunks: -36,7 +36,11; -49,6 +53,7; symbols: biased_grouped_topk_gpu, biased_grouped_topk_cpu, _remap_topk_for_deepep, touching `biased_grouped_topk_gpu, biased_grouped_topk_cpu, _remap_topk_for_deepep`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +20/-16 (36 lines); hunks: -48,7 +48,7; -197,10 +197,20 @@ def __init__(; symbols: __init__, _load_g_idx, _map_global_expert_id_to_local_expert_id, weight_loader, touching `__init__, _load_g_idx, _map_global_expert_id_to_local_expert_id`; `python/sglang/srt/layers/moe/utils.py` modified +6/-0 (6 lines); hunks: -254,6 +254,12 @@ def is_sbo_enabled() -> bool:; symbols: is_sbo_enabled, is_deepep_class_backend, get_tbo_token_distribution_threshold, touching `is_sbo_enabled, is_deepep_class_backend, get_tbo_token_distribution_threshold`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +84/-25 (109 lines); hunks: -103,6 +103,9; -364,11 +367,34 @@ def __init__(; symbols: __init__, forward_deepep
  - `python/sglang/srt/layers/moe/topk.py` modified +82/-8 (90 lines); hunks: -36,7 +36,11; -49,6 +53,7; symbols: biased_grouped_topk_gpu, biased_grouped_topk_cpu, _remap_topk_for_deepep
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +20/-16 (36 lines); hunks: -48,7 +48,7; -197,10 +197,20 @@ def __init__(; symbols: __init__, _load_g_idx, _map_global_expert_id_to_local_expert_id, weight_loader
  - `python/sglang/srt/layers/moe/utils.py` modified +6/-0 (6 lines); hunks: -254,6 +254,12 @@ def is_sbo_enabled() -> bool:; symbols: is_sbo_enabled, is_deepep_class_backend, get_tbo_token_distribution_threshold
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -663,6 +663,7 @@ class ServerArgs:; -5828,6 +5829,12 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -103,6 +103,9 @@
+    is_deepep_class_backend,
+    is_sbo_enabled,
+    is_tbo_enabled,
@@ -364,11 +367,34 @@ def __init__(
-        self.num_fused_shared_experts = (
-            0
diff -- python/sglang/srt/layers/moe/topk.py
@@ -36,7 +36,11 @@
-from sglang.srt.distributed import get_tp_group
+from sglang.srt.distributed import (
+    get_moe_expert_parallel_rank,
+    get_moe_expert_parallel_world_size,
+    get_tp_group,
+)
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -48,7 +48,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +84/-25; `python/sglang/srt/layers/moe/topk.py` modified +82/-8; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +20/-16; `python/sglang/srt/layers/moe/utils.py` modified +6/-0; `python/sglang/srt/server_args.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/moe/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22323 - [Lora] Lora quat info re-factor and support deepseekv3 mla lora

- Link: https://github.com/sgl-project/sglang/pull/22323
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +458/-80, 811 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Lora] Lora quat info re-factor and support deepseekv3 mla lora"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/quantization/moe_wna16.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py`; PR body summary: - MoE quant-info refactor Extracts get_triton_quant_info() into each quantization method (FP8, INT8, WNA16, etc.) so FusedMoEWithLoRA correctly receives quantization scales/flag....
- Key implementation: `python/sglang/srt/layers/quantization/fp8.py` modified +21/-20 (41 lines); hunks: -1487,6 +1487,26 @@ def create_moe_runner(; -1663,26 +1683,7 @@ def apply(; symbols: create_moe_runner, get_triton_quant_info, apply, touching `create_moe_runner, get_triton_quant_info, apply`; `python/sglang/srt/layers/quantization/moe_wna16.py` modified +13/-11 (24 lines); hunks: -364,19 +364,10 @@ def create_moe_runner(; -387,6 +378,17 @@ def apply(; symbols: create_moe_runner, apply, get_triton_quant_info, touching `create_moe_runner, apply, get_triton_quant_info`; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +13/-10 (23 lines); hunks: -331,6 +331,18 @@ def create_moe_runner(; -365,14 +377,5 @@ def apply(; symbols: create_moe_runner, get_triton_quant_info, apply, touching `create_moe_runner, get_triton_quant_info, apply`; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +10/-7 (17 lines); hunks: -360,13 +360,8 @@ def create_moe_runner(; -377,4 +372,12 @@ def apply(; symbols: create_moe_runner, apply, get_triton_quant_info, touching `create_moe_runner, apply, get_triton_quant_info`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8.py` modified +21/-20 (41 lines); hunks: -1487,6 +1487,26 @@ def create_moe_runner(; -1663,26 +1683,7 @@ def apply(; symbols: create_moe_runner, get_triton_quant_info, apply
  - `python/sglang/srt/layers/quantization/moe_wna16.py` modified +13/-11 (24 lines); hunks: -364,19 +364,10 @@ def create_moe_runner(; -387,6 +378,17 @@ def apply(; symbols: create_moe_runner, apply, get_triton_quant_info
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +13/-10 (23 lines); hunks: -331,6 +331,18 @@ def create_moe_runner(; -365,14 +377,5 @@ def apply(; symbols: create_moe_runner, get_triton_quant_info, apply
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +10/-7 (17 lines); hunks: -360,13 +360,8 @@ def create_moe_runner(; -377,4 +372,12 @@ def apply(; symbols: create_moe_runner, apply, get_triton_quant_info
  - `python/sglang/srt/layers/quantization/w8a8_fp8.py` modified +10/-7 (17 lines); hunks: -286,13 +286,8 @@ def create_moe_runner(; -302,4 +297,12 @@ def apply(; symbols: create_moe_runner, apply, get_triton_quant_info
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -1487,6 +1487,26 @@ def create_moe_runner(
+    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
+        return TritonMoeQuantInfo(
+            w13_weight=layer.w13_weight,
+            w2_weight=layer.w2_weight,
+            b13=getattr(layer, "w13_weight_bias", None),
+            b2=getattr(layer, "w2_weight_bias", None),
diff -- python/sglang/srt/layers/quantization/moe_wna16.py
@@ -364,19 +364,10 @@ def create_moe_runner(
-    def apply(
-        self,
-        layer: torch.nn.Module,
-        dispatch_output: StandardDispatchOutput,
-    ) -> CombineInput:
-        assert (
diff -- python/sglang/srt/layers/quantization/w8a8_int8.py
@@ -331,6 +331,18 @@ def create_moe_runner(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8.py` modified +21/-20; `python/sglang/srt/layers/quantization/moe_wna16.py` modified +13/-11; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +13/-10; `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +10/-7; `python/sglang/srt/layers/quantization/w8a8_fp8.py` modified +10/-7; `python/sglang/srt/layers/quantization/unquant.py` modified +9/-6
  - tests: `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py` added +156/-0
- Risk and verification: The diff ships test coverage in `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22316 - [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- Link: https://github.com/sgl-project/sglang/pull/22316
- Status/date: merged / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +91/-12, 177 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; PR body summary: profiling: deepseek-R1-0528: deepseek-R1-0528-w4afp8: 1.When DeepEP is enabled, the communication latency of the DeepSeek-R1-0508-W4AFP8 model is twice that of the DeepSeek-R1-0....
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton, touching `silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: -25,6 +25,7; -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, touching `cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3 (3 lines); hunks: -342,9 +342,6 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8_masked, touching `forward_cutlass_w4afp8_masked`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: -334,10 +334,11 @@ def apply_deepep_ll(; symbols: apply_deepep_ll, touching `apply_deepep_ll`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: -25,6 +25,7; -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3 (3 lines); hunks: -342,9 +342,6 @@ def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: -334,10 +334,11 @@ def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; symbols: _dispatch_core
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/kernels.py
@@ -1381,3 +1381,76 @@ def silu_and_mul_masked_post_per_tensor_quant_fwd(
+@triton.jit
+def _fp8_per_token_quant_to_per_tensor_quant_kernel(
+    x_ptr,
+    x_scale_ptr,
+    x_scale_stride0,
+    x_scale_stride1,
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -25,6 +25,7 @@
+    fp8_per_token_to_per_tensor_quant_triton,
@@ -411,7 +412,8 @@ def cutlass_w4a8_moe_deepep_normal(
-    a: torch.Tensor,
+    a_states: torch.Tensor,
+    a_scales: torch.Tensor,
@@ -473,7 +475,7 @@ def cutlass_w4a8_moe_deepep_ll(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -342,9 +342,6 @@ def forward_cutlass_w4afp8_masked(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3; `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22938 - [AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122)

- Link: https://github.com/sgl-project/sglang/pull/22938
- Status/date: open / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-8, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122)"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`; PR body summary: Re-apply #18242 and #18624 behavior in forward_mla.py (logic moved out of deepseek_v2.py by #19122 but not fully carried over): - Import fused_qk_rope_cat_and_cache_mla for all....
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-5 (17 lines); hunks: -19,7 +19,6; -67,6 +66,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; symbols: bmm_fp8, DeepseekMLAForwardMixin, forward_absorb_prepare, forward_absorb_core, touching `bmm_fp8, DeepseekMLAForwardMixin, forward_absorb_prepare`; `python/sglang/srt/models/deepseek_v2.py` modified +0/-3 (3 lines); hunks: -168,9 +168,6.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-5 (17 lines); hunks: -19,7 +19,6; -67,6 +66,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; symbols: bmm_fp8, DeepseekMLAForwardMixin, forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-3 (3 lines); hunks: -168,9 +168,6
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -19,7 +19,6 @@
-    _is_gfx95_supported,
@@ -67,6 +66,9 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
+    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla
@@ -78,7 +80,6 @@ def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
-    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla
@@ -317,7 +318,7 @@ def forward_absorb_prepare(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -168,9 +168,6 @@
-if _use_aiter:
-    pass
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-5; `python/sglang/srt/models/deepseek_v2.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/22128
- Status/date: merged / 2026-04-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +272/-18, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Allow piecewise CUDA graph with speculative decoding"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`; PR body summary: - Allow `--enable-piecewise-cuda-graph` to coexist with all speculative decoding algorithms (EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM) - Previously all speculative algorithms disable....
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

### PR #23195 - [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors

- Link: https://github.com/sgl-project/sglang/pull/23195
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +138/-14, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors"; model line: DeepSeek V3/R1; category: bug fix; main diff: `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`; PR body summary: Hey all, this is my first PR, thought I'd try my hand a a recent bug. Fixes #22855. When a DeepseekV2-architecture checkpoint is loaded with a `compressed_tensors`-wrapped AWQ q....
- Key implementation: `test/registered/unit/models/test_deepseek_v2_attention_mla.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: TestDeepseekV2AttentionMLA, _make_attn, test_get_fused_qkv_a_proj_weight_returns_none_when_missing, test_can_use_min_latency_fused_a_gemm_preserves_bf16_path, touching `TestDeepseekV2AttentionMLA, _make_attn, test_get_fused_qkv_a_proj_weight_returns_none_when_missing`; `python/sglang/srt/models/deepseek_v2.py` modified +18/-9 (27 lines); hunks: -1135,6 +1135,23 @@ class DeepseekV2AttentionMLA(; -1351,15 +1368,7 @@ def __init__(; symbols: DeepseekV2AttentionMLA, _get_fused_qkv_a_proj_weight, _can_use_min_latency_fused_a_gemm, __init__, touching `DeepseekV2AttentionMLA, _get_fused_qkv_a_proj_weight, _can_use_min_latency_fused_a_gemm`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +5/-4 (9 lines); hunks: -29,15 +29,16 @@ def init_mla_fused_rope_cpu_forward(self: DeepseekV2Attentio...; symbols: init_mla_fused_rope_cpu_forward, touching `init_mla_fused_rope_cpu_forward`; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +4/-1 (5 lines); hunks: -29,7 +29,10 @@ def _dispatch_mla_subtype(attn, forward_batch):; symbols: _dispatch_mla_subtype, touching `_dispatch_mla_subtype`.
- Code diff details:
  - `test/registered/unit/models/test_deepseek_v2_attention_mla.py` added +111/-0 (111 lines); hunks: -0,0 +1,111; symbols: TestDeepseekV2AttentionMLA, _make_attn, test_get_fused_qkv_a_proj_weight_returns_none_when_missing, test_can_use_min_latency_fused_a_gemm_preserves_bf16_path
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-9 (27 lines); hunks: -1135,6 +1135,23 @@ class DeepseekV2AttentionMLA(; -1351,15 +1368,7 @@ def __init__(; symbols: DeepseekV2AttentionMLA, _get_fused_qkv_a_proj_weight, _can_use_min_latency_fused_a_gemm, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +5/-4 (9 lines); hunks: -29,15 +29,16 @@ def init_mla_fused_rope_cpu_forward(self: DeepseekV2Attentio...; symbols: init_mla_fused_rope_cpu_forward
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +4/-1 (5 lines); hunks: -29,7 +29,10 @@ def _dispatch_mla_subtype(attn, forward_batch):; symbols: _dispatch_mla_subtype
- Key code excerpts:

```diff
diff -- test/registered/unit/models/test_deepseek_v2_attention_mla.py
@@ -0,0 +1,111 @@
+import unittest
+from types import SimpleNamespace
+from unittest.mock import patch
+import torch
+import torch.nn as nn
+from sglang.srt.models.deepseek_common.attention_backend_handler import (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1135,6 +1135,23 @@ class DeepseekV2AttentionMLA(
+    def _get_fused_qkv_a_proj_weight(self):
+        if not getattr(self, "has_fused_proj", False):
+            return None
+        return getattr(self.fused_qkv_a_proj_with_mqa, "weight", None)
+    def _can_use_min_latency_fused_a_gemm(self) -> bool:
+        fused_weight = self._get_fused_qkv_a_proj_weight()
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py
@@ -29,15 +29,16 @@ def init_mla_fused_rope_cpu_forward(self: DeepseekV2AttentionMLA):
```

- Reviewed files:
  - tests: `test/registered/unit/models/test_deepseek_v2_attention_mla.py` added +111/-0
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +18/-9; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +5/-4; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `test/registered/unit/models/test_deepseek_v2_attention_mla.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23257 - Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention

- Link: https://github.com/sgl-project/sglang/pull/23257
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-0, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`; PR body summary: Running `nvidia/DeepSeek-V3-0324-NVFP4` with `--moe-runner-backend flashinfer_cutedsl --enable-dp-attention --ep-size N --dp-size N` (N>1) hit two separate bugs. **Bug 1 — doubl....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunks: -89,6 +89,7; -648,6 +649,7 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook, touching `forward_normal_dual_stream, _post_combine_hook`; `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` modified +2/-0 (2 lines); hunks: -249,6 +249,8 @@ def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:; symbols: ensure_cutedsl_wrapper, touching `ensure_cutedsl_wrapper`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunks: -89,6 +89,7; -648,6 +649,7 @@ def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` modified +2/-0 (2 lines); hunks: -249,6 +249,8 @@ def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:; symbols: ensure_cutedsl_wrapper
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -89,6 +89,7 @@
+    should_use_dp_reduce_scatterv,
@@ -648,6 +649,7 @@ def forward_normal_dual_stream(
+            and not should_use_dp_reduce_scatterv()
@@ -736,6 +738,7 @@ def _post_combine_hook(
+            and not should_use_dp_reduce_scatterv()
diff -- python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py
@@ -249,6 +249,8 @@ def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:
+    if server_args.enable_dp_attention:
+        max_num_tokens *= server_args.dp_size
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +3/-0; `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1296/-33, 1579 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`; PR body summary: One of the core parameters of speculative decoding is `speculative_num_steps`, which controls how many autoregressive draft-model steps are executed in each round. It directly d....
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
- Motivation: Title: "Opt-in strip of thinking tokens from radix cache"; model line: DeepSeek V3/R1; category: bug fix; main diff: `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; PR body summary: Fixes #22373. Closes #22617, #22950. Opt-in: enable with `--strip-thinking-cache`. Off by default. Why Reasoning-model requests (`--reasoning-parser `) insert all output tokens....
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
- Motivation: Title: "[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)"; model line: DeepSeek V3/R1; category: bug fix; main diff: `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`; PR body summary: Fix dead reasoning branches in radix cache for multi-turn separated-thinking models Fixes #22373. Related to #22617. What is the problem Reasoning models (QwQ-32B, DeepSeek-R1,....
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

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- Link: https://github.com/sgl-project/sglang/pull/23336
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +193/-10, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SPEC V2][2/N] feat: adaptive spec support spec v2"; model line: DeepSeek V3/R1; category: performance/backend optimization; main diff: `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; PR body summary: adaptive spec support spec v2: 1. launch sgl+adaptive spec+spec v2 2. benchmark 3. result Models: - Target model: `/models/ZhipuAI/GLM-4.7-FP8` - Draft model: `/models/ZhipuAI/G....
- Key implementation: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: -30,8 +30,13; -671,6 +676,13 @@ def __init__(; symbols: __init__, target_worker, forward_batch_generation, on_verify_complete_cpu, touching `__init__, target_worker, forward_batch_generation`; `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: Sched...; -163,7 +167,7 @@ def prepare_for_decode(self: EagleDraftInput, batch: Schedul...; symbols: prepare_for_decode, prepare_for_v2_draft, touching `prepare_for_decode, prepare_for_v2_draft`; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: -358,8 +358,17 @@ def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids, touching `_resolve_spec_overlap_token_ids`; `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: -32,11 +32,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason, touching `adaptive_unsupported_reason`.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: -30,8 +30,13; -671,6 +676,13 @@ def __init__(; symbols: __init__, target_worker, forward_batch_generation, on_verify_complete_cpu
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: Sched...; -163,7 +167,7 @@ def prepare_for_decode(self: EagleDraftInput, batch: Schedul...; symbols: prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: -358,8 +358,17 @@ def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: -32,11 +32,6 @@ def adaptive_unsupported_reason(server_args: ServerArgs) -> s...; symbols: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunks: -27,6 +27,7 @@ class GenerationBatchResult:; symbols: GenerationBatchResult
- Key code excerpts:

```diff
diff -- python/sglang/srt/speculative/eagle_worker_v2.py
@@ -30,8 +30,13 @@
+from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
+from sglang.srt.speculative.adaptive_runtime_state import (
+    AdaptiveController,
+    SpecRuntimeState,
+)
@@ -671,6 +676,13 @@ def __init__(
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -114,14 +114,18 @@ def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
+        current_kv_lens_cpu = batch.seq_lens.to(device="cpu")
-        for r in batch.reqs:
-            # Over-allocation happens here
-            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
+        for i, r in enumerate(batch.reqs):
+            cur_kv_len = current_kv_lens_cpu[i].item()
diff -- python/sglang/srt/managers/scheduler_output_processor_mixin.py
@@ -358,8 +358,17 @@ def _resolve_spec_overlap_token_ids(
```

- Reviewed files:
  - runtime: `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0; `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4; `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1; `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5; `python/sglang/srt/managers/utils.py` modified +1/-0; `python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/scheduler_output_processor_mixin.py`, `python/sglang/srt/managers/utils.py`, `python/sglang/srt/speculative/adaptive_spec_params.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22933 - [CPU] expand the interface of shared_expert without scaling factor

- Link: https://github.com/sgl-project/sglang/pull/22933
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +313/-623, 1252 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] expand the interface of shared_expert without scaling factor"; model line: DeepSeek V3/R1; category: model implementation change; main diff: `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp`; PR body summary: This patch expands `shared_expert` kernel when scaling factor is None, which maps exactly the behavior of MoE when expert equals to one. Also simplifies moe kernel source codes....
- Key implementation: `sgl-kernel/csrc/cpu/moe_int4.cpp` modified +10/-176 (186 lines); hunks: -1,185 +1,19; `sgl-kernel/csrc/cpu/moe.h` added +173/-0 (173 lines); hunks: -0,0 +1,173; `sgl-kernel/csrc/cpu/moe.cpp` modified +25/-119 (144 lines); hunks: -1,6 +1,7; -25,112 +26,6 @@ namespace {; `sgl-kernel/csrc/cpu/moe_fp8.cpp` modified +6/-136 (142 lines); hunks: -1,139 +1,6; -372,6 +239,7 @@ void shared_expert_fp8_kernel_impl(.
- Code diff details:
  - `sgl-kernel/csrc/cpu/moe_int4.cpp` modified +10/-176 (186 lines); hunks: -1,185 +1,19
  - `sgl-kernel/csrc/cpu/moe.h` added +173/-0 (173 lines); hunks: -0,0 +1,173
  - `sgl-kernel/csrc/cpu/moe.cpp` modified +25/-119 (144 lines); hunks: -1,6 +1,7; -25,112 +26,6 @@ namespace {
  - `sgl-kernel/csrc/cpu/moe_fp8.cpp` modified +6/-136 (142 lines); hunks: -1,139 +1,6; -372,6 +239,7 @@ void shared_expert_fp8_kernel_impl(
  - `sgl-kernel/csrc/cpu/moe_int8.cpp` modified +6/-108 (114 lines); hunks: -1,114 +1,9; -885,6 +780,7 @@ void shared_expert_int8_kernel_impl(
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/cpu/moe_int4.cpp
@@ -1,185 +1,19 @@
-#include "vec.h"
-namespace {
+#include "moe.h"
-template <typename scalar_t>
-inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
-  using Vec = at::vec::Vectorized<scalar_t>;
diff -- sgl-kernel/csrc/cpu/moe.h
@@ -0,0 +1,173 @@
+#pragma once
+#include "vec.h"
+template <typename scalar_t>
+inline void fill_stub(scalar_t* __restrict__ out, scalar_t val, int64_t size) {
+  using Vec = at::vec::Vectorized<scalar_t>;
+  const Vec data_vec(val);
diff -- sgl-kernel/csrc/cpu/moe.cpp
@@ -1,6 +1,7 @@
```

- Reviewed files:
  - other: `sgl-kernel/csrc/cpu/moe_int4.cpp` modified +10/-176; `sgl-kernel/csrc/cpu/moe.h` added +173/-0; `sgl-kernel/csrc/cpu/moe.cpp` modified +25/-119; `sgl-kernel/csrc/cpu/moe_fp8.cpp` modified +6/-136; `sgl-kernel/csrc/cpu/moe_int8.cpp` modified +6/-108
  - tests: `test/srt/cpu/test_shared_expert.py` modified +64/-50; `test/srt/cpu/test_moe.py` modified +8/-27; `test/srt/cpu/utils.py` modified +18/-4
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_moe.py`, `test/srt/cpu/test_shared_expert.py`, `test/srt/cpu/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22774 - [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)

- Link: https://github.com/sgl-project/sglang/pull/22774
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_common/utils.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `b35213be11c7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +184/-44, 795 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)"; model line: DeepSeek V3/R1; category: model support/runtime entry; main diff: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; PR body summary: Enable SGLang to run DeepSeek models on Moore Threads MUSA GPUs. This PR adds MUSA backend support across the inference stack, including layers, quantization, MoE, attention, sp....
- Key implementation: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3 (17 lines); hunks: -1,6 +1,6; -14,10 +14,12; symbols: execute, deep_gemm_execution_hook, _deep_gemm_execution_hook, touching `execute, deep_gemm_execution_hook, _deep_gemm_execution_hook`; `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -141,6 +141,7; -182,6 +183,8; symbols: forward_normal_dual_stream, _post_combine_hook, __init__, determine_num_fused_shared_experts, touching `forward_normal_dual_stream, _post_combine_hook, __init__`; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +10/-1 (11 lines); hunks: -49,6 +49,7; -498,7 +499,7 @@ def post_load_weights(; symbols: post_load_weights, touching `post_load_weights`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -553,6 +554,11 @@ def forward_absorb_core(; symbols: forward_absorb_core, touching `forward_absorb_core`.
- Code diff details:
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3 (17 lines); hunks: -1,6 +1,6; -14,10 +14,12; symbols: execute, deep_gemm_execution_hook, _deep_gemm_execution_hook
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunks: -141,6 +141,7; -182,6 +183,8; symbols: forward_normal_dual_stream, _post_combine_hook, __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +10/-1 (11 lines); hunks: -49,6 +49,7; -498,7 +499,7 @@ def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +6/-0 (6 lines); hunks: -21,6 +21,7; -553,6 +554,11 @@ def forward_absorb_core(; symbols: forward_absorb_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +2/-1 (3 lines); hunks: -13,6 +13,7; -491,7 +492,7 @@ def _concat_and_cast_mha_k(; symbols: _concat_and_cast_mha_k
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py
@@ -1,6 +1,6 @@
-from contextlib import contextmanager
+from contextlib import contextmanager, nullcontext
@@ -14,10 +14,12 @@
-from sglang.srt.utils import ceil_div, get_available_gpu_memory
+from sglang.srt.utils import ceil_div, get_available_gpu_memory, is_musa
+_is_musa = is_musa()
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -141,6 +141,7 @@
+    _is_musa,
@@ -182,6 +183,8 @@
+elif _is_musa:
+    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
@@ -640,7 +643,9 @@ def forward_normal_dual_stream(
-            if not _is_cuda or isinstance(self.experts.quant_method, KTEPWrapperMethod):
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -49,6 +49,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +14/-3; `python/sglang/srt/models/deepseek_v2.py` modified +13/-4; `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +10/-1; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +6/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +2/-1; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/activation.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
