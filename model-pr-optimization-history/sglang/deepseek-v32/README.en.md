# sglang DeepSeek V3.2 Model PR Optimization History

## Scope

- Rebuilt on: 2026-04-25
- Source baseline: `sgl-project/sglang` trace worktree commit `880599cd43`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.
- Diffusion model families have been removed from this history set and are no longer part of model optimization skills.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs/basic_usage/deepseek_v32.md` | [#11877](https://github.com/sgl-project/sglang/pull/11877), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12130](https://github.com/sgl-project/sglang/pull/12130), [#12138](https://github.com/sgl-project/sglang/pull/12138), [#12296](https://github.com/sgl-project/sglang/pull/12296), [#12868](https://github.com/sgl-project/sglang/pull/12868), [#13459](https://github.com/sgl-project/sglang/pull/13459), [#13646](https://github.com/sgl-project/sglang/pull/13646), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14321](https://github.com/sgl-project/sglang/pull/14321), [#14336](https://github.com/sgl-project/sglang/pull/14336), [#14372](https://github.com/sgl-project/sglang/pull/14372), ... (22 total) |
| `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` | [#11877](https://github.com/sgl-project/sglang/pull/11877) |
| `docs_new/docs/basic_usage/deepseek_v32.mdx` | no direct PR-number commit |
| `docs_new/docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.mdx` | no direct PR-number commit |
| `examples/chat_template/tool_chat_template_deepseekv32.jinja` | [#11063](https://github.com/sgl-project/sglang/pull/11063) |
| `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` | [#19148](https://github.com/sgl-project/sglang/pull/19148) |
| `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` | [#14249](https://github.com/sgl-project/sglang/pull/14249), [#14353](https://github.com/sgl-project/sglang/pull/14353) |
| `python/sglang/srt/function_call/deepseekv32_detector.py` | [#14249](https://github.com/sgl-project/sglang/pull/14249), [#14573](https://github.com/sgl-project/sglang/pull/14573), [#14750](https://github.com/sgl-project/sglang/pull/14750), [#15278](https://github.com/sgl-project/sglang/pull/15278), [#16091](https://github.com/sgl-project/sglang/pull/16091), [#18174](https://github.com/sgl-project/sglang/pull/18174) |
| `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` | [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541), [#14572](https://github.com/sgl-project/sglang/pull/14572), [#15381](https://github.com/sgl-project/sglang/pull/15381), [#17007](https://github.com/sgl-project/sglang/pull/17007), [#19428](https://github.com/sgl-project/sglang/pull/19428) |
| `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#15086](https://github.com/sgl-project/sglang/pull/15086), [#15938](https://github.com/sgl-project/sglang/pull/15938) |
| `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#12520](https://github.com/sgl-project/sglang/pull/12520), [#13812](https://github.com/sgl-project/sglang/pull/13812), [#16841](https://github.com/sgl-project/sglang/pull/16841), [#18280](https://github.com/sgl-project/sglang/pull/18280), [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` | [#14781](https://github.com/sgl-project/sglang/pull/14781), [#17554](https://github.com/sgl-project/sglang/pull/17554) |
| `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11450](https://github.com/sgl-project/sglang/pull/11450), [#11565](https://github.com/sgl-project/sglang/pull/11565), [#11652](https://github.com/sgl-project/sglang/pull/11652), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12044](https://github.com/sgl-project/sglang/pull/12044), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12094](https://github.com/sgl-project/sglang/pull/12094), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12816](https://github.com/sgl-project/sglang/pull/12816), [#13236](https://github.com/sgl-project/sglang/pull/13236), ... (46 total) |
| `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` | [#17554](https://github.com/sgl-project/sglang/pull/17554) |
| `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#15938](https://github.com/sgl-project/sglang/pull/15938) |
| `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#16841](https://github.com/sgl-project/sglang/pull/16841), [#18488](https://github.com/sgl-project/sglang/pull/18488), [#19945](https://github.com/sgl-project/sglang/pull/19945), [#21511](https://github.com/sgl-project/sglang/pull/21511) |
| `python/sglang/srt/layers/attention/nsa/transform_index.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#12300](https://github.com/sgl-project/sglang/pull/12300) |
| `python/sglang/srt/layers/attention/nsa/triton_kernel.py` | [#11450](https://github.com/sgl-project/sglang/pull/11450), [#18526](https://github.com/sgl-project/sglang/pull/18526) |
| `python/sglang/srt/layers/attention/nsa/utils.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11682](https://github.com/sgl-project/sglang/pull/11682), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541), [#14781](https://github.com/sgl-project/sglang/pull/14781), [#15938](https://github.com/sgl-project/sglang/pull/15938), [#17076](https://github.com/sgl-project/sglang/pull/17076), [#19134](https://github.com/sgl-project/sglang/pull/19134), [#19829](https://github.com/sgl-project/sglang/pull/19829), [#22914](https://github.com/sgl-project/sglang/pull/22914) |
| `python/sglang/srt/layers/attention/nsa_backend.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11652](https://github.com/sgl-project/sglang/pull/11652), [#11655](https://github.com/sgl-project/sglang/pull/11655), [#11876](https://github.com/sgl-project/sglang/pull/11876), [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12215](https://github.com/sgl-project/sglang/pull/12215), [#12294](https://github.com/sgl-project/sglang/pull/12294), [#12583](https://github.com/sgl-project/sglang/pull/12583), [#12788](https://github.com/sgl-project/sglang/pull/12788), [#12964](https://github.com/sgl-project/sglang/pull/12964), [#13022](https://github.com/sgl-project/sglang/pull/13022), ... (39 total) |
| `python/sglang/srt/layers/communicator_nsa_cp.py` | [#12065](https://github.com/sgl-project/sglang/pull/12065), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14541](https://github.com/sgl-project/sglang/pull/14541) |
| `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` | [#18931](https://github.com/sgl-project/sglang/pull/18931) |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` | [#21405](https://github.com/sgl-project/sglang/pull/21405), [#21511](https://github.com/sgl-project/sglang/pull/21511) |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_common/utils.py` | no direct PR-number commit |
| `python/sglang/srt/models/deepseek_v2.py` | [#11061](https://github.com/sgl-project/sglang/pull/11061), [#11510](https://github.com/sgl-project/sglang/pull/11510), [#11892](https://github.com/sgl-project/sglang/pull/11892), [#12065](https://github.com/sgl-project/sglang/pull/12065), [#12094](https://github.com/sgl-project/sglang/pull/12094), [#12788](https://github.com/sgl-project/sglang/pull/12788), [#12816](https://github.com/sgl-project/sglang/pull/12816), [#12964](https://github.com/sgl-project/sglang/pull/12964), [#13459](https://github.com/sgl-project/sglang/pull/13459), [#13544](https://github.com/sgl-project/sglang/pull/13544), [#13959](https://github.com/sgl-project/sglang/pull/13959), [#14572](https://github.com/sgl-project/sglang/pull/14572), ... (22 total) |
| `test/manual/layers/attention/nsa/test_act_quant_triton.py` | no direct PR-number commit |
| `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` | [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `test/manual/layers/attention/nsa/test_index_buf_accessor.py` | [#13812](https://github.com/sgl-project/sglang/pull/13812), [#19319](https://github.com/sgl-project/sglang/pull/19319) |
| `test/manual/nightly/test_deepseek_v32_perf.py` | [#13646](https://github.com/sgl-project/sglang/pull/13646), [#21192](https://github.com/sgl-project/sglang/pull/21192) |
| `test/registered/8-gpu-models/test_deepseek_v32.py` | [#17951](https://github.com/sgl-project/sglang/pull/17951) |
| `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` | [#21405](https://github.com/sgl-project/sglang/pull/21405) |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v32_dp_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v32_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v32_mtp_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi30x/test_deepseek_v32_tc_eval_amd.py` | no direct PR-number commit |
| `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py` | [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` | [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` | [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/perf/mi30x/test_deepseek_v32_basic_perf_amd.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi30x/test_deepseek_v32_mtp_perf_amd.py` | no direct PR-number commit |
| `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` | [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` | [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17523](https://github.com/sgl-project/sglang/pull/17523) |
| `test/registered/amd/test_deepseek_v32_basic.py` | [#16934](https://github.com/sgl-project/sglang/pull/16934), [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17432](https://github.com/sgl-project/sglang/pull/17432), [#17633](https://github.com/sgl-project/sglang/pull/17633) |
| `test/registered/amd/test_deepseek_v32_mtp.py` | [#16934](https://github.com/sgl-project/sglang/pull/16934), [#17179](https://github.com/sgl-project/sglang/pull/17179), [#17432](https://github.com/sgl-project/sglang/pull/17432), [#17633](https://github.com/sgl-project/sglang/pull/17633) |
| `test/registered/cp/test_deepseek_v32_cp_single_node.py` | [#21192](https://github.com/sgl-project/sglang/pull/21192), [#21585](https://github.com/sgl-project/sglang/pull/21585) |
| `test/registered/gb300/test_deepseek_v32.py` | no direct PR-number commit |
| `test/registered/gb300/test_deepseek_v32_nvfp4.py` | no direct PR-number commit |
| `test/registered/kernels/test_nsa_indexer.py` | [#17076](https://github.com/sgl-project/sglang/pull/17076), [#17452](https://github.com/sgl-project/sglang/pull/17452), [#17682](https://github.com/sgl-project/sglang/pull/17682), [#18280](https://github.com/sgl-project/sglang/pull/18280), [#18389](https://github.com/sgl-project/sglang/pull/18389), [#19041](https://github.com/sgl-project/sglang/pull/19041) |
| `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` | [#19985](https://github.com/sgl-project/sglang/pull/19985), [#20062](https://github.com/sgl-project/sglang/pull/20062), [#20984](https://github.com/sgl-project/sglang/pull/20984), [#21003](https://github.com/sgl-project/sglang/pull/21003) |
| `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` | [#19985](https://github.com/sgl-project/sglang/pull/19985), [#20062](https://github.com/sgl-project/sglang/pull/20062), [#20984](https://github.com/sgl-project/sglang/pull/20984), [#21003](https://github.com/sgl-project/sglang/pull/21003) |
| `test/registered/unit/mem_cache/test_nsa_pool_host_unit.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 119
- Extra PRs preserved from existing docs: 116
- Total PRs in this document: 234
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA. | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` |
| 2025-10-05 | [#11063](https://github.com/sgl-project/sglang/pull/11063) | merged | Add DeepSeek-V3.2 Tool Call Template | `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2025-10-05 | [#11194](https://github.com/sgl-project/sglang/pull/11194) | merged | [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2 | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-06 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | Support DeepSeek V3.2 Exp | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-09 | [#11309](https://github.com/sgl-project/sglang/pull/11309) | merged | [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/server_args.py` |
| 2025-10-11 | [#11450](https://github.com/sgl-project/sglang/pull/11450) | merged | [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton | `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-13 | [#11557](https://github.com/sgl-project/sglang/pull/11557) | merged | Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3)) | `python/sglang/srt/server_args.py` |
| 2025-10-13 | [#11308](https://github.com/sgl-project/sglang/pull/11308) | merged | [CI] Add Basic Test for DeepSeek V3.2 | `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh` |
| 2025-10-14 | [#11565](https://github.com/sgl-project/sglang/pull/11565) | merged | [DSv32] Use torch.compile for _get_logits_head_gate | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-15 | [#11596](https://github.com/sgl-project/sglang/pull/11596) | closed | [Spec Decoding] Support MTP for dsv3.2 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-16 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py` |
| 2025-10-16 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-17 | [#11109](https://github.com/sgl-project/sglang/pull/11109) | closed | [Draft] Support MTP for DeepSeek-V3.2 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py` |
| 2025-10-17 | [#11682](https://github.com/sgl-project/sglang/pull/11682) | merged | Cleaning indexer for DeepSeek V3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-10-19 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | [Spec Decoding] Support MTP for dsv3.2 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-20 | [#11835](https://github.com/sgl-project/sglang/pull/11835) | merged | [CI] Add CI test for DeepSeek V3.2 MTP | `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py` |
| 2025-10-20 | [#11815](https://github.com/sgl-project/sglang/pull/11815) | merged | [DeepseekV32] Add fast_topk_transform_ragged_fused kernel | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-21 | [#11876](https://github.com/sgl-project/sglang/pull/11876) | merged | Rename flashmla kernel options of nsa backend for better readability | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-10-23 | [#11761](https://github.com/sgl-project/sglang/pull/11761) | closed | (beta)support context parallel with deepseekv3.2-DSA |  |
| 2025-10-24 | [#12017](https://github.com/sgl-project/sglang/pull/12017) | closed | (beta)support context parallel with deepseekv3.2-DSA | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-10-25 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | [Doc] Add documentation for DeepSeek V3.2 | `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md` |
| 2025-10-25 | [#12052](https://github.com/sgl-project/sglang/pull/12052) | closed | Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models | `python/sglang/srt/layers/dp_attention.py` |
| 2025-10-25 | [#12130](https://github.com/sgl-project/sglang/pull/12130) | merged | [Doc] Fix format for deepseek v3.2 document | `docs/basic_usage/deepseek_v32.md` |
| 2025-10-26 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | [Test] Add dsv3.2 nsa backend testing | `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py` |
| 2025-10-26 | [#12138](https://github.com/sgl-project/sglang/pull/12138) | merged | [Doc] Small update of DeepSeek v3.2 document | `docs/basic_usage/deepseek_v32.md` |
| 2025-10-28 | [#11655](https://github.com/sgl-project/sglang/pull/11655) | merged | [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | Update deepseek_v32.md | `docs/basic_usage/deepseek_v32.md` |
| 2025-10-29 | [#12294](https://github.com/sgl-project/sglang/pull/12294) | merged | [Deepseek V3.2] Enable flashmla_auto with MTP | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-10-30 | [#12094](https://github.com/sgl-project/sglang/pull/12094) | merged | Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-30 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-10-31 | [#12300](https://github.com/sgl-project/sglang/pull/12300) | merged | [DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type | `python/sglang/srt/layers/attention/nsa/transform_index.py` |
| 2025-11-04 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | Enable mixed type LayerNorm kernel for NSA indexer | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-04 | [#12645](https://github.com/sgl-project/sglang/pull/12645) | merged | [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2 | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-06 | [#11892](https://github.com/sgl-project/sglang/pull/11892) | merged | DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-07 | [#12788](https://github.com/sgl-project/sglang/pull/12788) | merged | [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-07 | [#12520](https://github.com/sgl-project/sglang/pull/12520) | merged | [Test] Add DeepSeekV3.2 NSA Indexer Test Suite | `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2025-11-07 | [#12820](https://github.com/sgl-project/sglang/pull/12820) | open | [WIP][Feature] support tp-sp on qwen2/3 & deepseek v2/3/3.2 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-11-07 | [#12816](https://github.com/sgl-project/sglang/pull/12816) | merged | [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-08 | [#12582](https://github.com/sgl-project/sglang/pull/12582) | merged | [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel | `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill | `docs/basic_usage/deepseek_v32.md` |
| 2025-11-12 | [#12583](https://github.com/sgl-project/sglang/pull/12583) | merged | [Deepseek V3.2] Fix accuracy bug in the Indexer | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-12 | [#12215](https://github.com/sgl-project/sglang/pull/12215) | merged | [DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat` | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-14 | [#13236](https://github.com/sgl-project/sglang/pull/13236) | merged | [Deepseek V3.2] Clean up MTP | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-17 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | (1/n)support context parallel with deepseekv3.2-DSA | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-17 | [#13022](https://github.com/sgl-project/sglang/pull/13022) | merged | [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-18 | [#13531](https://github.com/sgl-project/sglang/pull/13531) | closed | DeepSeek V3.2 indexer RoPE fix | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-20 | [#12964](https://github.com/sgl-project/sglang/pull/12964) | merged | [DeepseekV3.2] Deepseek fp8 support for MHA path | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-20 | [#13459](https://github.com/sgl-project/sglang/pull/13459) | merged | [Deepseek V3.2] Change indexer weights_proj to fp32 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-11-25 | [#13544](https://github.com/sgl-project/sglang/pull/13544) | merged | [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-26 | [#13958](https://github.com/sgl-project/sglang/pull/13958) | merged | Fix nightly test failures: NSA indexer dtype and CPP radix cache init | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-26 | [#14015](https://github.com/sgl-project/sglang/pull/14015) | merged | Revert "Fix nightly test failures: NSA indexer dtype and CPP radix cache init" | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-30 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | [DeepSeekV3.2] Enable pure TP & Partial DP Attention | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/manual/nightly/test_deepseek_v32_perf.py` |
| 2025-12-01 | [#14245](https://github.com/sgl-project/sglang/pull/14245) | merged | Fix NSA Bug in Centralize NSA Dispatch Logic | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-02 | [#14249](https://github.com/sgl-project/sglang/pull/14249) | merged | feat: DeepSeek new v3.2 encoding | `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-03 | [#14321](https://github.com/sgl-project/sglang/pull/14321) | merged | [Doc] Update DeepSeek-V3.2 document | `docs/basic_usage/deepseek_v32.md` |
| 2025-12-03 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels | `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-03 | [#14332](https://github.com/sgl-project/sglang/pull/14332) | open | feat: V32 tool call parsing for no-dsml tag | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-03 | [#14336](https://github.com/sgl-project/sglang/pull/14336) | merged | [Doc] Fix DeepSeek V32 Doc | `docs/basic_usage/deepseek_v32.md` |
| 2025-12-03 | [#14372](https://github.com/sgl-project/sglang/pull/14372) | merged | [Tiny]Small fixes in deepseek v32 doc | `docs/basic_usage/deepseek_v32.md` |
| 2025-12-04 | [#14325](https://github.com/sgl-project/sglang/pull/14325) | merged | [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-06 | [#14524](https://github.com/sgl-project/sglang/pull/14524) | open | [Test] Add test suite for NSA backend | `python/sglang/test/attention/test_nsa_backend.py` |
| 2025-12-08 | [#14573](https://github.com/sgl-project/sglang/pull/14573) | merged | [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-11 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | [NPU]dsv3.2 cp for npu | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py` |
| 2025-12-11 | [#14307](https://github.com/sgl-project/sglang/pull/14307) | merged | [SMG][DS32][fix] support dsv32, add role developer | `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs` |
| 2025-12-11 | [#14304](https://github.com/sgl-project/sglang/pull/14304) | merged | [FIX][DS32]openai protocol: support openai message role: developer | `python/sglang/srt/entrypoints/openai/protocol.py` |
| 2025-12-12 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | [NPU] optimization for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2025-12-13 | [#14904](https://github.com/sgl-project/sglang/pull/14904) | closed | [DeepSeek V3.2] Proper drop_thinking logic | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-13 | [#15064](https://github.com/sgl-project/sglang/pull/15064) | merged | fix: dpskv32 chat history processing, default drop_thinking to true | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-15 | [#15086](https://github.com/sgl-project/sglang/pull/15086) | merged | [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` |
| 2025-12-16 | [#15051](https://github.com/sgl-project/sglang/pull/15051) | closed | feat(ds32): support tag for deepseek 3.2 tool call | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-16 | [#15217](https://github.com/sgl-project/sglang/pull/15217) | closed | fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-16 | [#15242](https://github.com/sgl-project/sglang/pull/15242) | merged | [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations | `sgl-kernel/cmake/flashmla.cmake` |
| 2025-12-17 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | [DeepSeekV3.2] Add pure TP+MTP test | `docs/basic_usage/deepseek_v32.md` |
| 2025-12-17 | [#15322](https://github.com/sgl-project/sglang/pull/15322) | open | dsv32 support o_proj tp | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/linear.py` |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | [Deepseek V3.2] Support Overlap Spec + NSA | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-12-18 | [#15278](https://github.com/sgl-project/sglang/pull/15278) | merged | feat: DeepSeek-V3.2 Streaming tool call output | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-18 | [#14781](https://github.com/sgl-project/sglang/pull/14781) | merged | [Performance] optimize NSA backend metadata computation for multi-step speculative decoding | `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-19 | [#14353](https://github.com/sgl-project/sglang/pull/14353) | merged | feat(dsv32): better error handling for DeepSeek-v3.2 encoder | `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` |
| 2025-12-19 | [#15429](https://github.com/sgl-project/sglang/pull/15429) | merged | [Deepseek V3.2] Fix Deepseek MTP in V1 mode | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-19 | [#15040](https://github.com/sgl-project/sglang/pull/15040) | merged | [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-21 | [#14901](https://github.com/sgl-project/sglang/pull/14901) | merged | fix ds3.2 nsa backend prefill TBO | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-25 | [#14741](https://github.com/sgl-project/sglang/pull/14741) | merged | [1/N][Sparse With Hicache]: Add Sparse Interface | `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` |
| 2025-12-26 | [#14750](https://github.com/sgl-project/sglang/pull/14750) | merged | [Tool Call][DSV32] Streamline function call parameters | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | [cp] bug fix for dsv3.2 cp | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2025-12-30 | [#16148](https://github.com/sgl-project/sglang/pull/16148) | open | [Fix] DSV3.2 W4AFP8 MTP use FP8 draft model | `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-12-31 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | [cp] assert dsv3.2 cp in pd decode mode | `python/sglang/srt/server_args.py` |
| 2026-01-02 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | Multiple updates of DeepSeek V32 and context parallel | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-06 | [#15310](https://github.com/sgl-project/sglang/pull/15310) | closed | [Deepseek V3.2] Enable TRTLLM Allreduce Fusion | `python/sglang/srt/server_args.py` |
| 2026-01-06 | [#16520](https://github.com/sgl-project/sglang/pull/16520) | merged | fix: unimplemented methods in BaseIndexerMetadata | `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-07 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | Clean Some Environment Variables for DeepSeek V32 | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2026-01-08 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` |
| 2026-01-09 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-01-10 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | [DSv32] Overlap indexer weights_proj during dual_stream decode | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-11 | [#16881](https://github.com/sgl-project/sglang/pull/16881) | closed | [DSv32] Add returning DSA topk indices | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | add doc for dsv32 cp+pp | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | [Ascend] fix dsv3.2 weight cast bug | `python/sglang/srt/layers/quantization/unquant.py` |
| 2026-01-14 | [#16841](https://github.com/sgl-project/sglang/pull/16841) | merged | [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2026-01-14 | [#17054](https://github.com/sgl-project/sglang/pull/17054) | merged | Update deepseekV32 Cp doc | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-16 | [#17185](https://github.com/sgl-project/sglang/pull/17185) | open | [DeepSeek V3.2] [Feat] add tensor parallel o_proj linear in context parallel nsa | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/server_args.py` |
| 2026-01-16 | [#16934](https://github.com/sgl-project/sglang/pull/16934) | merged | [AMD] Enable DeepseekV3.2 test for AMD CI | `test/registered/amd/test_deepseek_v32_mtp.py`, `test/registered/amd/test_deepseek_v32_basic.py` |
| 2026-01-16 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2026-01-19 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-20 | [#17179](https://github.com/sgl-project/sglang/pull/17179) | merged | [AMD] Add DeepSeek-V3.2 and VLMs model in nightly tests | `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` |
| 2026-01-20 | [#17409](https://github.com/sgl-project/sglang/pull/17409) | merged | [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-01-20 | [#17205](https://github.com/sgl-project/sglang/pull/17205) | merged | [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-21 | [#17452](https://github.com/sgl-project/sglang/pull/17452) | merged | Fix NSA indexer in the nightly test | `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-22 | [#17518](https://github.com/sgl-project/sglang/pull/17518) | merged | [HotFix]Fix dtype mismatch in nsa indexer on AMD device | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-22 | [#17432](https://github.com/sgl-project/sglang/pull/17432) | merged | [AMD] fix amd ci dpskv32 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py` |
| 2026-01-23 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | [NPU]bugfix: fix for dsv3.2 and dsvl2 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-01-23 | [#16758](https://github.com/sgl-project/sglang/pull/16758) | merged | [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-24 | [#17682](https://github.com/sgl-project/sglang/pull/17682) | merged | Fix NSA indexer test and move it to pre commit test | `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-25 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-26 | [#17761](https://github.com/sgl-project/sglang/pull/17761) | open | fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates | `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2026-01-26 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | [NPU]DeepSeek-V3.2 support npu mlaprolog | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-01-27 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-27 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-28 | [#17633](https://github.com/sgl-project/sglang/pull/17633) | merged | [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites | `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py` |
| 2026-01-28 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | [DSv32] Overlap indexer qk projection and activation quant | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-28 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` |
| 2026-01-29 | [#17951](https://github.com/sgl-project/sglang/pull/17951) | merged | Add tool call tests for DeepSeek V3.2 in nightly CI | `test/registered/8-gpu-models/test_deepseek_v32.py` |
| 2026-02-02 | [#18094](https://github.com/sgl-project/sglang/pull/18094) | open | support deepseekv3.2-piecewise-cuda-graph | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-02-02 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-02 | [#17964](https://github.com/sgl-project/sglang/pull/17964) | merged | [NPU] support dsv32 radixcache on ascend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-03 | [#18167](https://github.com/sgl-project/sglang/pull/18167) | open | [Feature] Add DCP support for DeepSeek v3.2 | `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-05 | [#18275](https://github.com/sgl-project/sglang/pull/18275) | open | [NPU] allgather after qlora for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-10 | [#18297](https://github.com/sgl-project/sglang/pull/18297) | merged | Deepseekv32 compatibility with transformers v5 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-10 | [#18542](https://github.com/sgl-project/sglang/pull/18542) | open | fix: fixed aux hidden state index out of range when using eagle3 with nsa cp | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-10 | [#18488](https://github.com/sgl-project/sglang/pull/18488) | merged | Tilelang sparse decode fwd for dsv32 mi355 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-02-11 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | Fix Bug on dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | [V3.2] Change default CP token split method to `--round-robin-split` | `docs/basic_usage/deepseek_v32.md` |
| 2026-02-12 | [#18733](https://github.com/sgl-project/sglang/pull/18733) | open | Add DeepSeek V32 PD disaggregation test | `test/registered/distributed/test_disaggregation_deepseek_v32.py` |
| 2026-02-14 | [#17554](https://github.com/sgl-project/sglang/pull/17554) | merged | Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels | `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` |
| 2026-02-14 | [#18126](https://github.com/sgl-project/sglang/pull/18126) | merged | Fix dsv32 encode_messages | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/jinja_template_utils.py` |
| 2026-02-15 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | Fix model loading for DeepSeek-V3.2-AWQ | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-16 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md` |
| 2026-02-19 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | [AMD] Fix mi35x dsv32 mtp nightly | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-20 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | Fix NSA FP8 KV cache path for both-trtllm MHA one-shot | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-02-21 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | [DSv32] Fix MTP and CP compatability | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-22 | [#19134](https://github.com/sgl-project/sglang/pull/19134) | merged | Fix spec v2+dp attention in nsa backend | `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2026-02-22 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj` | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-02-24 | [#19211](https://github.com/sgl-project/sglang/pull/19211) | open | [Refactor][DeepSeek-V3.2] Extract V3.2/NSA logic into `DeepseekV32Mixin` | `python/sglang/srt/models/deepseek_common/v32_mixin.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-25 | [#19299](https://github.com/sgl-project/sglang/pull/19299) | open | [Perf] O(1) expert weight matching in DeepSeek weight loader | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `test/unit/test_deepseek_weight_loader.py` |
| 2026-02-26 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` |
| 2026-02-26 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | Fix NSA CP positions mismatch in eagle NextN model | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-26 | [#17199](https://github.com/sgl-project/sglang/pull/17199) | closed | [Feature] add feature mla_ag_after_qlora for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py` |
| 2026-02-27 | [#18319](https://github.com/sgl-project/sglang/pull/18319) | merged | [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-27 | [#18526](https://github.com/sgl-project/sglang/pull/18526) | merged | [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr… | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py` |
| 2026-02-27 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-03-01 | [#19609](https://github.com/sgl-project/sglang/pull/19609) | open | TP indexer weight in NSA attention | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-01 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | [Perf] Optimize NSA backend metadata under MTP | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-01 | [#17647](https://github.com/sgl-project/sglang/pull/17647) | closed | [Perf] opt nsa backend init forward metada | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-03-02 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | [Feature] add feature mla_ag_after_qlora for dsv3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-03-03 | [#18174](https://github.com/sgl-project/sglang/pull/18174) | merged | [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-03-03 | [#16091](https://github.com/sgl-project/sglang/pull/16091) | merged | [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format. | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-03-04 | [#19829](https://github.com/sgl-project/sglang/pull/19829) | merged | [NSA] Fix line-too-long lint in `can_nsa_prefill_cp_round_robin_split` | `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2026-03-05 | [#19975](https://github.com/sgl-project/sglang/pull/19975) | open | [AMD] Support context parallel for DeepSeek-V3.2 on AMD GPUs and add its test to nightly CI | `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-05 | [#19987](https://github.com/sgl-project/sglang/pull/19987) | closed | [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-06 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | [FIX] NSA backend page_table overflow in speculative decoding target_verify | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-06 | [#19985](https://github.com/sgl-project/sglang/pull/19985) | merged | [V32] Enhance deepseek v32 related tests | `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | [V32/GLM5] Change default setting of V32 nvfp4 on TP4 | `python/sglang/srt/server_args.py` |
| 2026-03-09 | [#20062](https://github.com/sgl-project/sglang/pull/20062) | merged | [V32/GLM5] Control the threshold of applying dense attention with an environ | `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` |
| 2026-03-10 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | Add DeepSeek3.2 and GlmMoeDsa into moe tune | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-03-11 | [#20360](https://github.com/sgl-project/sglang/pull/20360) | open | [AMD][Bug fix] Fix NSA context parallelism (round-robin-split) producing garbage output | `python/sglang/srt/layers/communicator_nsa_cp.py` |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | [Doc] Add DSA/NSA attention backend to support matrix | `docs/advanced_features/attention_backend.md` |
| 2026-03-11 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | [deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug | `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py` |
| 2026-03-13 | [#20531](https://github.com/sgl-project/sglang/pull/20531) | open | [bugfix] Fix NSA indexer ragged gather batch-view mismatch in CP round-robin split | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-13 | [#20534](https://github.com/sgl-project/sglang/pull/20534) | open | Transfer FP8 K/K_scale for CP indexer prefill gather | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-17 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2026-03-18 | [#20809](https://github.com/sgl-project/sglang/pull/20809) | open | [Bugfix] Add DeepseekV32ForCausalLM to MTP draft model mapping | `python/sglang/srt/configs/model_config.py` |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | [AMD] Fix dpsk-v32 accuracy issue on mi355 | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-03-18 | [#20880](https://github.com/sgl-project/sglang/pull/20880) | open | Reject HiCache L3 storage backend for NSA models at init time | `python/sglang/srt/mem_cache/hiradix_cache.py` |
| 2026-03-19 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-19 | [#17024](https://github.com/sgl-project/sglang/pull/17024) | closed | [PD] Fix DeepSeek V3.2 indexer cache transfer | `python/sglang/srt/disaggregation/prefill.py` |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | Fix DeepSeek V32 FP4 test | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | Revert "Fix DeepSeek V32 FP4 test" | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` |
| 2026-03-23 | [#21179](https://github.com/sgl-project/sglang/pull/21179) | open | [Bug] Preserve DeepSeek-V3.2 tool-call markers in reasoning parsing | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-03-23 | [#20343](https://github.com/sgl-project/sglang/pull/20343) | merged | HiSparse for Sparse Attention | `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-23 | [#21194](https://github.com/sgl-project/sglang/pull/21194) | open | [bugfix][AMD] Fix PPMissingLayer AttributeError for deepseek v2/v3 in aiter_gfx95 code path | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-23 | [#15807](https://github.com/sgl-project/sglang/pull/15807) | closed | [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side. | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-23 | [#14619](https://github.com/sgl-project/sglang/pull/14619) | closed | [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32. | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | Fix CP in-seq-split method for DeepSeek V32 and update related tests | `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `test/manual/nightly/test_deepseek_v32_perf.py` |
| 2026-03-24 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2 | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-24 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-03-25 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | Workaround of DSA performance drop on B200 + DP | `python/sglang/srt/server_args.py` |
| 2026-03-25 | [#16079](https://github.com/sgl-project/sglang/pull/16079) | closed | [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2 | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-26 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-27 | [#21506](https://github.com/sgl-project/sglang/pull/21506) | open | [WIP][NPU] DeepSeek-V3.2 adapt enable-torch-compile | `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-27 | [#21529](https://github.com/sgl-project/sglang/pull/21529) | open | Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-27 | [#21530](https://github.com/sgl-project/sglang/pull/21530) | open | [ROCm] Fix fused MLA decode rope path for Kimi K2.5 and DeepSeek-variant models | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`, `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py` |
| 2026-03-27 | [#21546](https://github.com/sgl-project/sglang/pull/21546) | open | [Fix] Catch MalformedJSON exception for DeepSeek-V3.2 function call partial parsing | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | [CI] Move v32 cp test to deepep running suite | `test/registered/cp/test_deepseek_v32_cp_single_node.py` |
| 2026-03-29 | [#21623](https://github.com/sgl-project/sglang/pull/21623) | open | [Test] Add unit tests for encoding_dsv32.py | `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` |
| 2026-03-30 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-01 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | [DSA] Support trtllm sparse mla kernel for prefill batches | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/test/run_eval.py` |
| 2026-04-02 | [#21889](https://github.com/sgl-project/sglang/pull/21889) | open | [AMD] Enable FP4 (E2M1) KV cache quantization for NSA with TileLang backend | `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | [DSA] Set trtllm kernels as default for Blackwell | `python/sglang/srt/server_args.py` |
| 2026-04-03 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models. | `python/sglang/srt/server_args.py` |
| 2026-04-05 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | Enable IndexCache for DeepSeek V3.2 | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22238](https://github.com/sgl-project/sglang/pull/22238) | merged | [HiSparse]: Add readme docs for HiSparse Feature | `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22268](https://github.com/sgl-project/sglang/pull/22268) | open | [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3 | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-07 | [#21932](https://github.com/sgl-project/sglang/pull/21932) | merged | [HiSparse] Optimize the scheduling of decode backup. | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/managers/hisparse_coordinator.py` |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | Reduce unnecessary kernels and copies in the NSA indexer | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | [HiSparse]: Add HiSpares-DSA Model's nightly CI | `test/registered/8-gpu-models/test_dsa_models_hisparse.py` |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-09 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | [DSA] Enable all reduce fusion for DSA models | `python/sglang/srt/server_args.py` |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | [Fix] Fix several bugs on DSA models | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py` |
| 2026-04-09 | [#22473](https://github.com/sgl-project/sglang/pull/22473) | open | [DSA] Add dense MLA decode fallback for short sequences | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/server_args.py` |
| 2026-04-10 | [#13546](https://github.com/sgl-project/sglang/pull/13546) | closed | [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-10 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-12 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | [DSA] Hopper FP8 FlashMLA KV padding | `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py` |
| 2026-04-14 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | [HiCache & HybridModel] mooncake backend support DSA & mamba model | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-04-14 | [#22792](https://github.com/sgl-project/sglang/pull/22792) | open | nsa indexer: use aiter indexer_k_quant_and_cache | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-15 | [#22851](https://github.com/sgl-project/sglang/pull/22851) | open | [RL] [DSv32] [GLM-5] Add `--nsa-topk-backend` and integrate FlashInfer and pytorch topk | `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`, `python/sglang/srt/server_args.py` |
| 2026-04-15 | [#22865](https://github.com/sgl-project/sglang/pull/22865) | open | [sparsity] extend framework to support non-NSA sparse algorithms | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2026-04-16 | [#22938](https://github.com/sgl-project/sglang/pull/22938) | open | [AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122) | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-17 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` |
| 2026-04-19 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors | `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` |
| 2026-04-20 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | Support allreduce fusion with cp | `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py` |
| 2026-04-20 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` |
| 2026-04-20 | [#23268](https://github.com/sgl-project/sglang/pull/23268) | open | 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-04-20 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-04-20 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | Support piecewise CUDA graph with NSA | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/radix_attention.py` |
| 2026-04-24 | [#22774](https://github.com/sgl-project/sglang/pull/22774) | merged | [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1) | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` |
| 2026-04-24 | [#23241](https://github.com/sgl-project/sglang/pull/23241) | merged | [HiCache & HybridModel] 3FS backend support DSA & mamba model | `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`, `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`, `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` |

## Per-PR Diff Audit Cards

### PR #11191 - [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA.

- Link: https://github.com/sgl-project/sglang/pull/11191
- Status/date: open / 2025-10-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 52 files, +18474/-70, 16143 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA.". The diff centers on `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`. PR body context: The main implementation is currently in a private repository. I will migrate the code to this branch within a week. ## Motivation Support sparse attention computation for LLMs u...
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +148/-70 (218 lines); hunks: -22,6 +22,14; -340,6 +348,32 @@ def __init__(; symbols: FlashAttentionMetadata, __init__, init_forward_metadata, touching `FlashAttentionMetadata, __init__, init_forward_metadata`; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -533,6 +533,8 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize, touching `initialize`; `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` added +2560/-0 (2560 lines); `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py` added +1547/-0 (1547 lines); hunks: -0,0 +1,1547; symbols: FlashAttentionBackwardSm80, __init__, can_implement, _check_type, touching `FlashAttentionBackwardSm80, __init__, can_implement`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +148/-70 (218 lines); hunks: -22,6 +22,14; -340,6 +348,32 @@ def __init__(; symbols: FlashAttentionMetadata, __init__, init_forward_metadata
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-0 (2 lines); hunks: -533,6 +533,8 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` added +2560/-0 (2560 lines)
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py` added +1547/-0 (1547 lines); hunks: -0,0 +1,1547; symbols: FlashAttentionBackwardSm80, __init__, can_implement, _check_type
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` added +1402/-0 (1402 lines); hunks: -0,0 +1,1402; symbols: FlashAttentionForwardSm90, __init__, _get_smem_layout_atom, _get_tiled_mma
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -22,6 +22,14 @@
+from sglang.srt.sparse_attention.cache_manager.cache_manager import ManagerConfig
+from sglang.srt.sparse_attention.kernels.attention.interface import (
+    flash_attn_with_kvcache as cute_flash_attn_with_kvcache,
+)
+from sglang.srt.sparse_attention.updater.flashattention.cache_updater import (
+    LServerUpdaterFlashAttentionBackend,
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -533,6 +533,8 @@ def initialize(self, min_per_gpu_memory: float):
+        if hasattr(self.attn_backend, "sparse_cache_updater"):
+            self.attn_backend.sparse_cache_updater.cache_manager.start_retrive_loop()
diff -- python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py
@@ -0,0 +1,1547 @@
+# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
+# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_bwd_sm80.hpp
+# from Cutlass C++ to Cute-DSL.
+import math
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +148/-70; `python/sglang/srt/model_executor/model_runner.py` modified +2/-0; `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` added +2560/-0; `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py` added +1547/-0; `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` added +1402/-0; `python/sglang/srt/sparse_attention/kernels/attention/interface.py` added +1266/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/duoattention/__init__.py`, `python/sglang/test/attention/duoattention/streaming_attention_ref.py`, `python/sglang/test/attention/duoattention/test_streaming_attention.py`, `python/sglang/test/attention/duoattention/test_streaming_mask.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11063 - Add DeepSeek-V3.2 Tool Call Template

- Link: https://github.com/sgl-project/sglang/pull/11063
- Status/date: merged / 2025-10-05
- Trace source: `git log --name-only -- <model-files>` found it through `examples/chat_template/tool_chat_template_deepseekv32.jinja`; associated commits `85c1f7937781`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +100/-0, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add DeepSeek-V3.2 Tool Call Template". The diff centers on `examples/chat_template/tool_chat_template_deepseekv32.jinja`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
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

### PR #11194 - [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2

- Link: https://github.com/sgl-project/sglang/pull/11194
- Status/date: merged / 2025-10-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +588/-1, 623 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2". The diff centers on `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`. PR body context: ## Motivation This kernel is written for DeepSeek v3.2 indexer. Indexer of Deepseek Sparse Attention requires a variable-length topk kernel, which is extremely crucial for ## Mo...
- Key implementation: `sgl-kernel/csrc/elementwise/topk.cu` added +422/-0 (422 lines); hunks: -0,0 +1,422; `sgl-kernel/tests/test_topk.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, assert_equal, test_topk_kernel, touching `_ref_torch_impl, _ref_torch_transform_decode_impl, assert_equal`; `sgl-kernel/python/sgl_kernel/top_k.py` modified +29/-0 (29 lines); hunks: -9,3 +9,32 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_transform_fused, touching `fast_topk, fast_topk_v2, fast_topk_transform_fused`; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: -174,6 +174,14 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor&....
- Code diff details:
  - `sgl-kernel/csrc/elementwise/topk.cu` added +422/-0 (422 lines); hunks: -0,0 +1,422
  - `sgl-kernel/tests/test_topk.py` added +120/-0 (120 lines); hunks: -0,0 +1,120; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, assert_equal, test_topk_kernel
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +29/-0 (29 lines); hunks: -9,3 +9,32 @@ def fast_topk(values, topk, dim):; symbols: fast_topk, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: -174,6 +174,14 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor&...
  - `sgl-kernel/csrc/common_extension.cc` modified +7/-0 (7 lines); hunks: -107,6 +107,13 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/elementwise/topk.cu
@@ -0,0 +1,422 @@
+/**
+ * @NOTE: This file is adapted from
+ * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
+ * We:
+ * 1. adapt from tilelang to pure cuda
+ * 2. optimize the performance a little
diff -- sgl-kernel/tests/test_topk.py
@@ -0,0 +1,120 @@
+import pytest
+import torch
+from sgl_kernel import fast_topk_transform_fused, fast_topk_v2
+def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
+    assert score.dim() == 2
+    return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
diff -- sgl-kernel/python/sgl_kernel/top_k.py
@@ -9,3 +9,32 @@ def fast_topk(values, topk, dim):
```

- Reviewed files:
  - other: `sgl-kernel/csrc/elementwise/topk.cu` added +422/-0; `sgl-kernel/python/sgl_kernel/top_k.py` modified +29/-0; `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0; `sgl-kernel/csrc/common_extension.cc` modified +7/-0; `sgl-kernel/python/sgl_kernel/__init__.py` modified +1/-1; `sgl-kernel/CMakeLists.txt` modified +1/-0
  - tests: `sgl-kernel/tests/test_topk.py` added +120/-0
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_topk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11061 - Support DeepSeek V3.2 Exp

- Link: https://github.com/sgl-project/sglang/pull/11061
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` and 9 files; associated commits `efbc687c2817`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +4542/-141, 5365 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Support DeepSeek V3.2 Exp". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: We now support DeepSeek V3.2. Try this model in the tracking issue #11060 --- Authors: General: @fzyzcjy @DarkSharpness @hnyls2002 @hebiao064 @Fridge003; AMD: @HaiShaw @kkHuang-...
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

### PR #11309 - [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size

- Link: https://github.com/sgl-project/sglang/pull/11309
- Status/date: merged / 2025-10-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +25/-7, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation For DS v3.2 to avoid OOM. ## Results On 8xB200: With this PR: With main (hardcoded mem-fraction-static=0.8) - KV size including index is actually 65.49 GB ## Modif...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunks: -1280,6 +1280,17 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: profile_max_num_token, touching `profile_max_num_token`; `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-4 (18 lines); hunks: -1177,7 +1177,9 @@ def __init__(; -1298,6 +1300,9 @@ def load_cpu_copy(self, kv_cache_cpu, indices):; symbols: __init__, get_kv_size_bytes, load_cpu_copy, NSATokenToKVPool, touching `__init__, get_kv_size_bytes, load_cpu_copy`; `python/sglang/srt/server_args.py` modified +0/-3 (3 lines); hunks: -857,9 +857,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunks: -1280,6 +1280,17 @@ def profile_max_num_token(self, total_gpu_memory: int):; symbols: profile_max_num_token
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-4 (18 lines); hunks: -1177,7 +1177,9 @@ def __init__(; -1298,6 +1300,9 @@ def load_cpu_copy(self, kv_cache_cpu, indices):; symbols: __init__, get_kv_size_bytes, load_cpu_copy, NSATokenToKVPool
  - `python/sglang/srt/server_args.py` modified +0/-3 (3 lines); hunks: -857,9 +857,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1280,6 +1280,17 @@ def profile_max_num_token(self, total_gpu_memory: int):
+            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
+            if is_deepseek_nsa(self.model_config.hf_config):
+                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
+                indexer_size_per_token = (
+                    index_head_dim
+                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -1177,7 +1177,9 @@ def __init__(
-        self._finalize_allocation_log(size)
+        if not use_nsa:
+            # NSA will allocate indexer KV cache later and then log the total size
+            self._finalize_allocation_log(size)
@@ -1298,6 +1300,9 @@ def load_cpu_copy(self, kv_cache_cpu, indices):
+    quant_block_size = 128
diff -- python/sglang/srt/server_args.py
@@ -857,9 +857,6 @@ def _handle_model_specific_adjustments(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +11/-0; `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-4; `python/sglang/srt/server_args.py` modified +0/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11450 - [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton

- Link: https://github.com/sgl-project/sglang/pull/11450
- Status/date: merged / 2025-10-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`; associated commits `451d15c44be3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +420/-1, 431 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton". The diff centers on `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation We want to remove the tilelang dependency because it relies on tvm, but the tvm version required by tilelang is incompatible with the one used by flashinfer. This...
- Key implementation: `python/sglang/srt/layers/attention/nsa/triton_kernel.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: _act_quant_kernel, act_quant, touching `_act_quant_kernel, act_quant`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-1 (4 lines); hunks: -505,8 +505,10 @@ def _forward(; symbols: _forward, touching `_forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` added +136/-0 (136 lines); hunks: -0,0 +1,136; symbols: _act_quant_kernel, act_quant
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-1 (4 lines); hunks: -505,8 +505,10 @@ def _forward(; symbols: _forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/triton_kernel.py
@@ -0,0 +1,136 @@
+from typing import Optional, Tuple
+import torch
+import triton
+import triton.language as tl
+# Triton implementation
+@triton.jit
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -505,8 +505,10 @@ def _forward(
-        if not is_npu():
+        if is_hip():
+        elif not is_npu():
+            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/triton_kernel.py` added +136/-0; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/srt/layers/attention/nsa/test_act_quant_triton.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11557 - Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3))

- Link: https://github.com/sgl-project/sglang/pull/11557
- Status/date: merged / 2025-10-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3))". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation Changes to DeepSeek-R1-FP4 default config in #11512 are accidently being applied to DeepSeek-V3.2-Exp. On blackwell, this causes the attention backend for 3.2 to b...
- Key implementation: `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -804,7 +804,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -804,7 +804,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -804,7 +804,7 @@ def _handle_model_specific_adjustments(self):
-        if model_arch in ["DeepseekV3ForCausalLM"]:
+        if model_arch in ["DeepseekV3ForCausalLM"] and not is_deepseek_nsa(hf_config):
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11308 - [CI] Add Basic Test for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11308
- Status/date: merged / 2025-10-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +137/-4, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[CI] Add Basic Test for DeepSeek V3.2". The diff centers on `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/srt/test_deepseek_v32_basic.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV3Basic, setUpClass, tearDownClass`; `.github/workflows/pr-test.yml` modified +30/-3 (33 lines); hunks: -664,6 +664,33 @@ jobs:; -674,19 +701,19 @@ jobs:; `scripts/ci/ci_install_dependency.sh` modified +26/-1 (27 lines); hunks: -3,6 +3,7; -68,7 +69,31 @@ if [ "$IS_BLACKWELL" != "1" ]; then; `test/srt/run_suite.py` modified +3/-0 (3 lines); hunks: -169,6 +169,9 @@ class TestFile:; symbols: TestFile, touching `TestFile`.
- Code diff details:
  - `test/srt/test_deepseek_v32_basic.py` added +78/-0 (78 lines); hunks: -0,0 +1,78; symbols: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `.github/workflows/pr-test.yml` modified +30/-3 (33 lines); hunks: -664,6 +664,33 @@ jobs:; -674,19 +701,19 @@ jobs:
  - `scripts/ci/ci_install_dependency.sh` modified +26/-1 (27 lines); hunks: -3,6 +3,7; -68,7 +69,31 @@ if [ "$IS_BLACKWELL" != "1" ]; then
  - `test/srt/run_suite.py` modified +3/-0 (3 lines); hunks: -169,6 +169,9 @@ class TestFile:; symbols: TestFile
- Key code excerpts:

```diff
diff -- test/srt/test_deepseek_v32_basic.py
@@ -0,0 +1,78 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
+from sglang.test.test_utils import (
diff -- .github/workflows/pr-test.yml
@@ -664,6 +664,33 @@ jobs:
+  unit-test-backend-8-gpu-deepseek-v32:
+    needs: [check-changes, unit-test-backend-2-gpu, sgl-kernel-build-wheels]
+    if: always() && !failure() && !cancelled() &&
+        ((needs.check-changes.outputs.main_package == 'true') || (needs.check-changes.outputs.sgl_kernel == 'true'))
+    runs-on: 8-gpu-h200
+    steps:
diff -- scripts/ci/ci_install_dependency.sh
@@ -3,6 +3,7 @@
```

- Reviewed files:
  - tests: `test/srt/test_deepseek_v32_basic.py` added +78/-0; `test/srt/run_suite.py` modified +3/-0
  - ci: `.github/workflows/pr-test.yml` modified +30/-3
  - other: `scripts/ci/ci_install_dependency.sh` modified +26/-1
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_basic.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11565 - [DSv32] Use torch.compile for _get_logits_head_gate

- Link: https://github.com/sgl-project/sglang/pull/11565
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `384733639a91`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DSv32] Use torch.compile for _get_logits_head_gate". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation fuse 2 unary mul + 1 binary mul in _get_logits_head_gate. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling 1.8% e2e total throughput With torch.com...
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

### PR #11596 - [Spec Decoding] Support MTP for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/11596
- Status/date: closed / 2025-10-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +515/-534, 1349 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Spec Decoding] Support MTP for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py`. PR body context: ## Motivation Based on https://github.com/sgl-project/sglang/pull/11109 We have implemented MTP support for DS v3.2 and ***cuda graph*** in our in-house maintained version of sg...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +396/-65 (461 lines); hunks: -30,6 +30,10; -149,7 +153,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__, touching `compute_cu_seqlens, NativeSparseAttnBackend, __init__`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +82/-8 (90 lines); hunks: -17,6 +17,8; -27,6 +29,8; symbols: _get_topk_paged, _get_verify_topk_paged, _get_topk_ragged, touching `_get_topk_paged, _get_verify_topk_paged, _get_topk_ragged`; `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: is_deepseek_nsa, touching `is_deepseek_nsa`; `.github/workflows/pr-test-amd.yml` removed +0/-352 (352 lines); hunks: -1,352 +0,0.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +396/-65 (461 lines); hunks: -30,6 +30,10; -149,7 +153,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; symbols: compute_cu_seqlens, NativeSparseAttnBackend, __init__
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +82/-8 (90 lines); hunks: -17,6 +17,8; -27,6 +29,8; symbols: _get_topk_paged, _get_verify_topk_paged, _get_topk_ragged
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunks: -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: is_deepseek_nsa
  - `.github/workflows/pr-test-amd.yml` removed +0/-352 (352 lines); hunks: -1,352 +0,0
  - `.github/workflows/release-docker-dev.yml` removed +0/-108 (108 lines); hunks: -1,108 +0,0
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -30,6 +30,10 @@
+import logging
+logger = logging.getLogger(__name__)
@@ -149,7 +153,14 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
-    def __init__(self, model_runner: ModelRunner):
+    def __init__(
+        self,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -17,6 +17,8 @@
+import logging
@@ -27,6 +29,8 @@
+logger = logging.getLogger(__name__)
@@ -336,6 +340,65 @@ def _get_topk_paged(
+    def _get_verify_topk_paged(
+        self,
diff -- python/sglang/srt/configs/model_config.py
@@ -53,7 +53,11 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +396/-65; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +82/-8; `python/sglang/srt/configs/model_config.py` modified +5/-1; `python/sglang/srt/speculative/draft_utils.py` modified +16/-0; `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +8/-0; `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +8/-0
  - ci: `.github/workflows/pr-test-amd.yml` removed +0/-352; `.github/workflows/release-docker-dev.yml` removed +0/-108
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- Link: https://github.com/sgl-project/sglang/pull/10912
- Status/date: merged / 2025-10-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +727/-186, 1433 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`. PR body context: ## Motivation and Modifications To support hybrid attention models like Qwen3-Next with mamba state or DeepSeek V3.2 Exp with indexer cache, this PR extends the current PD disag...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool, touching `init_memory_pool`; `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from, touching `__init__, for, get_speculative_mamba2_params_all_layers`; `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo, touching `TransferKVChunk, TransferInfo, from_zmq`; `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest, touching `clear, HybridMambaDecodeReqToTokenPool, __init__`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1670,19 +1670,34 @@ def init_memory_pool(
-                from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
+                from sglang.srt.disaggregation.decode import (
+                    DecodeReqToTokenPool,
+                    HybridMambaDecodeReqToTokenPool,
+                )
-                self.req_to_token_pool = DecodeReqToTokenPool(
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -142,72 +142,93 @@ def __init__(
-        # assume conv_state = (dim, state_len)
-        assert conv_state_shape[0] > conv_state_shape[1]
-        conv_state = torch.zeros(
-            size=(num_mamba_layers, size + 1) + conv_state_shape,
-            dtype=conv_dtype,
-            device=device,
diff -- python/sglang/srt/disaggregation/mooncake/conn.py
@@ -58,6 +58,7 @@ class TransferKVChunk:
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9; `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137; `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17; `python/sglang/srt/disaggregation/decode.py` modified +113/-8; `python/sglang/srt/disaggregation/prefill.py` modified +71/-1; `python/sglang/srt/disaggregation/base/conn.py` modified +17/-4
  - tests: `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_disaggregation_hybrid_attention.py`, `test/srt/test_mamba_unittest.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11510 - [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support

- Link: https://github.com/sgl-project/sglang/pull/11510
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `3cceaa381ad3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +102/-33, 359 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Fix Qwen3/DSV3/DSV3.2 model support". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation This pr fixes very models that failed to start on npu after dsv3.2 support pr was merged. ci and image release infra was also improved. ## Modifications - Fix mla...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Draft] Support MTP for DeepSeek-V3.2". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/configs/model_config.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
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
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "Cleaning indexer for DeepSeek V3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`. PR body context: ## Motivation - Fake indexer is for testing accuracy during early development, so can be deprecated now. - Cleaning forward logic. ## Modifications ## Accuracy Tests ## Benchmar...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Spec Decoding] Support MTP for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Based on https://github.com/sgl-project/sglang/pull/11109 We have implemented MTP support for DS v3.2 and ***cuda graph*** in our in-house maintained version of sg...
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

### PR #11835 - [CI] Add CI test for DeepSeek V3.2 MTP

- Link: https://github.com/sgl-project/sglang/pull/11835
- Status/date: merged / 2025-10-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +112/-3, 151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[CI] Add CI test for DeepSeek V3.2 MTP". The diff centers on `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/srt/test_deepseek_v32_mtp.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestDeepseekV32MTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32MTP, setUpClass, tearDownClass`; `test/srt/test_deepseek_v32_basic.py` modified +3/-3 (6 lines); hunks: -16,7 +16,7; -57,7 +57,7 @@ def test_a_gsm8k(; symbols: TestDeepseekV3Basic, TestDeepseekV32Basic, setUpClass, test_a_gsm8k, touching `TestDeepseekV3Basic, TestDeepseekV32Basic, setUpClass`; `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunks: -1201,6 +1201,9 @@ def _handle_speculative_decoding(self):; symbols: _handle_speculative_decoding, touching `_handle_speculative_decoding`; `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -181,6 +181,7 @@ class TestFile:; symbols: TestFile, touching `TestFile`.
- Code diff details:
  - `test/srt/test_deepseek_v32_mtp.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestDeepseekV32MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/test_deepseek_v32_basic.py` modified +3/-3 (6 lines); hunks: -16,7 +16,7; -57,7 +57,7 @@ def test_a_gsm8k(; symbols: TestDeepseekV3Basic, TestDeepseekV32Basic, setUpClass, test_a_gsm8k
  - `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunks: -1201,6 +1201,9 @@ def _handle_speculative_decoding(self):; symbols: _handle_speculative_decoding
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -181,6 +181,7 @@ class TestFile:; symbols: TestFile
- Key code excerpts:

```diff
diff -- test/srt/test_deepseek_v32_mtp.py
@@ -0,0 +1,105 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
diff -- test/srt/test_deepseek_v32_basic.py
@@ -16,7 +16,7 @@
-class TestDeepseekV3Basic(CustomTestCase):
+class TestDeepseekV32Basic(CustomTestCase):
@@ -57,7 +57,7 @@ def test_a_gsm8k(
-                f"### test_gsm8k (deepseek-v3)\n" f'{metrics["accuracy"]=:.3f}\n'
+                f"### test_gsm8k (deepseek-v32)\n" f'{metrics["accuracy"]=:.3f}\n'
@@ -69,7 +69,7 @@ def test_bs_1_speed(self):
diff -- python/sglang/srt/server_args.py
@@ -1201,6 +1201,9 @@ def _handle_speculative_decoding(self):
```

- Reviewed files:
  - tests: `test/srt/test_deepseek_v32_mtp.py` added +105/-0; `test/srt/test_deepseek_v32_basic.py` modified +3/-3; `test/srt/run_suite.py` modified +1/-0
  - runtime: `python/sglang/srt/server_args.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_basic.py`, `test/srt/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11815 - [DeepseekV32] Add fast_topk_transform_ragged_fused kernel

- Link: https://github.com/sgl-project/sglang/pull/11815
- Status/date: merged / 2025-10-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +201/-20, 323 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepseekV32] Add fast_topk_transform_ragged_fused kernel". The diff centers on `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`. PR body context: ## Motivation Add a fused kernel for `fast_topk_transform_ragged_fused`. The difference between this kernel and `fast_topk_transform_fused` is that `fast_topk_transform_fused` o...
- Key implementation: `sgl-kernel/csrc/elementwise/topk.cu` modified +81/-8 (89 lines); hunks: -51,6 +51,15 @@ __device__ void naive_topk_transform(; -322,8 +331,40 @@ __global__ __launch_bounds__(kThreadsPerBlock) // prefill; `sgl-kernel/tests/test_topk.py` modified +75/-4 (79 lines); hunks: -1,6 +1,12; -26,6 +32,21 @@ def _ref_torch_transform_decode_impl(; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal, touching `_ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl`; `sgl-kernel/python/sgl_kernel/top_k.py` modified +24/-1 (25 lines); hunks: -28,13 +28,36 @@ def fast_topk_transform_fused(; symbols: fast_topk_transform_fused, fast_topk_transform_ragged_fused, touching `fast_topk_transform_fused, fast_topk_transform_ragged_fused`; `sgl-kernel/include/sgl_kernel_ops.h` modified +11/-6 (17 lines); hunks: -174,13 +174,18 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor....
- Code diff details:
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +81/-8 (89 lines); hunks: -51,6 +51,15 @@ __device__ void naive_topk_transform(; -322,8 +331,40 @@ __global__ __launch_bounds__(kThreadsPerBlock) // prefill
  - `sgl-kernel/tests/test_topk.py` modified +75/-4 (79 lines); hunks: -1,6 +1,12; -26,6 +32,21 @@ def _ref_torch_transform_decode_impl(; symbols: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +24/-1 (25 lines); hunks: -28,13 +28,36 @@ def fast_topk_transform_fused(; symbols: fast_topk_transform_fused, fast_topk_transform_ragged_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +11/-6 (17 lines); hunks: -174,13 +174,18 @@ void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor...
  - `sgl-kernel/python/sgl_kernel/__init__.py` modified +6/-1 (7 lines); hunks: -325,7 +325,12 @@ def _find_cuda_home():; symbols: _find_cuda_home
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/elementwise/topk.cu
@@ -51,6 +51,15 @@ __device__ void naive_topk_transform(
+// keep the first `length` entries, set others to -1
+__device__ void naive_topk_transform_ragged(
+    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
+  const auto tid = threadIdx.x;
+  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
+    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
diff -- sgl-kernel/tests/test_topk.py
@@ -1,6 +1,12 @@
+from typing import Optional
-from sgl_kernel import fast_topk_transform_fused, fast_topk_v2
+from sgl_kernel import (
+    fast_topk_transform_fused,
+    fast_topk_transform_ragged_fused,
+    fast_topk_v2,
diff -- sgl-kernel/python/sgl_kernel/top_k.py
@@ -28,13 +28,36 @@ def fast_topk_transform_fused(
```

- Reviewed files:
  - other: `sgl-kernel/csrc/elementwise/topk.cu` modified +81/-8; `sgl-kernel/python/sgl_kernel/top_k.py` modified +24/-1; `sgl-kernel/include/sgl_kernel_ops.h` modified +11/-6; `sgl-kernel/python/sgl_kernel/__init__.py` modified +6/-1; `sgl-kernel/csrc/common_extension.cc` modified +4/-0
  - tests: `sgl-kernel/tests/test_topk.py` modified +75/-4
- Risk and verification: The diff ships test coverage in `sgl-kernel/tests/test_topk.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #11876 - Rename flashmla kernel options of nsa backend for better readability

- Link: https://github.com/sgl-project/sglang/pull/11876
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `ef4a8097b8e1`
- Diff scope read: GitHub Pull Request files API returned 3 files, +31/-31, 206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Rename flashmla kernel options of nsa backend for better readability". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation - Rename `--nsa-prefill` and `--nsa-decode` to `--nsa-prefill-backend` and `--nsa-decode-backend` - Add documents for `--nsa-prefill-backend` and `--nsa-decode-bac...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-21 (40 lines); hunks: -140,9 +140,7 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; -181,8 +179,8 @@ def __init__(; symbols: compute_cu_seqlens, __init__, init_forward_metadata, init_cuda_graph_state, touching `compute_cu_seqlens, __init__, init_forward_metadata`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-21 (40 lines); hunks: -140,9 +140,7 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; -181,8 +179,8 @@ def __init__(; symbols: compute_cu_seqlens, __init__, init_forward_metadata, init_cuda_graph_state
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -140,9 +140,7 @@ def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
-_NSA_IMPL_T: TypeAlias = Literal[
-    "flashmla_prefill", "flashmla_decode", "fa3", "tilelang"
-]
+_NSA_IMPL_T: TypeAlias = Literal["flashmla_sparse", "flashmla_kv", "fa3", "tilelang"]
@@ -181,8 +179,8 @@ def __init__(
-        NSA_PREFILL_IMPL = model_runner.server_args.nsa_prefill
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-21
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11761 - (beta)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/11761
- Status/date: closed / 2025-10-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 0 files, +0/-0, 0 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "(beta)support context parallel with deepseekv3.2-DSA". The diff centers on the file-level diff. PR body context: ## Motivation Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. Current description： 1. Curr...
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

### PR #12017 - (beta)support context parallel with deepseekv3.2-DSA

- Link: https://github.com/sgl-project/sglang/pull/12017
- Status/date: closed / 2025-10-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +595/-81, 1173 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "(beta)support context parallel with deepseekv3.2-DSA". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Motivation Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. Current description： Current...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +133/-50 (183 lines); hunks: -66,6 +66,7; -113,7 +114,10; symbols: forward, __init__, forward_deepep, touching `forward, __init__, forward_deepep`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +174/-5 (179 lines); hunks: -17,9 +17,15; -32,6 +38,8; symbols: BaseIndexerMetadata, __init__, _get_q_k_bf16, _get_topk_ragged, touching `BaseIndexerMetadata, __init__, _get_q_k_bf16`; `python/sglang/srt/models/deepseek_nextn.py` modified +50/-9 (59 lines); hunks: -14,15 +14,18; -38,7 +41,13; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/layers/communicator.py` modified +25/-9 (34 lines); hunks: -53,6 +53,8; -415,6 +417,8 @@ def _scattered_to_tp_attn_full(; symbols: _scattered_to_tp_attn_full, _scatter_hidden_states_and_residual, _scatter_hidden_states, _gather, touching `_scattered_to_tp_attn_full, _scatter_hidden_states_and_residual, _scatter_hidden_states`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +133/-50 (183 lines); hunks: -66,6 +66,7; -113,7 +114,10; symbols: forward, __init__, forward_deepep
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +174/-5 (179 lines); hunks: -17,9 +17,15; -32,6 +38,8; symbols: BaseIndexerMetadata, __init__, _get_q_k_bf16, _get_topk_ragged
  - `python/sglang/srt/models/deepseek_nextn.py` modified +50/-9 (59 lines); hunks: -14,15 +14,18; -38,7 +41,13; symbols: __init__, forward
  - `python/sglang/srt/layers/communicator.py` modified +25/-9 (34 lines); hunks: -53,6 +53,8; -415,6 +417,8 @@ def _scattered_to_tp_attn_full(; symbols: _scattered_to_tp_attn_full, _scatter_hidden_states_and_residual, _scatter_hidden_states, _gather
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-4 (28 lines); hunks: -32,6 +32,9; -117,18 +120,35 @@ def topk_transform(; symbols: topk_transform
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -66,6 +66,7 @@
+    attn_tp_all_gather_reorgan_into_tensor,
@@ -113,7 +114,10 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
+from sglang.srt.model_executor.forward_batch_info import (
+    ForwardBatch,
+    PPProxyTensors,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -17,9 +17,15 @@
-from sglang.srt.layers.dp_attention import get_attention_tp_group
+from sglang.srt.layers.dp_attention import (
+    get_attention_tp_group,
+    get_attention_tp_rank,
+    get_attention_tp_size,
+    attn_tp_all_gather_reorgan_into_tensor)
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -14,15 +14,18 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +133/-50; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +174/-5; `python/sglang/srt/models/deepseek_nextn.py` modified +50/-9; `python/sglang/srt/layers/communicator.py` modified +25/-9; `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-4; `python/sglang/srt/model_executor/forward_batch_info.py` modified +26/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11877 - [Doc] Add documentation for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/11877
- Status/date: merged / 2025-10-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`; associated commits `729b242934cb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +723/-3, 749 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Doc] Add documentation for DeepSeek V3.2". The diff centers on `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`. PR body context: Depends on #11844 and #11876 Also waiting for @whybeyoung on some PD updates ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Fo...
- Key implementation: `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunks: -0,0 +1,570; `docs/basic_usage/deepseek_v32.md` added +150/-0 (150 lines); hunks: -0,0 +1,150.
- Code diff details:
  - `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunks: -0,0 +1,570
  - `docs/basic_usage/deepseek_v32.md` added +150/-0 (150 lines); hunks: -0,0 +1,150
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
diff -- docs/basic_usage/deepseek_v32.md
@@ -0,0 +1,150 @@
+# DeepSeek V3.2 Usage
+[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA,
+For reporting issues or tracking upcoming features, please refer to this [Roadmap](https://github.com/sgl-project/sglang/issues/11060).
+## Installation
+### Docker
+'''bash
```

- Reviewed files:
  - docs: `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0; `docs/basic_usage/deepseek_v32.md` added +150/-0
- Risk and verification: This is mostly docs/examples in `docs/advanced_features/separate_reasoning.ipynb`, `docs/basic_usage/deepseek.md`, `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12052 - Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models

- Link: https://github.com/sgl-project/sglang/pull/12052
- Status/date: closed / 2025-10-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models". The diff centers on `python/sglang/srt/layers/dp_attention.py`. PR body context: ## Motivation Fix illegal memory access (IMA) errors when using DP attention with DeepSeek-V3.2 models. **Issue**: When running with `--enable-dp-attention`, the DP gather opera...
- Key implementation: `python/sglang/srt/layers/dp_attention.py` modified +18/-1 (19 lines); hunks: -419,7 +419,24 @@ def _dp_gather_via_all_reduce(; symbols: _dp_gather_via_all_reduce, touching `_dp_gather_via_all_reduce`.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +18/-1 (19 lines); hunks: -419,7 +419,24 @@ def _dp_gather_via_all_reduce(; symbols: _dp_gather_via_all_reduce
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/dp_attention.py
@@ -419,7 +419,24 @@ def _dp_gather_via_all_reduce(
-    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)
+    # LogitsMetadata should have dp_local_start_pos set by compute_dp_attention_metadata().
+    # Avoid calling get_dp_local_info() to maintain separation of concerns.
+    if type(forward_batch).__name__ == "LogitsMetadata":
+        assert (
+            forward_batch.dp_local_start_pos is not None
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/dp_attention.py` modified +18/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/dp_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12130 - [Doc] Fix format for deepseek v3.2 document

- Link: https://github.com/sgl-project/sglang/pull/12130
- Status/date: merged / 2025-10-25
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `bcecf27e7ca2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Doc] Fix format for deepseek v3.2 document". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +3/-3 (6 lines); hunks: -60,7 +60,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...; -71,10 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +3/-3 (6 lines); hunks: -60,7 +60,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...; -71,10 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -60,7 +60,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
-### Multi-token Prediction
+## Multi-token Prediction
@@ -71,10 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
-# Function Calling and Reasoning Parser
+## Function Calling and Reasoning Parser
-# PD Disaggregation
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +3/-3
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #11936 - [Test] Add dsv3.2 nsa backend testing

- Link: https://github.com/sgl-project/sglang/pull/11936
- Status/date: merged / 2025-10-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +125/-0, 133 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Test] Add dsv3.2 nsa backend testing". The diff centers on `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/srt/test_deepseek_v32_nsabackend.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: TestDeepseekV32NasBackend_flashmla, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32NasBackend_flashmla, setUpClass, tearDownClass`; `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -191,6 +191,7 @@ class TestFile:; symbols: TestFile, touching `TestFile`.
- Code diff details:
  - `test/srt/test_deepseek_v32_nsabackend.py` added +124/-0 (124 lines); hunks: -0,0 +1,124; symbols: TestDeepseekV32NasBackend_flashmla, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -191,6 +191,7 @@ class TestFile:; symbols: TestFile
- Key code excerpts:

```diff
diff -- test/srt/test_deepseek_v32_nsabackend.py
@@ -0,0 +1,124 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
+from sglang.test.test_utils import (
diff -- test/srt/run_suite.py
@@ -191,6 +191,7 @@ class TestFile:
+        TestFile("test_deepseek_v32_nsabackend.py", 600),
```

- Reviewed files:
  - tests: `test/srt/test_deepseek_v32_nsabackend.py` added +124/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_nsabackend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12138 - [Doc] Small update of DeepSeek v3.2 document

- Link: https://github.com/sgl-project/sglang/pull/12138
- Status/date: merged / 2025-10-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `97828878d833`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Doc] Small update of DeepSeek v3.2 document". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Fixing a wrong sentence ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +1/-1 (2 lines); hunks: -48,7 +48,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -48,7 +48,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
-- **DP Attention**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`. So
+- **DP Attention**: For DeepSeek V3.2 model, the kernels are customized for the use case of `dp_size=8`, so DP attention is enabled by default for better stability and performance
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +1/-1
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #11655 - [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache

- Link: https://github.com/sgl-project/sglang/pull/11655
- Status/date: merged / 2025-10-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `81a632ace647`
- Diff scope read: GitHub Pull Request files API returned 4 files, +367/-44, 626 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`. PR body context: ## Motivation Add logics to dequant the kvcache from fp8 to bf16 in a separate kernel and use `flashmla_prefill` kernel with fp8 kvcache. - [x] Verify accuracy and perf on both...
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

### PR #12296 - Update deepseek_v32.md

- Link: https://github.com/sgl-project/sglang/pull/12296
- Status/date: merged / 2025-10-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `0ee831dee0a6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-5, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "Update deepseek_v32.md". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Modifications Update the configuration tips of Deepseek V3.2 after the optimizations in https://github.com/sgl-project/sglang/pull/11655 ## Checklist - [ ] Format your code a...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +4/-5 (9 lines); hunks: -50,15 +50,14 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +4/-5 (9 lines); hunks: -50,15 +50,14 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -50,15 +50,14 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
-  - `flashmla_sparse`: `flash_mla_sparse_fwd` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs.
-  - `flashmla_kv`: `flash_mla_with_kvcache` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs.
-  - `fa3`: `flash_attn_with_kvcache` kernel from `flash_attn` library. Can only run on Hopper GPUs.
+  - `flashmla_sparse`: `flash_mla_sparse_fwd` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, kv inputs.
+  - `flashmla_kv`: `flash_mla_with_kvcache` kernel from `flash_mla` library. Can run on both Hopper and Blackwell GPUs. It requires bf16 q, fp8 k_cache inputs.
+  - `fa3`: `flash_attn_with_kvcache` kernel from `flash_attn` library. Can only run on Hopper GPUs. It requires bf16 q, kv inputs.
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +4/-5
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12294 - [Deepseek V3.2] Enable flashmla_auto with MTP

- Link: https://github.com/sgl-project/sglang/pull/12294
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `42e1a72efb78`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-3, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Deepseek V3.2] Enable flashmla_auto with MTP". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Follow up to https://github.com/sgl-project/sglang/pull/11655 ## Modifications With fp8 kvcache on B200, flashmla_sparse can be used in the extend (not draft_exten...
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

### PR #12094 - Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4

- Link: https://github.com/sgl-project/sglang/pull/12094
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `9ff9fa7f95be`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +110/-22, 231 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation Since wk and weight_proj in Indexer have the same input, they can be fused into a single gemm by concatenating weights. For the base DeepSeekV3.2-Exp model, wk is...
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

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- Link: https://github.com/sgl-project/sglang/pull/12123
- Status/date: merged / 2025-10-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +331/-9, 380 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)". The diff centers on `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`. PR body context: This commit fixes the double JSON encoding issue in DeepSeek chat templates (v3, v3.1, v3.2) that was causing incorrect tool call formatting in multi-round function calling scen...
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

### PR #12300 - [DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type

- Link: https://github.com/sgl-project/sglang/pull/12300
- Status/date: merged / 2025-10-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/transform_index.py`; associated commits `662725b936a2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[DeepSeekV32] Bug fix to ensure `page_table` and `result` in same type". The diff centers on `python/sglang/srt/layers/attention/nsa/transform_index.py`. PR body context: Running transform_index.py in NSA attention path fails at torch.gather(...) with a dtype mismatch: the index/out tensor is int (int32) while PyTorch requires Long (int64) for ga...
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

### PR #12044 - Enable mixed type LayerNorm kernel for NSA indexer

- Link: https://github.com/sgl-project/sglang/pull/12044
- Status/date: merged / 2025-11-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `e607850fcfe8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +166/-25, 251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Enable mixed type LayerNorm kernel for NSA indexer". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Currently we cast input to layernorm to fp32 and use native Torch layernorm, then cast back. Instead we pull in a more efficient TRT-LLM kernel (via flashinfer) th...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-21 (23 lines); hunks: -4,11 +4,10; -83,24 +82,6 @@ def rotate_activation(x: torch.Tensor) -> torch.Tensor:; symbols: rotate_activation, V32LayerNorm, __init__, forward, touching `rotate_activation, V32LayerNorm, __init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-21 (23 lines); hunks: -4,11 +4,10; -83,24 +82,6 @@ def rotate_activation(x: torch.Tensor) -> torch.Tensor:; symbols: rotate_activation, V32LayerNorm, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -4,11 +4,10 @@
-import torch.nn.functional as F
-from torch import nn
+from sglang.srt.layers.layernorm import LayerNorm
@@ -83,24 +82,6 @@ def rotate_activation(x: torch.Tensor) -> torch.Tensor:
-class V32LayerNorm(nn.Module):
-    """
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-21
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_layernorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12645 - [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/12645
- Status/date: merged / 2025-11-04
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2". The diff centers on `python/sglang/srt/mem_cache/memory_pool.py`. PR body context: ## Motivation Fix this issue https://github.com/sgl-project/sglang/issues/12643 Root Cause: The override_kv_cache_dim calculation in NSATokenToKVPool.__init__() was using the wr...
- Key implementation: `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-1 (4 lines); hunks: -1568,6 +1568,7 @@ def load_cpu_copy(self, kv_cache_cpu, indices):; -1589,10 +1590,11 @@ def __init__(; symbols: load_cpu_copy, NSATokenToKVPool, __init__, touching `load_cpu_copy, NSATokenToKVPool, __init__`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-1 (4 lines); hunks: -1568,6 +1568,7 @@ def load_cpu_copy(self, kv_cache_cpu, indices):; -1589,10 +1590,11 @@ def __init__(; symbols: load_cpu_copy, NSATokenToKVPool, __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -1568,6 +1568,7 @@ def load_cpu_copy(self, kv_cache_cpu, indices):
+    rope_storage_dtype = torch.bfloat16  # rope is always stored in bf16
@@ -1589,10 +1590,11 @@ def __init__(
+        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
-            + qk_rope_head_dim * dtype.itemsize
+            + qk_rope_head_dim * self.rope_storage_dtype.itemsize
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11892 - DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill

- Link: https://github.com/sgl-project/sglang/pull/11892
- Status/date: merged / 2025-11-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `f235498eca7a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +188/-4, 255 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation For DeepSeek-V3.2 models, using MLA (Multi-Latent Attention) uniformly across all sequence lengths during prefill is suboptimal. For short sequences, the overhead...
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

### PR #12788 - [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)

- Link: https://github.com/sgl-project/sglang/pull/12788
- Status/date: merged / 2025-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `7257525ccea6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-6, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation Follow up this PR: DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill. Enable and optimize MHA on B200 (SM100) in the NSA backend by TRT-...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`; associated commits `125f76ea44d8`
- Diff scope read: GitHub Pull Request files API returned 3 files, +617/-1, 633 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Test] Add DeepSeekV3.2 NSA Indexer Test Suite". The diff centers on `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`. PR body context: ## Motivation Address the issue https://github.com/sgl-project/sglang/issues/12509 ## Modifications Added 11 tests: 1. test_forward_decode_mode 2. test_forward_extend_mode 3. te...
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

### PR #12820 - [WIP][Feature] support tp-sp on qwen2/3 & deepseek v2/3/3.2

- Link: https://github.com/sgl-project/sglang/pull/12820
- Status/date: open / 2025-11-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +434/-58, 1063 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[WIP][Feature] support tp-sp on qwen2/3 & deepseek v2/3/3.2". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen3.py`. PR body context: ## Motivation For the classic dense decode layer structure (self-attention + MLP), in the pure TP case, tensors are parallelized in the attention layer and the MLP layer. Since...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +173/-23 (196 lines); hunks: -57,9 +57,12; -288,7 +291,7 @@ def handle_attention_ascend(attn, forward_batch):; symbols: handle_attention_ascend, forward, __init__, touching `handle_attention_ascend, forward, __init__`; `python/sglang/srt/layers/communicator.py` modified +119/-11 (130 lines); hunks: -22,6 +22,7; -96,6 +97,7 @@ class _LayerModeComputationContext:; symbols: _LayerModeComputationContext, previous_layer, LayerScatterModes, init_new, touching `_LayerModeComputationContext, previous_layer, LayerScatterModes`; `python/sglang/srt/models/qwen3.py` modified +56/-7 (63 lines); hunks: -9,9 +9,18; -118,6 +127,7 @@ def __init__(; symbols: __init__, forward, get_layer_communicator, touching `__init__, forward, get_layer_communicator`; `python/sglang/srt/models/qwen2.py` modified +27/-4 (31 lines); hunks: -25,6 +25,7; -89,13 +90,13 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +173/-23 (196 lines); hunks: -57,9 +57,12; -288,7 +291,7 @@ def handle_attention_ascend(attn, forward_batch):; symbols: handle_attention_ascend, forward, __init__
  - `python/sglang/srt/layers/communicator.py` modified +119/-11 (130 lines); hunks: -22,6 +22,7; -96,6 +97,7 @@ class _LayerModeComputationContext:; symbols: _LayerModeComputationContext, previous_layer, LayerScatterModes, init_new
  - `python/sglang/srt/models/qwen3.py` modified +56/-7 (63 lines); hunks: -9,9 +9,18; -118,6 +127,7 @@ def __init__(; symbols: __init__, forward, get_layer_communicator
  - `python/sglang/srt/models/qwen2.py` modified +27/-4 (31 lines); hunks: -25,6 +25,7; -89,13 +90,13 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/linear.py` modified +17/-6 (23 lines); hunks: -10,6 +10,7; -1233,6 +1234,7 @@ def __init__(; symbols: __init__, weight_loader_v2, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -57,9 +57,12 @@
+    ScatterMode,
+    attn_tp_all_gather_into_tensor,
+    get_attention_tp_group,
@@ -288,7 +291,7 @@ def handle_attention_ascend(attn, forward_batch):
-    ):
+    ) or (forward_batch.enable_sp and forward_batch.forward_mode.is_idle()):
diff -- python/sglang/srt/layers/communicator.py
@@ -22,6 +22,7 @@
+    sp_tensor_model_parallel_all_gather,
@@ -96,6 +97,7 @@ class _LayerModeComputationContext:
+    enable_sp: Optional[bool] = False
@@ -104,6 +106,7 @@ def previous_layer(self):
+            enable_sp=self.enable_sp,
@@ -115,23 +118,47 @@ class LayerScatterModes:
diff -- python/sglang/srt/models/qwen3.py
@@ -9,9 +9,18 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +173/-23; `python/sglang/srt/layers/communicator.py` modified +119/-11; `python/sglang/srt/models/qwen3.py` modified +56/-7; `python/sglang/srt/models/qwen2.py` modified +27/-4; `python/sglang/srt/layers/linear.py` modified +17/-6; `python/sglang/srt/model_executor/forward_batch_info.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/communication_op.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12816 - [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative

- Link: https://github.com/sgl-project/sglang/pull/12816
- Status/date: merged / 2025-11-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `bef37d6de86a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-18, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Fixed a bug in https://github.com/sgl-project/sglang/pull/12788 ## Modifications We should only skip Indexer logits computation when forward_mode.is_extend_without...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel". The diff centers on `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py`. PR body context: ## Motivation Part1 of the fix for bug in https://github.com/sgl-project/sglang/issues/11629 ## Modifications In the topk kernel for prefill, the q and k inputs are both ragged....
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

### PR #12868 - [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill

- Link: https://github.com/sgl-project/sglang/pull/12868
- Status/date: merged / 2025-11-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `190002c613bd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Update DeepSeek V3.2 documentation to document the adaptive MHA short-sequence prefill mechanism, helping users understand the new attention pathway selection logi...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +3/-2 (5 lines); hunks: -43,15 +43,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +3/-2 (5 lines); hunks: -43,15 +43,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -43,15 +43,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
+- **Short-sequence MHA prefill (adaptive)**: For short prefill sequences (default threshold: **2048 tokens**), the NSA backend uses standard MHA automatically (no extra flags). On
-  - H200: `flashmla_sparse` prefill attention, `fa3` decode attention, `bf16` kv cache dtype.
-  - B200: `flashmla_auto` prefill attention, `flashmla_kv` decode attention, `fp8_e4m3` kv cache dtype. `flashmla_auto` enables automatic selection of either `flashmla_sparse` or
+  - H200: `flashmla_sparse` prefill attention (short-seq prefill uses MHA via FlashAttention varlen), `fa3` decode attention, `bf16` kv cache dtype.
+  - B200: `flashmla_auto` prefill attention (short-seq prefill uses MHA via TRT-LLM ragged), `flashmla_kv` decode attention, `fp8_e4m3` kv cache dtype. `flashmla_auto` enables aut
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +3/-2
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #12583 - [Deepseek V3.2] Fix accuracy bug in the Indexer

- Link: https://github.com/sgl-project/sglang/pull/12583
- Status/date: merged / 2025-11-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `0d4a41842424`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +96/-17, 231 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Deepseek V3.2] Fix accuracy bug in the Indexer". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Must merge after https://github.com/sgl-project/sglang/pull/12582 Part 2 of the fix for https://github.com/sgl-project/sglang/issues/11629 This PR also fixed the m...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `4eda9969e8b9`
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-6, 62 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[DeepseekV32]: use `_concat_mla_absorb_q_general` to replace `torch.cat`". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/11989 `torch.cat([q_nope, q_rope], dim=-1)` is heavily used in `nsa_backend`, which is less efficient. ## Modification...
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

### PR #13236 - [Deepseek V3.2] Clean up MTP

- Link: https://github.com/sgl-project/sglang/pull/13236
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `a7002e614bbd`
- Diff scope read: GitHub Pull Request files API returned 3 files, +68/-76, 248 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Deepseek V3.2] Clean up MTP". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Modifications - Fix an accuracy bug in MTP draft_extend stage. The `seqlens_int32_cpu` used in the calculations of `seqlens_expanded` is incorrect. We should use `seq_lens_cp...
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/communicator_nsa_cp.py` and 6 files; associated commits `d368c7451a48`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +1247/-54, 1729 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "(1/n)support context parallel with deepseekv3.2-DSA". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Currently, under deepseek3.2-DSA, prefill-ttft of long text sequences takes a long time. Introducing context parallel can reduce ttft. **Main design ideas：** Takin...
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
  - docs: `docs/basic_usage/deepseek_v32.md` modified +20/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_cp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13022 - [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa

- Link: https://github.com/sgl-project/sglang/pull/13022
- Status/date: merged / 2025-11-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `a8fcbf6fe3a2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +22/-1, 37 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Use torch.compile to speed up torch.cat used in prefill. ## Modifications Replace the cat ops for k in `forward_extend` in the nsa attention with the torch.compile...
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

### PR #13531 - DeepSeek V3.2 indexer RoPE fix

- Link: https://github.com/sgl-project/sglang/pull/13531
- Status/date: closed / 2025-11-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "DeepSeek V3.2 indexer RoPE fix". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/13530 ## Modifications Optimized the rotary position embedding configuration in the sparse-attention indexer to improv...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -168,7 +168,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -168,7 +168,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -168,7 +168,7 @@ def __init__(
-            is_neox_style=False,
+            is_neox_style=True,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12964 - [DeepseekV3.2] Deepseek fp8 support for MHA path

- Link: https://github.com/sgl-project/sglang/pull/12964
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `fa924410276d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +55/-9, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepseekV3.2] Deepseek fp8 support for MHA path". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Enable FP8 KV cache support for MHA (Multi-Head Attention) path in DeepSeek NSA backend. Previously, MHA could only be used when KV cache dtype is BF16, limiting i...
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `7291c72e575d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +92/-124, 345 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[Deepseek V3.2] Change indexer weights_proj to fp32". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Following https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/commit/8631a813356d39b09a5c1ee1cde8ed6015559eb3, change the indexer `weights_proj` precision from bf16 t...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunks: -109,7 +109,6 @@ def __init__(; -120,7 +119,6 @@ def __init__(; symbols: __init__, _get_logits_head_gate, touching `__init__, _get_logits_head_gate`; `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunks: -239,17 +239,6 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1226,9 +1215,6 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__, touching `add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod`; `docs/basic_usage/deepseek_v32.md` modified +66/-0 (66 lines); hunks: -129,6 +129,13 @@ Latency: 25.109 s; -143,6 +150,65 @@ Repeat: 8, mean: 0.797.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunks: -109,7 +109,6 @@ def __init__(; -120,7 +119,6 @@ def __init__(; symbols: __init__, _get_logits_head_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunks: -239,17 +239,6 @@ def add_forward_absorb_core_attention_backend(backend_name):; -1226,9 +1215,6 @@ def __init__(; symbols: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
  - `docs/basic_usage/deepseek_v32.md` modified +66/-0 (66 lines); hunks: -129,6 +129,13 @@ Latency: 25.109 s; -143,6 +150,65 @@ Repeat: 8, mean: 0.797
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
diff -- docs/basic_usage/deepseek_v32.md
@@ -129,6 +129,13 @@ Latency: 25.109 s
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53; `python/sglang/srt/models/deepseek_v2.py` modified +0/-71
  - docs: `docs/basic_usage/deepseek_v32.md` modified +66/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13544 - [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend

- Link: https://github.com/sgl-project/sglang/pull/13544
- Status/date: merged / 2025-11-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `5eed5fc0b091`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +74/-78, 342 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation NativeSparseAttnBackend currently spreads dispatch logic for NSA prefill/decode implementations and MHA vs. MLA selection across multiple places: - Global `NSA_PRE...
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

### PR #13958 - Fix nightly test failures: NSA indexer dtype and CPP radix cache init

- Link: https://github.com/sgl-project/sglang/pull/13958
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `846ba3c62c22`
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-4, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix nightly test failures: NSA indexer dtype and CPP radix cache init". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Summary Fixes two nightly test failures: 1. **test_nsa_indexer.py** - dtype mismatch error 2. **test_cpp_radix_cache.py** - server crash during initialization ## Changes ###...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-3 (7 lines); hunks: -169,9 +169,10 @@ def __init__(; symbols: __init__, _get_logits_head_gate, _get_q_k_bf16, touching `__init__, _get_logits_head_gate, _get_q_k_bf16`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-3 (7 lines); hunks: -169,9 +169,10 @@ def __init__(; symbols: __init__, _get_logits_head_gate, _get_q_k_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -169,9 +169,10 @@ def __init__(
-        weights, _ = self.weights_proj(x.float())
-        weights = weights * self.n_heads**-0.5
-        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
+        # Keep x in original dtype (bfloat16) for projection, then convert to float32
+        weights, _ = self.weights_proj(x)
+        weights = weights.float() * self.n_heads**-0.5
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/mem_cache/radix_cache_cpp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14015 - Revert "Fix nightly test failures: NSA indexer dtype and CPP radix cache init"

- Link: https://github.com/sgl-project/sglang/pull/14015
- Status/date: merged / 2025-11-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `8a9b8b8457d0`
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-5, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Revert "Fix nightly test failures: NSA indexer dtype and CPP radix cache init"". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: Reverts sgl-project/sglang#13958 This change caused dpsk v32 to break https://github.com/sgl-project/sglang/actions/runs/19694975166/job/56476095353?pr=13151 @alisonshao Please...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-4 (7 lines); hunks: -169,10 +169,9 @@ def __init__(; symbols: __init__, _get_logits_head_gate, _get_q_k_bf16, touching `__init__, _get_logits_head_gate, _get_q_k_bf16`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-4 (7 lines); hunks: -169,10 +169,9 @@ def __init__(; symbols: __init__, _get_logits_head_gate, _get_q_k_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -169,10 +169,9 @@ def __init__(
-        # Keep x in original dtype (bfloat16) for projection, then convert to float32
-        weights, _ = self.weights_proj(x)
-        weights = weights.float() * self.n_heads**-0.5
-        weights = weights.unsqueeze(-1) * q_scale.float() * self.softmax_scale
+        weights, _ = self.weights_proj(x.float())
+        weights = weights * self.n_heads**-0.5
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/mem_cache/radix_cache_cpp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13646 - [DeepSeekV3.2] Enable pure TP & Partial DP Attention

- Link: https://github.com/sgl-project/sglang/pull/13646
- Status/date: merged / 2025-11-30
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/manual/nightly/test_deepseek_v32_perf.py`; associated commits `decb48965dd1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +286/-24, 460 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeekV3.2] Enable pure TP & Partial DP Attention". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/manual/nightly/test_deepseek_v32_perf.py`. PR body context: ## Motivation DeepSeekV3.2 NSA currently has rough edges when running in **pure TP mode** (`dp_size Serving with one request: TP mode: DP Attention: ## Checklist - [x] Format yo...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunks: -330,6 +330,25 @@ def _get_topk_paged(; -409,24 +428,86 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _forward_cuda_k_only, touching `_get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunks: -170,14 +170,18 @@ def topk_transform(; -286,9 +290,11 @@ def __init__(; symbols: topk_transform, __init__, forward_extend, forward_decode, touching `topk_transform, __init__, forward_extend`; `test/manual/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: -25,6 +25,9 @@ def setUpClass(cls):; -35,6 +38,9 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: -25,6 +25,9 @@ def setUpClass(cls):; -35,6 +38,9 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunks: -330,6 +330,25 @@ def _get_topk_paged(; -409,24 +428,86 @@ def _get_topk_ragged(; symbols: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _forward_cuda_k_only
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunks: -170,14 +170,18 @@ def topk_transform(; -286,9 +290,11 @@ def __init__(; symbols: topk_transform, __init__, forward_extend, forward_decode
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: -25,6 +25,9 @@ def setUpClass(cls):; -35,6 +38,9 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunks: -25,6 +25,9 @@ def setUpClass(cls):; -35,6 +38,9 @@ def setUpClass(cls):; symbols: setUpClass
  - `docs/basic_usage/deepseek_v32.md` modified +6/-2 (8 lines); hunks: -34,15 +34,19 @@ pip3 install -e "python"
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
diff -- test/manual/nightly/test_deepseek_v32_perf.py
@@ -25,6 +25,9 @@ def setUpClass(cls):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14; `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7
  - tests: `test/manual/nightly/test_deepseek_v32_perf.py` modified +25/-0; `test/nightly/test_deepseek_v32_perf.py` modified +25/-0
  - docs: `docs/basic_usage/deepseek_v32.md` modified +6/-2
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_deepseek_v32_perf.py`, `test/nightly/test_deepseek_v32_nsabackend.py`, `test/nightly/test_deepseek_v32_perf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14245 - Fix NSA Bug in Centralize NSA Dispatch Logic

- Link: https://github.com/sgl-project/sglang/pull/14245
- Status/date: merged / 2025-12-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `1f2b84d28d63`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix NSA Bug in Centralize NSA Dispatch Logic". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Fix operator precedence bug in `set_nsa_prefill_impl` that caused `use_mha` to always be True on H200 (SM90), bypassing the `max_kv_len B`). Added parentheses to e...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-2 (5 lines); hunks: -1442,8 +1442,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[For...; symbols: set_nsa_prefill_impl, touching `set_nsa_prefill_impl`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-2 (5 lines); hunks: -1442,8 +1442,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[For...; symbols: set_nsa_prefill_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1442,8 +1442,9 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
-                device_sm == 90
-                or (device_sm >= 100 and device_sm < 110)  # SM90/SM100f only
+                (
+                    device_sm == 90 or (device_sm >= 100 and device_sm < 110)
+                )  # SM90/SM100 only
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14249 - feat: DeepSeek new v3.2 encoding

- Link: https://github.com/sgl-project/sglang/pull/14249
- Status/date: merged / 2025-12-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `7c38eca1e4a7`
- Diff scope read: GitHub Pull Request files API returned 6 files, +1153/-89, 1359 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "feat: DeepSeek new v3.2 encoding". The diff centers on `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/14227 DeepSeek official release a new encoding func to replace chat_template, and I made one workable version(though i...
- Key implementation: `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format, touching `to_json, tools_from_openai_format, tool_calls_from_openai_format`; `python/sglang/srt/function_call/deepseekv32_detector.py` added +321/-0 (321 lines); hunks: -0,0 +1,321; symbols: DeepSeekV32Detector, __init__, has_tool_call, _parse_parameters_from_xml, touching `DeepSeekV32Detector, __init__, has_tool_call`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` added +451/-0 (451 lines); hunks: -0,0 +1,451; symbols: to_json, tools_from_openai_format, tool_calls_from_openai_format, tool_calls_to_openai_format
  - `python/sglang/srt/function_call/deepseekv32_detector.py` added +321/-0 (321 lines); hunks: -0,0 +1,321; symbols: DeepSeekV32Detector, __init__, has_tool_call, _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv32.py
@@ -0,0 +1,451 @@
+# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py
+import copy
+import json
+import re
+from typing import Any, Dict, List, Optional, Tuple, Union
+TOOLS_SYSTEM_TEMPLATE = """## Tools
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -0,0 +1,321 @@
+import json
+import logging
+import re
+from typing import List
+from sglang.srt.entrypoints.openai.protocol import Tool
+from sglang.srt.function_call.base_format_detector import BaseFormatDetector
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` added +451/-0; `python/sglang/srt/function_call/deepseekv32_detector.py` added +321/-0
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`, `test/srt/openai_server/basic/test_serving_chat.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14321 - [Doc] Update DeepSeek-V3.2 document

- Link: https://github.com/sgl-project/sglang/pull/14321
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `922054079c29`
- Diff scope read: GitHub Pull Request files API returned 2 files, +76/-13, 170 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[Doc] Update DeepSeek-V3.2 document". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Following #14249 ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit....
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +74/-12 (86 lines); hunks: -1,9 +1,12; -31,7 +34,7 @@ pip3 install -e "python".
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +74/-12 (86 lines); hunks: -1,9 +1,12; -31,7 +34,7 @@ pip3 install -e "python"
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -1,9 +1,12 @@
-[DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA,
+DeepSeek-V3.2 model families equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism pow
+Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
@@ -31,7 +34,7 @@ pip3 install -e "python"
-To serve DeepSeek-V3.2-Exp on 8xH200/B200 GPUs:
+To serve [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) on 8xH200/B200 GPUs:
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +74/-12
- Risk and verification: This is mostly docs/examples in `docs/advanced_features/tool_parser.ipynb`, `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #13812 - [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels

- Link: https://github.com/sgl-project/sglang/pull/13812
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; associated commits `043f13171fb9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +896/-8, 948 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels". The diff centers on `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/13811 ## Modifications Implement fused Triton kernels that: 1. Retrieve both K and S data in a single kernel call 2. I...
- Key implementation: `test/manual/layers/attention/nsa/test_index_buf_accessor.py` added +554/-0 (554 lines); hunks: -0,0 +1,554; symbols: MockNSATokenToKVPool, __init__, create_test_buffer, TestGetK, touching `MockNSATokenToKVPool, __init__, create_test_buffer`; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +318/-2 (320 lines); hunks: -16,7 +16,7; -67,11 +67,28 @@ def torch_fast(; symbols: GetK, execute, slow, torch_fast, touching `GetK, execute, slow`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-6 (8 lines); hunks: -385,12 +385,8 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, touching `_get_topk_ragged`.
- Code diff details:
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` added +554/-0 (554 lines); hunks: -0,0 +1,554; symbols: MockNSATokenToKVPool, __init__, create_test_buffer, TestGetK
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +318/-2 (320 lines); hunks: -16,7 +16,7; -67,11 +67,28 @@ def torch_fast(; symbols: GetK, execute, slow, torch_fast
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-6 (8 lines); hunks: -385,12 +385,8 @@ def _get_topk_ragged(; symbols: _get_topk_ragged
- Key code excerpts:

```diff
diff -- test/manual/layers/attention/nsa/test_index_buf_accessor.py
@@ -0,0 +1,554 @@
+"""
+Correctness tests for NSA Indexer K/S Buffer Access with Fused Triton Kernels.
+This test verifies that the optimized Triton implementations (GetK, GetS, GetKAndS)
+produce identical results to the torch_fast baseline implementations.
+Test coverage:
+- GetK.triton() vs GetK.torch_fast()
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -16,7 +16,7 @@
-        return cls.torch_fast(*args, **kwargs)
+        return cls.triton(*args, **kwargs)
@@ -67,11 +67,28 @@ def torch_fast(
+    @classmethod
+    def triton(
+        cls, pool: "NSATokenToKVPool", buf, seq_len: int, page_indices: torch.Tensor
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -385,12 +385,8 @@ def _get_topk_ragged(
```

- Reviewed files:
  - tests: `test/manual/layers/attention/nsa/test_index_buf_accessor.py` added +554/-0
  - runtime: `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +318/-2; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-6
- Risk and verification: The diff ships test coverage in `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14332 - feat: V32 tool call parsing for no-dsml tag

- Link: https://github.com/sgl-project/sglang/pull/14332
- Status/date: open / 2025-12-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +481/-44, 660 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "feat: V32 tool call parsing for no-dsml tag". The diff centers on `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation my last pr for new v32 encoding and tool-call parsing is merged. but with openrouter's help, we found out there's a low possibility that v32 model could output too...
- Key implementation: `test/registered/function_call/test_function_call_parser.py` modified +334/-0 (334 lines); hunks: -1122,6 +1122,28 @@ def setUp(self):; -1277,6 +1299,318 @@ def test_streaming_json_format(self):; symbols: setUp, test_streaming_json_format, test_detect_and_parse_xml_format_without_dsml, test_detect_and_parse_json_format_without_dsml, touching `setUp, test_streaming_json_format, test_detect_and_parse_xml_format_without_dsml`; `python/sglang/srt/function_call/deepseekv32_detector.py` modified +147/-44 (191 lines); hunks: -20,9 +20,9 @@ class DeepSeekV32Detector(BaseFormatDetector):; -32,7 +32,7 @@ class DeepSeekV32Detector(BaseFormatDetector):; symbols: DeepSeekV32Detector, touching `DeepSeekV32Detector`.
- Code diff details:
  - `test/registered/function_call/test_function_call_parser.py` modified +334/-0 (334 lines); hunks: -1122,6 +1122,28 @@ def setUp(self):; -1277,6 +1299,318 @@ def test_streaming_json_format(self):; symbols: setUp, test_streaming_json_format, test_detect_and_parse_xml_format_without_dsml, test_detect_and_parse_json_format_without_dsml
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +147/-44 (191 lines); hunks: -20,9 +20,9 @@ class DeepSeekV32Detector(BaseFormatDetector):; -32,7 +32,7 @@ class DeepSeekV32Detector(BaseFormatDetector):; symbols: DeepSeekV32Detector
- Key code excerpts:

```diff
diff -- test/registered/function_call/test_function_call_parser.py
@@ -1122,6 +1122,28 @@ def setUp(self):
+            Tool(
+                type="function",
+                function=Function(
+                    name="get_weather",
+                    description="Get weather information for a location.",
+                    parameters={
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -20,9 +20,9 @@ class DeepSeekV32Detector(BaseFormatDetector):
-    Supports two parameter formats:
+    Supports multiple parameter formats:
-    Format 1 - XML Parameter Tags:
+    Format 1 - XML Parameter Tags (with DSML):
@@ -32,7 +32,7 @@ class DeepSeekV32Detector(BaseFormatDetector):
-    Format 2 - Direct JSON:
```

- Reviewed files:
  - tests: `test/registered/function_call/test_function_call_parser.py` modified +334/-0
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +147/-44
- Risk and verification: The diff ships test coverage in `test/registered/function_call/test_function_call_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14336 - [Doc] Fix DeepSeek V32 Doc

- Link: https://github.com/sgl-project/sglang/pull/14336
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `4bcc5879af61`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Doc] Fix DeepSeek V32 Doc". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -183,7 +183,7 @@ python3 -m sglang.test.run_eval --port 30000 --eval-name gpq...; -217,7 +217,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -183,7 +183,7 @@ python3 -m sglang.test.run_eval --port 30000 --eval-name gpq...; -217,7 +217,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -183,7 +183,7 @@ python3 -m sglang.test.run_eval --port 30000 --eval-name gpqa --num-examples 198
-Repeat: 8, mean: **0**.797
+Repeat: 8, mean: 0.797
@@ -217,7 +217,7 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
-Hardcode the thinking mode to be `thinking` in (`_apply_jinja_template`)[https://github.com/sgl-project/sglang/blob/7c38eca1e4a704bf09fe6b52ea040a41d3cfc55d/python/sglang/srt/entr
+Hardcode the thinking mode to be `thinking` in `_apply_jinja_template` function [here](https://github.com/sgl-project/sglang/blob/7c38eca1e4a704bf09fe6b52ea040a41d3cfc55d/python/s
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #14372 - [Tiny]Small fixes in deepseek v32 doc

- Link: https://github.com/sgl-project/sglang/pull/14372
- Status/date: merged / 2025-12-03
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `7e78825d5af5`
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Tiny]Small fixes in deepseek v32 doc". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +4/-2 (6 lines); hunks: -76,13 +76,15 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +4/-2 (6 lines); hunks: -76,13 +76,15 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -76,13 +76,15 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
+> Note: It is recommended to specify the chat-template, ensuring that you are within the sglang's root directory.
-  --model-path deepseek-ai/DeepSeek-V3.2 \
+  --model-path deepseek-ai/DeepSeek-V3.2-Exp \
-  --reasoning-parser deepseek-v3
+  --reasoning-parser deepseek-v3 \
+  --chat-template ./examples/chat_template/tool_chat_template_deepseekv32.jinja
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +4/-2
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #14325 - [DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200

- Link: https://github.com/sgl-project/sglang/pull/14325
- Status/date: merged / 2025-12-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `7dfcc78155b6`
- Diff scope read: GitHub Pull Request files API returned 3 files, +195/-63, 300 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepseekV3.2][NSA][Indexer] Fix PAGED top-k transform for NSA indexer chunked execution on H200". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation In `extend` mode with NSA indexer enabled, H200 setups that select **PAGED** top-k transform (`flashmla_kv` + FP8 KV cache) may trigger chunked execution in `_get_...
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

### PR #14524 - [Test] Add test suite for NSA backend

- Link: https://github.com/sgl-project/sglang/pull/14524
- Status/date: open / 2025-12-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +709/-0, 710 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Test] Add test suite for NSA backend". The diff centers on `python/sglang/test/attention/test_nsa_backend.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/14523 ## Modifications 20+ test cases covering: 1. Core forward operations - test_forward_extend - Standard extend ope...
- Key implementation: `python/sglang/test/attention/test_nsa_backend.py` added +709/-0 (709 lines); hunks: -0,0 +1,709; symbols: MockNSAConfig, __init__, MockModelRunner, TestNSABackend, touching `MockNSAConfig, __init__, MockModelRunner`.
- Code diff details:
  - `python/sglang/test/attention/test_nsa_backend.py` added +709/-0 (709 lines); hunks: -0,0 +1,709; symbols: MockNSAConfig, __init__, MockModelRunner, TestNSABackend
- Key code excerpts:

```diff
diff -- python/sglang/test/attention/test_nsa_backend.py
@@ -0,0 +1,709 @@
+import unittest
+import torch
+from sglang.srt.configs.model_config import AttentionArch
+from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
+from sglang.srt.layers.radix_attention import RadixAttention
+from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
```

- Reviewed files:
  - tests: `python/sglang/test/attention/test_nsa_backend.py` added +709/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/test_nsa_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14573 - [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode

- Link: https://github.com/sgl-project/sglang/pull/14573
- Status/date: merged / 2025-12-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `b7b7524e9560`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +144/-7, 165 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation When using `--tool-call-parser deepseekv32` in streaming mode, tool calls for functions with no parameters (e.g., `get_date()`) were being silently skipped, while...
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

### PR #14541 - [NPU]dsv3.2 cp for npu

- Link: https://github.com/sgl-project/sglang/pull/14541
- Status/date: merged / 2025-12-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`; associated commits `388018a5bd41`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +281/-134, 587 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[NPU]dsv3.2 cp for npu". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`. PR body context: ## Motivation Now, sglang already has the --enable-nsa-prefill-context-parallel option proposed to support CP parallelism, (1/n)support context parallel with deepseekv3.2-DSA ,...
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

### PR #14307 - [SMG][DS32][fix] support dsv32, add role developer

- Link: https://github.com/sgl-project/sglang/pull/14307
- Status/date: merged / 2025-12-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +36/-9, 80 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[SMG][DS32][fix] support dsv32, add role developer". The diff centers on `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs`. PR body context: Hi from novita.ai team 👋 same as #14304 add role developer for ChatMessage
- Key implementation: `sgl-model-gateway/src/protocols/chat.rs` modified +12/-9 (21 lines); hunks: -56,6 +56,14 @@ pub enum ChatMessage {; -638,7 +646,10 @@ impl GenerationRequest for ChatCompletionRequest {; `sgl-model-gateway/src/routers/grpc/harmony/builder.rs` modified +20/-0 (20 lines); hunks: -711,6 +711,26 @@ impl HarmonyBuilder {; `sgl-model-gateway/src/routers/http/pd_router.rs` modified +4/-0 (4 lines); hunks: -1132,6 +1132,10 @@ impl RouterTrait for PDRouter {.
- Code diff details:
  - `sgl-model-gateway/src/protocols/chat.rs` modified +12/-9 (21 lines); hunks: -56,6 +56,14 @@ pub enum ChatMessage {; -638,7 +646,10 @@ impl GenerationRequest for ChatCompletionRequest {
  - `sgl-model-gateway/src/routers/grpc/harmony/builder.rs` modified +20/-0 (20 lines); hunks: -711,6 +711,26 @@ impl HarmonyBuilder {
  - `sgl-model-gateway/src/routers/http/pd_router.rs` modified +4/-0 (4 lines); hunks: -1132,6 +1132,10 @@ impl RouterTrait for PDRouter {
- Key code excerpts:

```diff
diff -- sgl-model-gateway/src/protocols/chat.rs
@@ -56,6 +56,14 @@ pub enum ChatMessage {
+    #[serde(rename = "developer")]
+    Developer {
+        content: MessageContent,
+        #[serde(skip_serializing_if = "Option::is_none")]
+        tools: Option<Vec<Tool>>,
+        #[serde(skip_serializing_if = "Option::is_none")]
diff -- sgl-model-gateway/src/routers/grpc/harmony/builder.rs
@@ -711,6 +711,26 @@ impl HarmonyBuilder {
+                ChatMessage::Developer {
+                    content,
+                    name,
+                    tools: _,
+                } => {
+                    // Developer messages stay as-is
diff -- sgl-model-gateway/src/routers/http/pd_router.rs
@@ -1132,6 +1132,10 @@ impl RouterTrait for PDRouter {
```

- Reviewed files:
  - other: `sgl-model-gateway/src/protocols/chat.rs` modified +12/-9; `sgl-model-gateway/src/routers/grpc/harmony/builder.rs` modified +20/-0; `sgl-model-gateway/src/routers/http/pd_router.rs` modified +4/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #14304 - [FIX][DS32]openai protocol: support openai message role: developer

- Link: https://github.com/sgl-project/sglang/pull/14304
- Status/date: merged / 2025-12-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-3, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[FIX][DS32]openai protocol: support openai message role: developer". The diff centers on `python/sglang/srt/entrypoints/openai/protocol.py`. PR body context: Hi from novita.ai team 👋 ## Motivation support for deepseeek v3.2. example test_input_search_w_data.json got error ## Checklist - [x] Format your code according to the Format co...
- Key implementation: `python/sglang/srt/entrypoints/openai/protocol.py` modified +4/-3 (7 lines); hunks: -387,23 +387,24 @@ class ToolCall(BaseModel):; symbols: ToolCall, ChatCompletionMessageGenericParam, _normalize_role, touching `ToolCall, ChatCompletionMessageGenericParam, _normalize_role`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +4/-3 (7 lines); hunks: -387,23 +387,24 @@ class ToolCall(BaseModel):; symbols: ToolCall, ChatCompletionMessageGenericParam, _normalize_role
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/protocol.py
@@ -387,23 +387,24 @@ class ToolCall(BaseModel):
-    role: Literal["system", "assistant", "tool", "function"]
+    role: Literal["system", "assistant", "tool", "function", "developer"]
+    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
-            if v_lower not in {"system", "assistant", "tool", "function"}:
+            if v_lower not in {"system", "assistant", "tool", "function", "developer"}:
-                    "'role' must be one of 'system', 'assistant', 'tool', or 'function' (case-insensitive)."
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/protocol.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/protocol.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14572 - [NPU] optimization for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/14572
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `c05d3afb5d8b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +141/-68, 393 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[NPU] optimization for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: Co-author: @jiaming1130 ## Motivation In this PR, we optimized the performance of the DeepSeek-V3.2 on NPU to improve the performance of the network in the decode phase, and als...
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

### PR #14904 - [DeepSeek V3.2] Proper drop_thinking logic

- Link: https://github.com/sgl-project/sglang/pull/14904
- Status/date: closed / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DeepSeek V3.2] Proper drop_thinking logic". The diff centers on `python/sglang/srt/entrypoints/openai/serving_chat.py`. PR body context: ## Motivation DeepSeek's technical report describes the context management logic, which describes a mechanism for discarding thinking blocks when receiving a new message with th...
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +5/-1 (6 lines); hunks: -295,8 +295,12 @@ def _apply_jinja_template(; symbols: _apply_jinja_template, touching `_apply_jinja_template`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +5/-1 (6 lines); hunks: -295,8 +295,12 @@ def _apply_jinja_template(; symbols: _apply_jinja_template
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -295,8 +295,12 @@ def _apply_jinja_template(
+            # Historical reasoning content is discarded when a new user message is introduced
+            drop_thinking = messages[-1]["role"] == "user"
-                messages, thinking_mode=thinking_mode, drop_thinking=False
+                messages, thinking_mode=thinking_mode, drop_thinking=drop_thinking
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15064 - fix: dpskv32 chat history processing, default drop_thinking to true

- Link: https://github.com/sgl-project/sglang/pull/15064
- Status/date: merged / 2025-12-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-3, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: dpskv32 chat history processing, default drop_thinking to true". The diff centers on `python/sglang/srt/entrypoints/openai/serving_chat.py`. PR body context: ## Motivation Co-authored by: @vladnosiv As stated in #14904, the DeepSeek tech report shows a specific way for history thinking processing, so the current implementation, sets...
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +1/-3 (4 lines); hunks: -295,9 +295,7 @@ def _apply_jinja_template(; symbols: _apply_jinja_template, touching `_apply_jinja_template`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +1/-3 (4 lines); hunks: -295,9 +295,7 @@ def _apply_jinja_template(; symbols: _apply_jinja_template
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -295,9 +295,7 @@ def _apply_jinja_template(
-            real_input = encode_messages(
-                messages, thinking_mode=thinking_mode, drop_thinking=False
-            )
+            real_input = encode_messages(messages, thinking_mode=thinking_mode)
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +1/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15086 - [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache

- Link: https://github.com/sgl-project/sglang/pull/15086
- Status/date: merged / 2025-12-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `c96903074c4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-5, 52 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`. PR body context: ## Motivation Fix NSA backend assertion error when running DeepSeek-V3.2 triggered when running DeepSeek-V3.2 PP with radix-cache. The assertion `num_tokens <= total_num_tokens`...
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

### PR #15051 - feat(ds32): support tag for deepseek 3.2 tool call

- Link: https://github.com/sgl-project/sglang/pull/15051
- Status/date: closed / 2025-12-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +56/-10, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "feat(ds32): support tag for deepseek 3.2 tool call". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation #15042 ## Modifications add more match tag
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +56/-10 (66 lines); hunks: -79,7 +79,7 @@ def __init__(self):; -105,7 +105,11 @@ def _parse_parameters_from_xml(self, invoke_content: str) -...; symbols: __init__, has_tool_call, _parse_parameters_from_xml, detect_and_parse, touching `__init__, has_tool_call, _parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +56/-10 (66 lines); hunks: -79,7 +79,7 @@ def __init__(self):; -105,7 +105,11 @@ def _parse_parameters_from_xml(self, invoke_content: str) -...; symbols: __init__, has_tool_call, _parse_parameters_from_xml, detect_and_parse
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -79,7 +79,7 @@ def __init__(self):
-        return self.bot_token in text
+        return self.bot_token in text or self.bot_token.replace("｜DSML｜", "") in text
@@ -105,7 +105,11 @@ def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
-        param_matches = re.findall(self.parameter_regex, invoke_content, re.DOTALL)
+        param_matches = re.findall(
+            self.parameter_regex, invoke_content, re.DOTALL
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +56/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15217 - fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector

- Link: https://github.com/sgl-project/sglang/pull/15217
- Status/date: closed / 2025-12-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation When the buffer contains accumulated content from previous chunks (e.g., when a chunk ends with " `: - Chunk 1: "... ..." (returns only new_text, discards " ` bein...
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-3 (6 lines); hunks: -196,9 +196,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment, touching `parse_streaming_increment`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-3 (6 lines); hunks: -196,9 +196,9 @@ def parse_streaming_increment(; symbols: parse_streaming_increment
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -196,9 +196,9 @@ def parse_streaming_increment(
-                if e_token in new_text:
-                    new_text = new_text.replace(e_token, "")
-            return StreamingParseResult(normal_text=new_text)
+                if e_token in current_text:
+                    current_text = current_text.replace(e_token, "")
+            return StreamingParseResult(normal_text=current_text)
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15242 - [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations

- Link: https://github.com/sgl-project/sglang/pull/15242
- Status/date: merged / 2025-12-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations". The diff centers on `sgl-kernel/cmake/flashmla.cmake`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/15211 ## Accuracy Tests gsm8k 20shots: Accuracy: 0.955 Invalid: 0.000 Latency: 175.612 s Output throughput: 727.763 to...
- Key implementation: `sgl-kernel/cmake/flashmla.cmake` modified +1/-1 (2 lines); hunks: -4,7 +4,7 @@ include(FetchContent).
- Code diff details:
  - `sgl-kernel/cmake/flashmla.cmake` modified +1/-1 (2 lines); hunks: -4,7 +4,7 @@ include(FetchContent)
- Key code excerpts:

```diff
diff -- sgl-kernel/cmake/flashmla.cmake
@@ -4,7 +4,7 @@ include(FetchContent)
-    GIT_TAG be055fb7df0090fde45f08e9cb5b8b4c0272da73
+    GIT_TAG b1860dc4949e94241eb38503abd7b62eb62d36f6
```

- Reviewed files:
  - other: `sgl-kernel/cmake/flashmla.cmake` modified +1/-1
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #15088 - [DeepSeekV3.2] Add pure TP+MTP test

- Link: https://github.com/sgl-project/sglang/pull/15088
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `2bdbaef18e5f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +107/-7, 167 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeekV3.2] Add pure TP+MTP test". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Add test coverage for DeepSeek V3.2 with pure tensor parallel mode and MTP (EAGLE speculative decoding) without DP attention. This addresses the "Support pure TP+M...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +7/-1 (8 lines); hunks: -64,10 +64,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +7/-1 (8 lines); hunks: -64,10 +64,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -64,10 +64,16 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
-Example usage:
+Example usage with DP Attention:
+Example usage with Pure TP:
+'''bash
+python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-dr
+'''
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +7/-1
- Risk and verification: The diff ships test coverage in `test/nightly/test_deepseek_v32_tp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15322 - dsv32 support o_proj tp

- Link: https://github.com/sgl-project/sglang/pull/15322
- Status/date: open / 2025-12-17
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +472/-23, 804 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "dsv32 support o_proj tp". The diff centers on `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/linear.py`. PR body context: ## Motivation In order to improve the performance of DeepSeek-V3.2, we enabled the `o_proj` layers to support tensor parallel and tensor parallel independently of `attn_tp`. ##...
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +179/-5 (184 lines); hunks: -21,20 +21,32; -47,6 +59,7; symbols: enable_moe_dense_fully_dp, get_max_bs_across_dp, LayerCommunicator, __init__, touching `enable_moe_dense_fully_dp, get_max_bs_across_dp, LayerCommunicator`; `python/sglang/srt/models/deepseek_v2.py` modified +74/-14 (88 lines); hunks: -45,6 +45,9; -70,6 +73,7; symbols: forward_deepep, __init__, dispatch_attn_forward_method, touching `forward_deepep, __init__, dispatch_attn_forward_method`; `python/sglang/srt/layers/linear.py` modified +73/-1 (74 lines); hunks: -11,6 +11,7; -1326,6 +1327,7 @@ def weight_loader(self, param: Parameter, loaded_weight: t...; symbols: weight_loader, extra_repr, TP2DPandTPRowParallelLinear, __init__, touching `weight_loader, extra_repr, TP2DPandTPRowParallelLinear`; `python/sglang/srt/layers/rotary_embedding.py` modified +8/-0 (8 lines); hunks: -949,6 +949,14 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +179/-5 (184 lines); hunks: -21,20 +21,32; -47,6 +59,7; symbols: enable_moe_dense_fully_dp, get_max_bs_across_dp, LayerCommunicator, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-14 (88 lines); hunks: -45,6 +45,9; -70,6 +73,7; symbols: forward_deepep, __init__, dispatch_attn_forward_method
  - `python/sglang/srt/layers/linear.py` modified +73/-1 (74 lines); hunks: -11,6 +11,7; -1326,6 +1327,7 @@ def weight_loader(self, param: Parameter, loaded_weight: t...; symbols: weight_loader, extra_repr, TP2DPandTPRowParallelLinear, __init__
  - `python/sglang/srt/layers/rotary_embedding.py` modified +8/-0 (8 lines); hunks: -949,6 +949,14 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-0 (8 lines); hunks: -770,6 +770,11 @@ def replay_prepare(; -782,6 +787,9 @@ def replay_prepare(; symbols: replay_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -21,20 +21,32 @@
+    get_o_proj_data_parallel_rank,
+    get_o_proj_data_parallel_world_size,
+    get_o_proj_dp_group,
+    get_o_proj_tensor_parallel_rank,
+    get_o_proj_tensor_parallel_world_size,
+    get_o_proj_tp_group,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -45,6 +45,9 @@
+    get_o_proj_data_parallel_world_size,
+    get_o_proj_tensor_parallel_rank,
+    get_o_proj_tensor_parallel_world_size,
@@ -70,6 +73,7 @@
+    BeforeOproj,
@@ -87,6 +91,7 @@
diff -- python/sglang/srt/layers/linear.py
@@ -11,6 +11,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +179/-5; `python/sglang/srt/models/deepseek_v2.py` modified +74/-14; `python/sglang/srt/layers/linear.py` modified +73/-1; `python/sglang/srt/layers/rotary_embedding.py` modified +8/-0; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +8/-0; `python/sglang/srt/model_executor/forward_batch_info.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/communication_op.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/environ.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15307 - [Deepseek V3.2] Support Overlap Spec + NSA

- Link: https://github.com/sgl-project/sglang/pull/15307
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `d20699a33c50`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +25/-8, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Deepseek V3.2] Support Overlap Spec + NSA". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Part of V3.2 Roadmap https://github.com/sgl-project/sglang/issues/15025 Enable overlap spec and EAGLE + NSA backend. ## Modifications In EAGLE V1, we had (with `py...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunks: -389,7 +389,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -422,9 +422,20 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph, touching `init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunks: -295,7 +295,7 @@ def _get_topk_paged(; -900,7 +900,7 @@ def forward_cuda(; symbols: _get_topk_paged, forward_cuda, touching `_get_topk_paged, forward_cuda`; `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunks: -71,6 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V....
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunks: -389,7 +389,7 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -422,9 +422,20 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunks: -295,7 +295,7 @@ def _get_topk_paged(; -900,7 +900,7 @@ def forward_cuda(; symbols: _get_topk_paged, forward_cuda
  - `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunks: -71,6 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V...
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
diff -- docs/basic_usage/deepseek_v32.md
@@ -71,6 +71,10 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
+'''{tip}
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2
  - docs: `docs/basic_usage/deepseek_v32.md` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15278 - feat: DeepSeek-V3.2 Streaming tool call output

- Link: https://github.com/sgl-project/sglang/pull/15278
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `41683536d394`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +111/-69, 328 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "feat: DeepSeek-V3.2 Streaming tool call output". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation FIxes #14711 ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [...
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

### PR #14781 - [Performance] optimize NSA backend metadata computation for multi-step speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/14781
- Status/date: merged / 2025-12-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `e0026f7c92c9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +440/-16, 502 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Performance] optimize NSA backend metadata computation for multi-step speculative decoding". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`. PR body context: The NSA backend for multi-step speculative decoding. The optimization implements a precompute-once-copy-many strategy: - Old approach: Compute metadata N times (one per backend)...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` added +324/-0 (324 lines); hunks: -0,0 +1,324; symbols: PrecomputedMetadata, compute_cu_seqlens, NativeSparseAttnBackendMTPPrecomputeMixin, providing, touching `PrecomputedMetadata, compute_cu_seqlens, NativeSparseAttnBackendMTPPrecomputeMixin`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +111/-16 (127 lines); hunks: -10,13 +10,19; -224,17 +230,12 @@ def topk_transform(; symbols: topk_transform, compute_cu_seqlens, NativeSparseAttnBackend, __init__, touching `topk_transform, compute_cu_seqlens, NativeSparseAttnBackend`; `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-0 (5 lines); hunks: -19,6 +19,11; symbols: print_nsa_bool_env_vars, touching `print_nsa_bool_env_vars`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` added +324/-0 (324 lines); hunks: -0,0 +1,324; symbols: PrecomputedMetadata, compute_cu_seqlens, NativeSparseAttnBackendMTPPrecomputeMixin, providing
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +111/-16 (127 lines); hunks: -10,13 +10,19; -224,17 +230,12 @@ def topk_transform(; symbols: topk_transform, compute_cu_seqlens, NativeSparseAttnBackend, __init__
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-0 (5 lines); hunks: -19,6 +19,11; symbols: print_nsa_bool_env_vars
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py
@@ -0,0 +1,324 @@
+"""Multi-step precompute utilities for Native Sparse Attention backend.
+This module provides optimization utilities for multi-step speculative decoding
+by precomputing shared metadata once and copying it to multiple backend instances.
+"""
+from __future__ import annotations
+from dataclasses import dataclass
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -10,13 +10,19 @@
+from sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute import (
+    NativeSparseAttnBackendMTPPrecomputeMixin,
+    PrecomputedMetadata,
+    compute_cu_seqlens,
+)
+    NSA_ENABLE_MTP_PRECOMPUTE_METADATA,
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -19,6 +19,11 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` added +324/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +111/-16; `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14353 - feat(dsv32): better error handling for DeepSeek-v3.2 encoder

- Link: https://github.com/sgl-project/sglang/pull/14353
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`; associated commits `216067c0cba2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +53/-32, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "feat(dsv32): better error handling for DeepSeek-v3.2 encoder". The diff centers on `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`. PR body context: Hi from novita.ai team 👋 ## Motivation when users' input illegal and cause error in deepseek3.2 encoder, return 400 instead of 500 ## Modifications add a kind of exception for d...
- Key implementation: `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` modified +45/-32 (77 lines); hunks: -4,6 +4,11; -148,11 +153,12 @@ def find_last_user_index(messages: List[Dict[str, Any]]) -...; symbols: DS32EncodingError, find_last_user_index, render_message, touching `DS32EncodingError, find_last_user_index, render_message`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` modified +45/-32 (77 lines); hunks: -4,6 +4,11; -148,11 +153,12 @@ def find_last_user_index(messages: List[Dict[str, Any]]) -...; symbols: DS32EncodingError, find_last_user_index, render_message
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/encoding_dsv32.py
@@ -4,6 +4,11 @@
+class DS32EncodingError(Exception):
+    pass
@@ -148,11 +153,12 @@ def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
-    assert 0 <= index < len(messages)
-    assert thinking_mode in [
-        "chat",
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` modified +45/-32
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15429 - [Deepseek V3.2] Fix Deepseek MTP in V1 mode

- Link: https://github.com/sgl-project/sglang/pull/15429
- Status/date: merged / 2025-12-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `e88e75a9dfdd`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Deepseek V3.2] Fix Deepseek MTP in V1 mode". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation ## Modifications Restore the earlier behaviour of Deepseek V3 + MTP. (Accidental change) Fix https://github.com/sgl-project/sglang/issues/15428
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/15025 ## Changes - Add `paged_mqa_schedule_metadata` to `NSAMetadata` (batch-level caching). - Compute once in `init_f...
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

### PR #14901 - fix ds3.2 nsa backend prefill TBO

- Link: https://github.com/sgl-project/sglang/pull/14901
- Status/date: merged / 2025-12-21
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `350fbbf4dc1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +76/-2, 148 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "fix ds3.2 nsa backend prefill TBO". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation After applying the fixes, tbo can be enabled on ds3.2. And resolving • https://github.com/sgl-project/sglang/issues/12698 • https://github.com/sgl-project/sglang/i...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: -66,6 +66,7; -421,7 +422,10 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, _get_mla_kv_buffer_from_fp8, touching `handle_attention_nsa, _get_mla_kv_buffer_from_fp8`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: -66,6 +66,7; -421,7 +422,10 @@ def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_nsa, _get_mla_kv_buffer_from_fp8
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -66,6 +66,7 @@
+from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
@@ -421,7 +422,10 @@ def handle_attention_nsa(attn, forward_batch):
+    if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
+        backend = backend.primary
@@ -2665,7 +2669,10 @@ def _get_mla_kv_buffer_from_fp8(
-        kv_indices = forward_batch.attn_backend.forward_metadata.page_table_1_flattened
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `test/srt/ep/test_deepep_large.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14741 - [1/N][Sparse With Hicache]: Add Sparse Interface

- Link: https://github.com/sgl-project/sglang/pull/14741
- Status/date: merged / 2025-12-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +642/-0, 646 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[1/N][Sparse With Hicache]: Add Sparse Interface". The diff centers on `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py`. PR body context: ## Motivation This PR primarily introduces a generic Retrievable Sparse Algorithm interface. Upstream PR：https://github.com/sgl-project/sglang/pull/14619 ## Modifications ## Acc...
- Key implementation: `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunks: -0,0 +1,383; symbols: BaseSparseAlgorithm, for, provides, __init__, touching `BaseSparseAlgorithm, for, provides`; `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: QuestAlgorithm, __init__, _initialize_representation_pools, _compute_page_representations, touching `QuestAlgorithm, __init__, _initialize_representation_pools`; `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: DeepSeekNSAAlgorithm, __init__, retrieve_topk, initialize_representation_pool, touching `DeepSeekNSAAlgorithm, __init__, retrieve_topk`; `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py` added +13/-0 (13 lines); hunks: -0,0 +1,13.
- Code diff details:
  - `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunks: -0,0 +1,383; symbols: BaseSparseAlgorithm, for, provides, __init__
  - `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` added +166/-0 (166 lines); hunks: -0,0 +1,166; symbols: QuestAlgorithm, __init__, _initialize_representation_pools, _compute_page_representations
  - `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` added +80/-0 (80 lines); hunks: -0,0 +1,80; symbols: DeepSeekNSAAlgorithm, __init__, retrieve_topk, initialize_representation_pool
  - `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py` added +13/-0 (13 lines); hunks: -0,0 +1,13
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py
@@ -0,0 +1,383 @@
+from abc import ABC, abstractmethod
+from typing import TYPE_CHECKING
+import torch
+if TYPE_CHECKING:
+    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+class BaseSparseAlgorithm(ABC):
diff -- python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py
@@ -0,0 +1,166 @@
+"""
+Quest sparse attention algorithm.
+This implementation follows the Quest paper's bounding-box estimation for
+query-aware page selection. For each KV page, it maintains per-dimension
+min/max of keys and uses them to upper-bound attention scores without
+materializing full dot products.
diff -- python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py
@@ -0,0 +1,80 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0; `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` added +166/-0; `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` added +80/-0; `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py` added +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14750 - [Tool Call][DSV32] Streamline function call parameters

- Link: https://github.com/sgl-project/sglang/pull/14750
- Status/date: merged / 2025-12-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/function_call/deepseekv32_detector.py`; associated commits `01bd0d3e8b8c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-29, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Tool Call][DSV32] Streamline function call parameters". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation As titled and Issue #14711. **IMPORTANT: This is an initial version aimed at streamlining function call parameters for `DeepSeek-V3.2`. I have tested it, and it pa...
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

### PR #16119 - [cp] bug fix for dsv3.2 cp

- Link: https://github.com/sgl-project/sglang/pull/16119
- Status/date: merged / 2025-12-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[cp] bug fix for dsv3.2 cp". The diff centers on `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -338,7 +338,7 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, touching `forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunks: -338,7 +338,7 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py
@@ -338,7 +338,7 @@ def forward_dsa_prepare_npu(
-        k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
+        k_nope = m.kv_a_layernorm(k_nope)
```

- Reviewed files:
  - runtime: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16148 - [Fix] DSV3.2 W4AFP8 MTP use FP8 draft model

- Link: https://github.com/sgl-project/sglang/pull/16148
- Status/date: open / 2025-12-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-1, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] DSV3.2 W4AFP8 MTP use FP8 draft model". The diff centers on `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/quantization/w4afp8.py`. PR body context: ## Motivation DeepSeekV3.2 W4A8 MTP acceptance rate is very low use this commad: `SGLANG_DEEPEP_BF16_DISPATCH=1 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION...
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-1 (3 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; -634,6 +634,7 @@ def _dispatch_core(; symbols: _dispatch_core, combine_a, touching `_dispatch_core, combine_a`; `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-0 (1 lines); hunks: -283,6 +283,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading, create_moe_runner, touching `process_weights_after_loading, create_moe_runner`.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-1 (3 lines); hunks: -609,7 +609,7 @@ def _dispatch_core(; -634,6 +634,7 @@ def _dispatch_core(; symbols: _dispatch_core, combine_a
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-0 (1 lines); hunks: -283,6 +283,7 @@ def process_weights_after_loading(self, layer: Module) -> None:; symbols: process_weights_after_loading, create_moe_runner
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/deepep.py
@@ -609,7 +609,7 @@ def _dispatch_core(
-        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
+        elif not envs.SGLANG_DEEPEP_BF16_DISPATCH.get() or self.quant_config.get("dispatch_weight_scale") is None:
@@ -634,6 +634,7 @@ def _dispatch_core(
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -283,6 +283,7 @@ def process_weights_after_loading(self, layer: Module) -> None:
+        layer.dispatcher.set_quant_config({"dispatch_weight_scale", (layer.w13_weight_scale_inv)})
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +2/-1; `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16156 - [cp] assert dsv3.2 cp in pd decode mode

- Link: https://github.com/sgl-project/sglang/pull/16156
- Status/date: merged / 2025-12-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[cp] assert dsv3.2 cp in pd decode mode". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/server_args.py` modified +4/-0 (4 lines); hunks: -1093,6 +1093,10 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +4/-0 (4 lines); hunks: -1093,6 +1093,10 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -1093,6 +1093,10 @@ def _handle_model_specific_adjustments(self):
+                if self.enable_nsa_prefill_context_parallel:
+                    assert (
+                        self.disaggregation_mode != "decode"
+                    ), "CP is only supported for prefill when PD disaggregation, please remove --enable-nsa-prefill-context-parallel."
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13959 - [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache

- Link: https://github.com/sgl-project/sglang/pull/13959
- Status/date: merged / 2026-01-02
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py` and 7 files; associated commits `0d244116d28a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +603/-264, 1414 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation The original default token splitting scheme of cp does not support prefill multi-batch. A new token splitting method is introduced to enable multi-batch support, f...
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
  - docs: `docs/basic_usage/deepseek_v32.md` modified +12/-0
- Risk and verification: The diff ships test coverage in `test/manual/test_deepseek_v32_cp_single_node.py`, `test/srt/test_prefill_adder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16305 - Multiple updates of DeepSeek V32 and context parallel

- Link: https://github.com/sgl-project/sglang/pull/16305
- Status/date: merged / 2026-01-02
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `f07e76b229db`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +190/-35, 308 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Multiple updates of DeepSeek V32 and context parallel". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation - Refine the handler in `server_args.py` - Move dpsk v32 cp test to pr-test, and add pure tp tests - Update the document for better readibility ## Modifications ##...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +30/-21 (51 lines); hunks: -274,31 +274,40 @@ DeepSeek-V3.2-Speciale:.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +30/-21 (51 lines); hunks: -274,31 +274,40 @@ DeepSeek-V3.2-Speciale:
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -274,31 +274,40 @@ DeepSeek-V3.2-Speciale:
-Accuracy benchmark on long context can be tested on GPQA-diamond dataset with long output tokens and thinking enabled:
+**Note: This feature is only verified on Hopper machines**
-Example usage:
+For context parallel in DeepSeek V3.2 model, we provide two different modes of splitting tokens, which can be controlled with argument `--nsa-prefill-cp-mode`.
+### In sequence splitting (default setting)
+The first mode can be enabled by `--nsa-prefill-cp-mode in-seq-split`. This mode implements context parallel for DSA by splitting the sequence uniformly between context parallel r
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +30/-21
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_basic.py`, `test/srt/test_deepseek_v32_cp_single_node.py`, `test/srt/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15310 - [Deepseek V3.2] Enable TRTLLM Allreduce Fusion

- Link: https://github.com/sgl-project/sglang/pull/15310
- Status/date: closed / 2026-01-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Deepseek V3.2] Enable TRTLLM Allreduce Fusion". The diff centers on `python/sglang/srt/server_args.py`. PR body context: This should close some of the gap in TP MoE (around 3-4%). TODO: provide perf numbers
- Key implementation: `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1530,6 +1530,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunks: -1530,6 +1530,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -1530,6 +1530,7 @@ def _handle_model_specific_adjustments(self):
+                "DeepseekV32ForCausalLM",
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16520 - fix: unimplemented methods in BaseIndexerMetadata

- Link: https://github.com/sgl-project/sglang/pull/16520
- Status/date: merged / 2026-01-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +39/-1, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "fix: unimplemented methods in BaseIndexerMetadata". The diff centers on `test/registered/kernels/test_nsa_indexer.py`. PR body context: ## Motivation On Jan 2, commit 0d244116d introduced a breaking change as part of the Context Parallelism optimization for DeepSeek V3.2. This commit added 3 new non-abstract met...
- Key implementation: `test/registered/kernels/test_nsa_indexer.py` modified +39/-1 (40 lines); hunks: -1,5 +1,5; -89,11 +89,49 @@ def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx, touching `get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu`.
- Code diff details:
  - `test/registered/kernels/test_nsa_indexer.py` modified +39/-1 (40 lines); hunks: -1,5 +1,5; -89,11 +89,49 @@ def get_seqlens_expanded(self) -> torch.Tensor:; symbols: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
- Key code excerpts:

```diff
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -1,5 +1,5 @@
-from typing import Optional
+from typing import Optional, Tuple
@@ -89,11 +89,49 @@ def get_seqlens_expanded(self) -> torch.Tensor:
+    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
+        """
+        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache for each token.
```

- Reviewed files:
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +39/-1
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15938 - Clean Some Environment Variables for DeepSeek V32

- Link: https://github.com/sgl-project/sglang/pull/15938
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `7d757d6f17ef`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +39/-108, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Clean Some Environment Variables for DeepSeek V32". The diff centers on `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`. PR body context: ## Motivation ## Modifications - Move `SGLANG_NSA_FUSE_TOPK` and `SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA` to environ.py - Deprecate `SGLANG_NSA_DUAL_STREAM`, `SGLANG_NSA_FLAS...
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

### PR #16306 - [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition

- Link: https://github.com/sgl-project/sglang/pull/16306
- Status/date: merged / 2026-01-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +255/-228, 536 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +18/-228 (246 lines); hunks: -21,7 +21,6; -109,7 +108,6; symbols: add_forward_absorb_core_attention_backend, AttnForwardMethod, _dispatch_mla_subtype, AttentionBackendRegistry, touching `add_forward_absorb_core_attention_backend, AttnForwardMethod, _dispatch_mla_subtype`; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: AttentionBackendRegistry, register, get_handler, _dispatch_mla_subtype, touching `AttentionBackendRegistry, register, get_handler`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: AttnForwardMethod, touching `AttnForwardMethod`; `python/sglang/srt/models/deepseek_common/utils.py` added +23/-0 (23 lines); hunks: -0,0 +1,23.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-228 (246 lines); hunks: -21,7 +21,6; -109,7 +108,6; symbols: add_forward_absorb_core_attention_backend, AttnForwardMethod, _dispatch_mla_subtype, AttentionBackendRegistry
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` added +182/-0 (182 lines); hunks: -0,0 +1,182; symbols: AttentionBackendRegistry, register, get_handler, _dispatch_mla_subtype
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: AttnForwardMethod
  - `python/sglang/srt/models/deepseek_common/utils.py` added +23/-0 (23 lines); hunks: -0,0 +1,23
  - `python/sglang/srt/models/deepseek_common/__init__.py` added +0/-0 (0 lines)
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -21,7 +21,6 @@
-from enum import IntEnum, auto
@@ -109,7 +108,6 @@
-    is_fp8_fnuz,
@@ -138,40 +136,39 @@
+from sglang.srt.models.deepseek_common.attention_backend_handler import (
+    AttentionBackendRegistry,
diff -- python/sglang/srt/models/deepseek_common/attention_backend_handler.py
@@ -0,0 +1,182 @@
+from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
+from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
+from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
+    AttnForwardMethod,
+)
+from sglang.srt.models.deepseek_common.utils import _is_hip
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py
@@ -0,0 +1,32 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +18/-228; `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` added +182/-0; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` added +32/-0; `python/sglang/srt/models/deepseek_common/utils.py` added +23/-0; `python/sglang/srt/models/deepseek_common/__init__.py` added +0/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/__init__.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16380 - [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled

- Link: https://github.com/sgl-project/sglang/pull/16380
- Status/date: merged / 2026-01-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +72/-36, 198 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`. PR body context: ## Motivation For issue #15358. CC @whybeyoung Support pipeline parallelism for the Prefill CP scenario (`--enable-nsa-prefill-context-parallel`). In this scenario, the linear l...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +57/-33 (90 lines); hunks: -1,5 +1,6; -23,6 +24,7; symbols: __init__, _with_real_sm_count, _get_logits_head_gate, _get_topk_paged, touching `__init__, _with_real_sm_count, _get_logits_head_gate`; `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +15/-3 (18 lines); hunks: -504,6 +504,10 @@ def event_loop_pp_disagg_decode(self: Scheduler):; -906,7 +910,9 @@ def _pp_send_dict_to_next_stage(; symbols: event_loop_pp_disagg_decode, init_pp_loop_state, _pp_send_dict_to_next_stage, _pp_recv_proxy_tensors, touching `event_loop_pp_disagg_decode, init_pp_loop_state, _pp_send_dict_to_next_stage`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +57/-33 (90 lines); hunks: -1,5 +1,6; -23,6 +24,7; symbols: __init__, _with_real_sm_count, _get_logits_head_gate, _get_topk_paged
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +15/-3 (18 lines); hunks: -504,6 +504,10 @@ def event_loop_pp_disagg_decode(self: Scheduler):; -906,7 +910,9 @@ def _pp_send_dict_to_next_stage(; symbols: event_loop_pp_disagg_decode, init_pp_loop_state, _pp_send_dict_to_next_stage, _pp_recv_proxy_tensors
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -1,5 +1,6 @@
+import contextlib
@@ -23,6 +24,7 @@
+from sglang.srt.distributed.parallel_state import get_pp_group
@@ -146,6 +148,10 @@ def __init__(
+            pp_size = get_global_server_args().pp_size
+            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
diff -- python/sglang/srt/managers/scheduler_pp_mixin.py
@@ -504,6 +504,10 @@ def event_loop_pp_disagg_decode(self: Scheduler):
+        # In CP mode, attention weights are duplicated, eliminating the need for the attention TP all-gather operation.
+        self.require_attn_tp_allgather = (
+            not self.server_args.enable_nsa_prefill_context_parallel
+        )
@@ -906,7 +910,9 @@ def _pp_send_dict_to_next_stage(
-                all_gather_group=self.attn_tp_group,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +57/-33; `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +15/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16637 - [DSv32] Overlap indexer weights_proj during dual_stream decode

- Link: https://github.com/sgl-project/sglang/pull/16637
- Status/date: merged / 2026-01-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `20abaee26cd8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +64/-22, 134 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DSv32] Overlap indexer weights_proj during dual_stream decode". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation The weights_proj in the DSA indexer uses float type, which is slow on modern GPUs. Profiler traces show this accounts for ~20% of decode layer runtime. The traces...
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

### PR #16881 - [DSv32] Add returning DSA topk indices

- Link: https://github.com/sgl-project/sglang/pull/16881
- Status/date: closed / 2026-01-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +205/-2, 599 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSv32] Add returning DSA topk indices". The diff centers on `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation https://github.com/sgl-project/sglang/issues/16856 ## Modifications Incrementally add DSA topk indices capturing on top of https://github.com/sgl-project/sglang/pu...
- Key implementation: `python/sglang/srt/layers/moe/routed_experts_capturer.py` modified +118/-0 (118 lines); hunks: -34,6 +34,8 @@ def __init__(; -48,23 +50,56 @@ def __init__(; symbols: __init__, get_buffer_size_bytes, get_dsa_topk_indices_buffer_size_bytes, capture_fwd_routed_experts, touching `__init__, get_buffer_size_bytes, get_dsa_topk_indices_buffer_size_bytes`; `python/sglang/srt/managers/detokenizer_manager.py` modified +17/-2 (19 lines); hunks: -328,6 +328,7 @@ def _decode_batch_token_id_output(self, recv_obj: BatchToken...; -339,7 +340,18 @@ def _extract_routed_experts(self, recv_obj: BatchTokenIDOut...; symbols: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out, touching `_decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out`; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -97,6 +97,7; -1857,6 +1858,11 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, touching `forward_absorb_prepare`; `python/sglang/srt/managers/multi_tokenizer_mixin.py` modified +3/-0 (3 lines); hunks: -281,6 +281,9 @@ def _handle_output_by_index(output, i):; symbols: _handle_output_by_index, touching `_handle_output_by_index`.
- Code diff details:
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` modified +118/-0 (118 lines); hunks: -34,6 +34,8 @@ def __init__(; -48,23 +50,56 @@ def __init__(; symbols: __init__, get_buffer_size_bytes, get_dsa_topk_indices_buffer_size_bytes, capture_fwd_routed_experts
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +17/-2 (19 lines); hunks: -328,6 +328,7 @@ def _decode_batch_token_id_output(self, recv_obj: BatchToken...; -339,7 +340,18 @@ def _extract_routed_experts(self, recv_obj: BatchTokenIDOut...; symbols: _decode_batch_token_id_output, _extract_routed_experts, handle_batch_token_id_out
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -97,6 +97,7; -1857,6 +1858,11 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
  - `python/sglang/srt/managers/multi_tokenizer_mixin.py` modified +3/-0 (3 lines); hunks: -281,6 +281,9 @@ def _handle_output_by_index(output, i):; symbols: _handle_output_by_index
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +3/-0 (3 lines); hunks: -926,6 +926,7 @@ def _create_tokenized_object(; -1523,6 +1524,8 @@ def _handle_batch_output(; symbols: _create_tokenized_object, _handle_batch_output
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/routed_experts_capturer.py
@@ -34,6 +34,8 @@ def __init__(
+        enable_capture_dsa_topk_indices: bool = False,
+        num_dsa_topk_indices: int = 2048,
@@ -48,23 +50,56 @@ def __init__(
+        self.dsa_topk_indices_buffer = None
+        if enable_capture_dsa_topk_indices:
+            self.dsa_topk_indices_buffer = torch.zeros(
diff -- python/sglang/srt/managers/detokenizer_manager.py
@@ -328,6 +328,7 @@ def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
+        output_dsa_topk_indices = None
@@ -339,7 +340,18 @@ def _extract_routed_experts(self, recv_obj: BatchTokenIDOutput) -> List[List[int
-        return output_routed_experts
+        if recv_obj.output_dsa_topk_indices is not None:
+            output_dsa_topk_indices = [
+                (
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -97,6 +97,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/routed_experts_capturer.py` modified +118/-0; `python/sglang/srt/managers/detokenizer_manager.py` modified +17/-2; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0; `python/sglang/srt/managers/multi_tokenizer_mixin.py` modified +3/-0; `python/sglang/srt/managers/tokenizer_manager.py` modified +3/-0; `python/sglang/srt/entrypoints/engine.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16916 - add doc for dsv32 cp+pp

- Link: https://github.com/sgl-project/sglang/pull/16916
- Status/date: merged / 2026-01-12
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `aab640c99f22`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +114/-0, 125 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "add doc for dsv32 cp+pp". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation add doc for dsv32 cp+pp
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +114/-0 (114 lines); hunks: -186,6 +186,7 @@ Latency: 29.545 s; -321,3 +322,116 @@ Example usage:.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +114/-0 (114 lines); hunks: -186,6 +186,7 @@ Latency: 29.545 s; -321,3 +322,116 @@ Example usage:
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -186,6 +186,7 @@ Latency: 29.545 s
@@ -321,3 +322,116 @@ Example usage:
+### Pipeline Parallel + Context Parallel (PP + CP)
+This mode combines Pipeline Parallelism (PP) and Context Parallelism (CP) to scale across multiple nodes, which can achieve better throughput and Time To First Token (TTFT). Note
+#### Standard Usage
+To launch with PP=2 and CP (via `round-robin-split` mode) on 2 nodes. This configuration uses the fused MoE kernel by default, which generally provides better performance.
+For related development details, please refer to:
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +114/-0
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #16990 - [Ascend] fix dsv3.2 weight cast bug

- Link: https://github.com/sgl-project/sglang/pull/16990
- Status/date: merged / 2026-01-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Ascend] fix dsv3.2 weight cast bug". The diff centers on `python/sglang/srt/layers/quantization/unquant.py`. PR body context: Add NPU format casting for layer weights in processing. ## Motivation fix the bug `NPU function error: call aclnnMatmulWeightNz failed` when loading `deepseekv3.2 `weight. ## Mo...
- Key implementation: `python/sglang/srt/layers/quantization/unquant.py` modified +3/-2 (5 lines); hunks: -122,8 +122,6 @@ def create_weights(; -235,6 +233,9 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: create_weights, process_weights_after_loading, apply, touching `create_weights, process_weights_after_loading, apply`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +3/-2 (5 lines); hunks: -122,8 +122,6 @@ def create_weights(; -235,6 +233,9 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: create_weights, process_weights_after_loading, apply
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/unquant.py
@@ -122,8 +122,6 @@ def create_weights(
-        if _is_npu:
-            layer.weight.data = npu_format_cast(layer.weight.data)
@@ -235,6 +233,9 @@ def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
+        if is_npu():
+            layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
+            layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/unquant.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/unquant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16841 - [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant

- Link: https://github.com/sgl-project/sglang/pull/16841
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `afe285f7bd9a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +260/-81, 587 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`. PR body context: Co-author: @wufann ## Motivation This PR enables cuda graph (`--cuda-graph-max-bs 64`) and fixes crashes/memory faults observed when running DeepSeek-V3.2-Exp with NSA (tilelang...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +171/-55 (226 lines); hunks: -12,14 +12,16; -42,7 +44,8; symbols: BaseIndexerMetadata, get_page_table_64, get_page_table_1, get_seqlens_expanded, touching `BaseIndexerMetadata, get_page_table_64, get_page_table_1`; `python/sglang/srt/models/deepseek_v2.py` modified +58/-18 (76 lines); hunks: -1523,10 +1523,34 @@ def forward_normal_prepare(; -1703,31 +1727,47 @@ def forward_absorb_prepare(; symbols: forward_normal_prepare, forward_absorb_prepare, touching `forward_normal_prepare, forward_absorb_prepare`; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +10/-3 (13 lines); hunks: -4,6 +4,8; -347,12 +349,17 @@ def _set_k_and_s_triton(; symbols: _set_k_and_s_triton, touching `_set_k_and_s_triton`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunks: -174,6 +174,9 @@ def get_seqlens_int32(self) -> torch.Tensor:; symbols: get_seqlens_int32, get_page_table_64, get_page_table_1, get_seqlens_expanded, touching `get_seqlens_int32, get_page_table_64, get_page_table_1`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +171/-55 (226 lines); hunks: -12,14 +12,16; -42,7 +44,8; symbols: BaseIndexerMetadata, get_page_table_64, get_page_table_1, get_seqlens_expanded
  - `python/sglang/srt/models/deepseek_v2.py` modified +58/-18 (76 lines); hunks: -1523,10 +1523,34 @@ def forward_normal_prepare(; -1703,31 +1727,47 @@ def forward_absorb_prepare(; symbols: forward_normal_prepare, forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +10/-3 (13 lines); hunks: -4,6 +4,8; -347,12 +349,17 @@ def _set_k_and_s_triton(; symbols: _set_k_and_s_triton
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunks: -174,6 +174,9 @@ def get_seqlens_int32(self) -> torch.Tensor:; symbols: get_seqlens_int32, get_page_table_64, get_page_table_1, get_seqlens_expanded
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +2/-0 (2 lines); hunks: -147,6 +147,8 @@ def fp8_index_kernel_(; symbols: fp8_index_kernel_
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -12,14 +12,16 @@
-if is_cuda():
+_is_cuda = is_cuda()
+_is_hip = is_hip()
+_is_npu = is_npu()
+if _is_cuda:
-if is_npu():
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1523,10 +1523,34 @@ def forward_normal_prepare(
-                q_lora = self.q_a_layernorm(q)
-                q = self.q_b_proj(q_lora)[0].view(
-                    -1, self.num_local_heads, self.qk_head_dim
-                )
+                # NSA requires unquantized q_lora for the indexer. When q_b_proj is FP8
+                # on gfx95, we can still use fused RMSNorm+FP8 quant, but MUST request
diff -- python/sglang/srt/layers/attention/nsa/index_buf_accessor.py
@@ -4,6 +4,8 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +171/-55; `python/sglang/srt/models/deepseek_v2.py` modified +58/-18; `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +10/-3; `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0; `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17054 - Update deepseekV32 Cp doc

- Link: https://github.com/sgl-project/sglang/pull/17054
- Status/date: merged / 2026-01-14
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `2122fea3c408`
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Update deepseekV32 Cp doc". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: fix doc typo
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -394,7 +394,7 @@ python -m sglang.launch_server \; -419,7 +419,7 @@ python -m sglang.launch_server \.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -394,7 +394,7 @@ python -m sglang.launch_server \; -419,7 +419,7 @@ python -m sglang.launch_server \
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -394,7 +394,7 @@ python -m sglang.launch_server \
-  --nsa-prefill-cp-mode continuous-split \
+  --nsa-prefill-cp-mode round-robin-split  \
@@ -419,7 +419,7 @@ python -m sglang.launch_server \
-  --nsa-prefill-cp-mode continuous-split \
+  --nsa-prefill-cp-mode round-robin-split  \
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +2/-2
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17185 - [DeepSeek V3.2] [Feat] add tensor parallel o_proj linear in context parallel nsa

- Link: https://github.com/sgl-project/sglang/pull/17185
- Status/date: open / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +61/-5, 154 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek V3.2] [Feat] add tensor parallel o_proj linear in context parallel nsa". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation The current NSA context parallel partitioning causes the o_proj to be copied across all cards, resulting in each card using 10GB more VRAM compared to DeepSeek v3....
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +49/-5 (54 lines); hunks: -24,6 +24,7; -47,6 +48,7; symbols: __init__, forward_absorb_core, forward_absorb_fused_mla_rope_core, forward_absorb_fused_mla_rope_cpu_core, touching `__init__, forward_absorb_core, forward_absorb_fused_mla_rope_core`; `python/sglang/srt/layers/attention/nsa/utils.py` modified +4/-0 (4 lines); hunks: -31,6 +31,10 @@ def is_nsa_enable_prefill_cp():; symbols: is_nsa_enable_prefill_cp, is_enable_o_proj_tp, is_nsa_prefill_cp_in_seq_split, touching `is_nsa_enable_prefill_cp, is_enable_o_proj_tp, is_nsa_prefill_cp_in_seq_split`; `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -628,6 +628,7 @@ class ServerArgs:; -4647,6 +4648,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +49/-5 (54 lines); hunks: -24,6 +24,7; -47,6 +48,7; symbols: __init__, forward_absorb_core, forward_absorb_fused_mla_rope_core, forward_absorb_fused_mla_rope_cpu_core
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +4/-0 (4 lines); hunks: -31,6 +31,10 @@ def is_nsa_enable_prefill_cp():; symbols: is_nsa_enable_prefill_cp, is_enable_o_proj_tp, is_nsa_prefill_cp_in_seq_split
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -628,6 +628,7 @@ class ServerArgs:; -4647,6 +4648,13 @@ def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs, add_cli_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -24,6 +24,7 @@
+from torch import distributed as dist
@@ -47,6 +48,7 @@
+from sglang.srt.distributed.parallel_state import get_tp_group
@@ -60,6 +62,7 @@
+    is_enable_o_proj_tp,
@@ -1093,10 +1096,16 @@ def __init__(
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -31,6 +31,10 @@ def is_nsa_enable_prefill_cp():
+def is_enable_o_proj_tp():
+    return get_global_server_args().enable_o_proj_tensor_parallel
diff -- python/sglang/srt/server_args.py
@@ -628,6 +628,7 @@ class ServerArgs:
+    enable_o_proj_tensor_parallel: bool = False
@@ -4647,6 +4648,13 @@ def add_cli_args(parser: argparse.ArgumentParser):
+        parser.add_argument(
+            "--enable-o-proj-tensor-parallel",
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +49/-5; `python/sglang/srt/layers/attention/nsa/utils.py` modified +4/-0; `python/sglang/srt/server_args.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16934 - [AMD] Enable DeepseekV3.2 test for AMD CI

- Link: https://github.com/sgl-project/sglang/pull/16934
- Status/date: merged / 2026-01-16
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; associated commits `968c4f55b194`
- Diff scope read: GitHub Pull Request files API returned 4 files, +379/-1, 403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable DeepseekV3.2 test for AMD CI". The diff centers on `test/registered/amd/test_deepseek_v32_mtp.py`, `test/registered/amd/test_deepseek_v32_basic.py`. PR body context: ## Motivation To enable DeepseekV3.2 test for AMD CI ## Modifications ### test/registered/amd/test_deepseek_v32_basic.py from test/srt/test_deepseek_v32_basic.py ### test/regist...
- Key implementation: `test/registered/amd/test_deepseek_v32_mtp.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32DPMTP, setUpClass, tearDownClass`; `test/registered/amd/test_deepseek_v32_basic.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV32DP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32DP, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/amd/test_deepseek_v32_mtp.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: TestDeepseekV32DPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/test_deepseek_v32_basic.py` added +163/-0 (163 lines); hunks: -0,0 +1,163; symbols: TestDeepseekV32DP, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/amd/test_deepseek_v32_mtp.py
@@ -0,0 +1,213 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
diff -- test/registered/amd/test_deepseek_v32_basic.py
@@ -0,0 +1,163 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_amd_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.send_one import BenchArgs, send_one_prompt
```

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v32_mtp.py` added +213/-0; `test/registered/amd/test_deepseek_v32_basic.py` added +163/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- Link: https://github.com/sgl-project/sglang/pull/17133
- Status/date: merged / 2026-01-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +959/-217, 1311 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab". The diff centers on `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`. PR body context: ## Motivation 1. Performance tuning based on the code after fused moe swapab #16723. The optimal configuration of fused MoE changes when swapab is taken into consideration. 2. O...
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

### PR #16961 - [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation

- Link: https://github.com/sgl-project/sglang/pull/16961
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `d2105d4abda6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-12, 105 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation 1、`draft_extend` and `target_verify` use `nsa_decode_backend` as backend implementation instead of `nsa_prefill_backend` In the MTP scenario, the NSA attention imp...
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
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py`, `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; associated commits `a3addd6203ca`
- Diff scope read: GitHub Pull Request files API returned 18 files, +1140/-99, 1539 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add DeepSeek-V3.2 and VLMs model in nightly tests". The diff centers on `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py`. PR body context: ## Motivation Add 5 unique test in AMD coverage: https://github.com/sgl-project/sglang/actions/runs/21153414554/attempts/1#summary-60833755716 - Add accuracy and perf tests for...
- Key implementation: `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: ModelConfig, __post_init__, get_display_name, get_one_example, touching `ModelConfig, __post_init__, get_display_name`; `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch, touching `generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass`; `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32BasicPerformance, setUpClass, test_bench_one_batch, touching `generate_simple_markdown_report, TestNightlyDeepseekV32BasicPerformance, setUpClass`; `test/registered/amd/test_deepseek_v32_basic.py` modified +1/-1 (2 lines); hunks: -16,7 +16,7; symbols: TestDeepseekV32DP, touching `TestDeepseekV32DP`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` added +251/-0 (251 lines); hunks: -0,0 +1,251; symbols: ModelConfig, __post_init__, get_display_name, get_one_example
  - `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` added +146/-0 (146 lines); hunks: -0,0 +1,146; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32BasicPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/test_deepseek_v32_basic.py` modified +1/-1 (2 lines); hunks: -16,7 +16,7; symbols: TestDeepseekV32DP
  - `test/registered/amd/test_deepseek_v32_mtp.py` modified +1/-1 (2 lines); hunks: -18,7 +18,7; symbols: TestDeepseekV32DPMTP
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py
@@ -0,0 +1,251 @@
+"""MI35x DeepSeek-V3.2 GSM8K Completion Evaluation Test (8-GPU)
+Tests DeepSeek-V3.2 with basic configuration using few-shot completion
+benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32 suite
+"""
+import ast
diff -- test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py
@@ -0,0 +1,146 @@
+"""MI35x Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).
+This test benchmarks the DeepSeek-V3.2 model with MTP (EAGLE speculative decoding)
+configuration on 8 GPUs.
+The model path can be configured via DEEPSEEK_V32_MODEL_PATH environment variable.
+Registry: nightly-perf-8-gpu-mi35x-deepseek-v32-mtp suite
+Example usage:
diff -- test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py
@@ -0,0 +1,134 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` added +251/-0; `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` added +146/-0; `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` added +134/-0; `test/registered/amd/test_deepseek_v32_basic.py` modified +1/-1; `test/registered/amd/test_deepseek_v32_mtp.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/accuracy/test_gsm8k_eval_amd.py`, `test/registered/amd/accuracy/test_vlms_mmmu_eval_amd.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17409 - [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton

- Link: https://github.com/sgl-project/sglang/pull/17409
- Status/date: merged / 2026-01-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix]: correctly fetch ds32 config in tuning_fused_moe_triton". The diff centers on `benchmark/kernels/fused_moe_triton/common_utils.py`. PR body context: ## Motivation fix "deepseek_v32" KeyError ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with...
- Key implementation: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +2/-2 (4 lines); hunks: -2,13 +2,13; -36,7 +36,7 @@ def get_model_config(; symbols: BenchmarkConfig, get_model_config, touching `BenchmarkConfig, get_model_config`.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +2/-2 (4 lines); hunks: -2,13 +2,13; -36,7 +36,7 @@ def get_model_config(; symbols: BenchmarkConfig, get_model_config
- Key code excerpts:

```diff
diff -- benchmark/kernels/fused_moe_triton/common_utils.py
@@ -2,13 +2,13 @@
-from transformers import AutoConfig
+from sglang.srt.utils.hf_transformers_utils import get_config
@@ -36,7 +36,7 @@ def get_model_config(
-    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
+    config = get_config(model_name, trust_remote_code=True)
```

- Reviewed files:
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +2/-2
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #17205 - [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance

- Link: https://github.com/sgl-project/sglang/pull/17205
- Status/date: merged / 2026-01-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `612026ad2c82`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation From these pr[https://github.com/sgl-project/sglang/pull/16637 and https://github.com/sgl-project/sglang/issues/16861 and https://github.com/sgl-project/sglang/pul...
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

### PR #17452 - Fix NSA indexer in the nightly test

- Link: https://github.com/sgl-project/sglang/pull/17452
- Status/date: merged / 2026-01-21
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/kernels/test_nsa_indexer.py`; associated commits `be5121b45219`
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-0, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix NSA indexer in the nightly test". The diff centers on `test/registered/kernels/test_nsa_indexer.py`. PR body context: https://github.com/sgl-project/sglang/actions/runs/21194510000
- Key implementation: `test/registered/kernels/test_nsa_indexer.py` modified +16/-0 (16 lines); hunks: -79,6 +79,22 @@ def get_page_table_64(self) -> torch.Tensor:; symbols: get_page_table_64, get_page_table_1, get_seqlens_expanded, touching `get_page_table_64, get_page_table_1, get_seqlens_expanded`.
- Code diff details:
  - `test/registered/kernels/test_nsa_indexer.py` modified +16/-0 (16 lines); hunks: -79,6 +79,22 @@ def get_page_table_64(self) -> torch.Tensor:; symbols: get_page_table_64, get_page_table_1, get_seqlens_expanded
- Key code excerpts:

```diff
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -79,6 +79,22 @@ def get_page_table_64(self) -> torch.Tensor:
+    def get_page_table_1(self) -> torch.Tensor:
+        """Return: (batch_size, num_blocks) int32, page table with page size 1."""
+        # Create a simple page table for testing with page size 1
+        max_seq_len = max(self.seq_lens)
+        num_blocks = max_seq_len  # Page size 1 means num_blocks == max_seq_len
+        page_table = torch.zeros(
```

- Reviewed files:
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +16/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17518 - [HotFix]Fix dtype mismatch in nsa indexer on AMD device

- Link: https://github.com/sgl-project/sglang/pull/17518
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `3373545b9fba`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HotFix]Fix dtype mismatch in nsa indexer on AMD device". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Fix this CI failure https://github.com/sgl-project/sglang/actions/runs/21192748451/job/60966055534 Introduced in https://github.com/sgl-project/sglang/pull/17205/f...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunks: -186,7 +186,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -186,7 +186,7 @@ def __init__(
-            params_dtype=torch.bfloat16,
+            params_dtype=torch.bfloat16 if _is_cuda else torch.float32,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17432 - [AMD] fix amd ci dpskv32

- Link: https://github.com/sgl-project/sglang/pull/17432
- Status/date: merged / 2026-01-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; associated commits `17807caf82c8`
- Diff scope read: GitHub Pull Request files API returned 5 files, +16/-7, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[AMD] fix amd ci dpskv32". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`. PR body context: ## Motivation Fix the runtime error from PR-17205 : https://github.com/sgl-project/sglang/actions/runs/21157007917/job/60858903195?pr=17205#step:6:17583 ## Modifications ## Accu...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-0 (4 lines); hunks: -220,13 +220,17 @@ def _with_real_sm_count(self):; symbols: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, touching `_with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate`; `test/registered/amd/test_deepseek_v32_basic.py` modified +7/-2 (9 lines); hunks: -15,10 +15,16; -90,7 +96,6 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, setUpClass, test_bs_1_speed, TestDeepseekV32TP, touching `TestDeepseekV32DP, setUpClass, test_bs_1_speed`; `test/registered/amd/test_deepseek_v32_mtp.py` modified +2/-2 (4 lines); hunks: -21,6 +21,7; -116,7 +117,6 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DPMTP, setUpClass, test_bs_1_speed, TestDeepseekV32TPMTP, touching `TestDeepseekV32DPMTP, setUpClass, test_bs_1_speed`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-0 (4 lines); hunks: -220,13 +220,17 @@ def _with_real_sm_count(self):; symbols: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate
  - `test/registered/amd/test_deepseek_v32_basic.py` modified +7/-2 (9 lines); hunks: -15,10 +15,16; -90,7 +96,6 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DP, setUpClass, test_bs_1_speed, TestDeepseekV32TP
  - `test/registered/amd/test_deepseek_v32_mtp.py` modified +2/-2 (4 lines); hunks: -21,6 +21,7; -116,7 +117,6 @@ def test_bs_1_speed(self):; symbols: TestDeepseekV32DPMTP, setUpClass, test_bs_1_speed, TestDeepseekV32TPMTP
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -220,13 +220,17 @@ def _with_real_sm_count(self):
+        if _is_hip:
+            x = x.to(self.weights_proj.weight.dtype)
+        if _is_hip:
+            x = x.to(self.weights_proj.weight.dtype)
diff -- test/registered/amd/test_deepseek_v32_basic.py
@@ -15,10 +15,16 @@
-register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")
+register_amd_ci(
+    est_time=3600,
+    suite="stage-c-test-large-8-gpu-amd-mi35x",
+    disabled="move to nightly for saving time",
+)
diff -- test/registered/amd/test_deepseek_v32_mtp.py
@@ -21,6 +21,7 @@
+@unittest.skipIf(is_in_amd_ci(), "Skip DP test for AMD CI, run TP only.")
@@ -116,7 +117,6 @@ def test_bs_1_speed(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-0
  - tests: `test/registered/amd/test_deepseek_v32_basic.py` modified +7/-2; `test/registered/amd/test_deepseek_v32_mtp.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17007 - [NPU]bugfix: fix for dsv3.2 and dsvl2

- Link: https://github.com/sgl-project/sglang/pull/17007
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `c0b5a180febe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +129/-46, 224 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[NPU]bugfix: fix for dsv3.2 and dsvl2". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: ## Motivation 1、There are some bugs for DS-Vl2 on rotary_embedding 2、DSV32 is not compatible with the scenario where m.alt_stream is not None. ## Modifications 1、Fix the ds-v12...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Enables NSA backend with trtllm kernels for sparse attention. This can be more efficient than FlashMLA when the head size isn't a multiple of 64 and hence requires...
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

### PR #17682 - Fix NSA indexer test and move it to pre commit test

- Link: https://github.com/sgl-project/sglang/pull/17682
- Status/date: merged / 2026-01-24
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/kernels/test_nsa_indexer.py`; associated commits `137eb5b95cb1`
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix NSA indexer test and move it to pre commit test". The diff centers on `test/registered/kernels/test_nsa_indexer.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/registered/kernels/test_nsa_indexer.py` modified +3/-1 (4 lines); hunks: -24,7 +24,7; -41,6 +41,7; symbols: __init__, touching `__init__`.
- Code diff details:
  - `test/registered/kernels/test_nsa_indexer.py` modified +3/-1 (4 lines); hunks: -24,7 +24,7; -41,6 +41,7; symbols: __init__
- Key code excerpts:

```diff
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -24,7 +24,7 @@
-register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)
+register_cuda_ci(est_time=2, suite="stage-b-test-small-1-gpu")
@@ -41,6 +41,7 @@
+    "qk_nope_head_dim": 128,
@@ -190,6 +191,7 @@ def __init__(self, config=None):
+                "qk_nope_head_dim": self.config["qk_nope_head_dim"],
```

- Reviewed files:
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17662 - [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend

- Link: https://github.com/sgl-project/sglang/pull/17662
- Status/date: merged / 2026-01-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `1674b9ef4494`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +18/-1, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation Quick follow‑up to #16758: speculative decoding still fails with `--nsa-decode-backend trtllm`. ## Modifications - Add TRT‑LLM handling in the extend path for spec...
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

### PR #17761 - fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates

- Link: https://github.com/sgl-project/sglang/pull/17761
- Status/date: open / 2026-01-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +79/-2, 102 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "fix: missing Assistant token after tool output in DeepSeek v3.1/v3.2 chat templates". The diff centers on `test/manual/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`, `examples/chat_template/tool_chat_template_deepseekv32.jinja`. PR body context: ## Motivation Fix missing ` ` token after tool output in DeepSeek v3.1 and v3.2 chat templates. When an assistant message follows a tool output, the ` ` token was not being adde...
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

### PR #15381 - [NPU]DeepSeek-V3.2 support npu mlaprolog

- Link: https://github.com/sgl-project/sglang/pull/15381
- Status/date: merged / 2026-01-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `b56366f8275a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +195/-61, 364 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[NPU]DeepSeek-V3.2 support npu mlaprolog". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: ## Motivation we enabled the DSV3.2 to support npu mla_prolog_v3 to improve the performance ## Modifications - Added support for the mla_prolog_v3 custom operator, which include...
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

### PR #17783 - [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build

- Link: https://github.com/sgl-project/sglang/pull/17783
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `df42f4d386d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +81/-88, 214 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Improve the AMD GPU user experience for DeepSeek V3.2 by documenting correct ROCm images and launch flags, while also consolidating the TileLang build to use the u...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +10/-1 (11 lines); hunks: -16,7 +16,13 @@ Note: This document is originally written for the usage of [D...; -45,6 +51,9 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +10/-1 (11 lines); hunks: -16,7 +16,13 @@ Note: This document is originally written for the usage of [D...; -45,6 +51,9 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -16,7 +16,13 @@ Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](h
-docker pull lmsysorg/sglang:dsv32-rocm
+docker pull lmsysorg/sglang:v0.5.8-rocm700-mi35x
+# MI300
+# v0.5.8-rocm700-mi30x does not include PR #17504. Prefer the newest MI30x ROCm
+# image tag from Docker Hub when available, or build from source (below).
+docker pull lmsysorg/sglang:v0.5.8-rocm700-mi30x
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +10/-1
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17657 - [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint

- Link: https://github.com/sgl-project/sglang/pull/17657
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `1d942e4eef5e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +88/-0, 103 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Accuracy Tests Some accuracy results: GSM8K 20shot GPQA-Diamond TODO：Update Cookbook ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +8/-0 (8 lines); hunks: -116,6 +116,14 @@ python3 -m sglang.launch_server \.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +8/-0 (8 lines); hunks: -116,6 +116,14 @@ python3 -m sglang.launch_server \
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -116,6 +116,14 @@ python3 -m sglang.launch_server \
+## NVFP4 Checkpoint
+To launch deepseek v3.2 [NVFP4 checkpoint](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4) on Blackwell devices, the user needs to specify the quantization method as `modelopt_
+An example launching command can be:
+'''bash
+python -m sglang.launch_server --model nvidia/DeepSeek-V3.2-NVFP4 --tp 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --tool-call-parser deepseekv32  --reaso
+'''
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +8/-0
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/test_deepseek_v32_fp4_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17633 - [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites

- Link: https://github.com/sgl-project/sglang/pull/17633
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`; associated commits `52bca4287033`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +88/-230, 490 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites". The diff centers on `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`. PR body context: ## Motivation Enable/move 2 CI cases to mi325-8gpu runner - deepseek v3.2 - Kimi-K2-Instruct-0905 Update AMD CI config to align with https://github.com/sgl-project/sglang/pull/1...
- Key implementation: `test/registered/amd/test_deepseek_v32_basic.py` modified +2/-6 (8 lines); hunks: -15,11 +15,7; -159,7 +155,7 @@ def test_bs_1_speed(self):; symbols: test_bs_1_speed, touching `test_bs_1_speed`; `test/registered/amd/test_deepseek_v32_mtp.py` modified +6/-1 (7 lines); hunks: -17,7 +17,12.
- Code diff details:
  - `test/registered/amd/test_deepseek_v32_basic.py` modified +2/-6 (8 lines); hunks: -15,11 +15,7; -159,7 +155,7 @@ def test_bs_1_speed(self):; symbols: test_bs_1_speed
  - `test/registered/amd/test_deepseek_v32_mtp.py` modified +6/-1 (7 lines); hunks: -17,7 +17,12
- Key code excerpts:

```diff
diff -- test/registered/amd/test_deepseek_v32_basic.py
@@ -15,11 +15,7 @@
-register_amd_ci(
-    est_time=3600,
-    suite="stage-c-test-large-8-gpu-amd-mi35x",
-    disabled="move to nightly for saving time",
-)
+register_amd_ci(est_time=1800, suite="stage-c-test-large-8-gpu-amd")
diff -- test/registered/amd/test_deepseek_v32_mtp.py
@@ -17,7 +17,12 @@
-register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")
+register_amd_ci(
+    est_time=3600,
+    suite="stage-c-test-large-8-gpu-amd-mi35x",
+    disabled="move to nightly for saving time",
+)
```

- Reviewed files:
  - tests: `test/registered/amd/test_deepseek_v32_basic.py` modified +2/-6; `test/registered/amd/test_deepseek_v32_mtp.py` modified +6/-1
- Risk and verification: The diff ships test coverage in `test/registered/amd/test_deepseek_v32_basic.py`, `test/registered/amd/test_deepseek_v32_mtp.py`, `test/registered/amd/test_kimi_k2_instruct.py`, `test/registered/attention/test_triton_sliding_window.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17688 - [DSv32] Overlap indexer qk projection and activation quant

- Link: https://github.com/sgl-project/sglang/pull/17688
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `a8dda2aa5727`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[DSv32] Overlap indexer qk projection and activation quant". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation @humansand After https://github.com/sgl-project/sglang/pull/17205, the indexer weight projection is fully hidden and no longer exposes latency. ## Modifications ##...
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
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py`; associated commits `f8636fbb253a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +1540/-43, 1823 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI". The diff centers on `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py`. PR body context: ## Motivation 1. Add **Kimi-K2**, **DeepSeek-V3.2** accuracy and performance tests for MI325 (MI30x) platform, update Mi35x tests, consolidate test jobs, and fix various CI fail...
- Key implementation: `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32TPMTP, setUpClass, tearDownClass`; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: TestDeepseekV32DP, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32DP, setUpClass, tearDownClass`; `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` modified +59/-3 (62 lines); hunks: -13,12 +13,17; -57,11 +62,58 @@ def generate_simple_markdown_report(results: List[BenchmarkR...; symbols: generate_simple_markdown_report, _run_benchmark_with_timeout, TestNightlyDeepseekV32MTPPerformance, setUpClass, touching `generate_simple_markdown_report, _run_benchmark_with_timeout, TestNightlyDeepseekV32MTPPerformance`; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -215,6 +215,9 @@ def test_deepseek_v32_accuracy(self):; symbols: test_deepseek_v32_accuracy, touching `test_deepseek_v32_accuracy`.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: -0,0 +1,142; symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py` added +119/-0 (119 lines); hunks: -0,0 +1,119; symbols: TestDeepseekV32DP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` modified +59/-3 (62 lines); hunks: -13,12 +13,17; -57,11 +62,58 @@ def generate_simple_markdown_report(results: List[BenchmarkR...; symbols: generate_simple_markdown_report, _run_benchmark_with_timeout, TestNightlyDeepseekV32MTPPerformance, setUpClass
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` modified +3/-0 (3 lines); hunks: -215,6 +215,9 @@ def test_deepseek_v32_accuracy(self):; symbols: test_deepseek_v32_accuracy
  - `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` modified +2/-0 (2 lines); hunks: -93,6 +93,8 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py
@@ -0,0 +1,142 @@
+"""MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU)
+Tests DeepSeek-V3.2 with TP=8 + MTP (EAGLE speculative decoding) using few-shot
+completion benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-mtp suite
+"""
+import os
diff -- test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py
@@ -0,0 +1,119 @@
+"""MI35x DeepSeek-V3.2 DP GSM8K Accuracy Evaluation Test (8-GPU)
+Tests DeepSeek-V3.2 with DP=8 + TP=8 + dp-attention using few-shot
+completion benchmark on MI35x.
+Registry: nightly-amd-accuracy-8-gpu-mi35x-deepseek-v32-dp suite
+"""
+import os
diff -- test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py
@@ -13,12 +13,17 @@
```

- Reviewed files:
  - tests: `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py` added +119/-0; `test/registered/amd/perf/mi35x/test_deepseek_v32_mtp_perf_mi35x.py` modified +59/-3; `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py` modified +3/-0; `test/registered/amd/perf/mi35x/test_deepseek_v32_basic_perf_mi35x.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/registered/amd/accuracy/mi35x/test_deepseek_r1_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_dp_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17951 - Add tool call tests for DeepSeek V3.2 in nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17951
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/8-gpu-models/test_deepseek_v32.py`; associated commits `d417c6809e21`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +363/-5, 472 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add tool call tests for DeepSeek V3.2 in nightly CI". The diff centers on `test/registered/8-gpu-models/test_deepseek_v32.py`. PR body context: ## Summary Add tool call tests for DeepSeek V3.2 to nightly 8-GPU CI. - New `tool_call_test_runner.py`: reusable runner with 9 test cases (basic format, streaming, tool_choice r...
- Key implementation: `test/registered/8-gpu-models/test_deepseek_v32.py` modified +11/-4 (15 lines); hunks: -5,6 +5,7; -16,6 +17,11; symbols: test_deepseek_v32_all_variants, touching `test_deepseek_v32_all_variants`.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v32.py` modified +11/-4 (15 lines); hunks: -5,6 +5,7; -16,6 +17,11; symbols: test_deepseek_v32_all_variants
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v32.py
@@ -5,6 +5,7 @@
+from sglang.test.tool_call_test_runner import ToolCallTestParams
@@ -16,6 +17,11 @@
+TOOL_CALL_ARGS = [
+    "--tool-call-parser=deepseekv32",
+    "--reasoning-parser=deepseek-v3",
+]
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32.py` modified +11/-4
- Risk and verification: The diff ships test coverage in `python/sglang/test/run_combined_tests.py`, `python/sglang/test/tool_call_test_runner.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18094 - support deepseekv3.2-piecewise-cuda-graph

- Link: https://github.com/sgl-project/sglang/pull/18094
- Status/date: open / 2026-02-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +243/-91, 656 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "support deepseekv3.2-piecewise-cuda-graph". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. PR body context: ## Motivation ## Modifications support deepseekv3.2-piecewise-cuda-graph ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the For...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +148/-48 (196 lines); hunks: -32,6 +32,10; -64,6 +68,7; symbols: forward, forward_deepep, _post_combine_hook, __init__, touching `forward, forward_deepep, _post_combine_hook`; `python/sglang/srt/layers/radix_attention.py` modified +19/-19 (38 lines); hunks: -114,25 +114,25 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-3 (15 lines); hunks: -1405,21 +1405,30 @@ def forward_impl(self, hidden_states: torch.Tensor, topk...; symbols: forward_impl, moe_forward_piecewise_cuda_graph_impl, touching `forward_impl, moe_forward_piecewise_cuda_graph_impl`; `python/sglang/srt/layers/moe/topk.py` modified +12/-3 (15 lines); hunks: -740,15 +740,20 @@ def is_power_of_two(n):; -775,6 +780,11 @@ def biased_grouped_topk_gpu(; symbols: is_power_of_two, _mask_topk_ids_padded_region, _biased_grouped_topk_postprocess, biased_grouped_topk_gpu, touching `is_power_of_two, _mask_topk_ids_padded_region, _biased_grouped_topk_postprocess`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +148/-48 (196 lines); hunks: -32,6 +32,10; -64,6 +68,7; symbols: forward, forward_deepep, _post_combine_hook, __init__
  - `python/sglang/srt/layers/radix_attention.py` modified +19/-19 (38 lines); hunks: -114,25 +114,25 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-3 (15 lines); hunks: -1405,21 +1405,30 @@ def forward_impl(self, hidden_states: torch.Tensor, topk...; symbols: forward_impl, moe_forward_piecewise_cuda_graph_impl
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-3 (15 lines); hunks: -740,15 +740,20 @@ def is_power_of_two(n):; -775,6 +780,11 @@ def biased_grouped_topk_gpu(; symbols: is_power_of_two, _mask_topk_ids_padded_region, _biased_grouped_topk_postprocess, biased_grouped_topk_gpu
  - `python/sglang/srt/layers/communicator.py` modified +7/-5 (12 lines); hunks: -534,11 +534,13 @@ def prepare_attn(; symbols: prepare_attn, _tp_reduce_scatter
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -32,6 +32,10 @@
+from sglang.srt.compilation.compilation_config import register_split_op
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+)
@@ -64,6 +68,7 @@
+    AttentionInputs,
diff -- python/sglang/srt/layers/radix_attention.py
@@ -114,25 +114,25 @@ def forward(
-        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
-            if self.qk_head_dim != self.v_head_dim:
-                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
-            else:
-                output = torch.empty_like(q)
-            unified_attention_with_output(
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -1405,21 +1405,30 @@ def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +148/-48; `python/sglang/srt/layers/radix_attention.py` modified +19/-19; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-3; `python/sglang/srt/layers/moe/topk.py` modified +12/-3; `python/sglang/srt/layers/communicator.py` modified +7/-5; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/eplb/expert_distribution.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17076 - [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph

- Link: https://github.com/sgl-project/sglang/pull/17076
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `677f3c49dabf`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +58/-7, 135 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation This PR #15227 only resolves the issue occurring during `forward_idle`. If running in Eagle mode (MTP Spec2, batch size exceeding the CUDA maximum batch size, or w...
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
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `b0a6d5244cef`
- Diff scope read: GitHub Pull Request files API returned 4 files, +16/-2, 60 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[NPU] support dsv32 radixcache on ascend". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Ascend currently does not support the radixcache feature on the DeepSeek-3.2 model ## Modifications 1. We have modified the inputs of `npu_sparse_flash_attention`...
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

### PR #18167 - [Feature] Add DCP support for DeepSeek v3.2

- Link: https://github.com/sgl-project/sglang/pull/18167
- Status/date: open / 2026-02-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +567/-62, 1284 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Feature] Add DCP support for DeepSeek v3.2". The diff centers on `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation Following PR https://github.com/sgl-project/sglang/pull/14982, add decode context parallel (DCP) support for DeepSeek v3.2. The KV cache redundancy issue in DeepSe...
- Key implementation: `python/sglang/srt/layers/attention/utils.py` modified +211/-0 (211 lines); hunks: -4,6 +4,8; -411,3 +413,212 @@ def concat_mla_absorb_q_general(q_nope, q_rope):; symbols: concat_mla_absorb_q_general, _correct_attn_cp_out_kernel, CPTritonContext, __init__, touching `concat_mla_absorb_q_general, _correct_attn_cp_out_kernel, CPTritonContext`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +91/-37 (128 lines); hunks: -7,6 +7,7; -28,6 +29,7; symbols: __init__, get_device_int32_arange, init_forward_metadata_replay_cuda_graph_from_precomputed, _save_kv_cache, touching `__init__, get_device_int32_arange, init_forward_metadata_replay_cuda_graph_from_precomputed`; `python/sglang/srt/models/deepseek_v2.py` modified +33/-0 (33 lines); hunks: -42,6 +42,7; -1126,6 +1127,9 @@ def __init__(; symbols: __init__, rebuild_cp_kv_cache, _filter_topk_indices_by_dcp, forward_absorb_prepare, touching `__init__, rebuild_cp_kv_cache, _filter_topk_indices_by_dcp`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +20/-0 (20 lines); hunks: -113,6 +113,7 @@ class DecodeInputBuffers(ForwardInputBuffers):; -126,6 +127,7 @@ def create(; symbols: DecodeInputBuffers, create, touching `DecodeInputBuffers, create`.
- Code diff details:
  - `python/sglang/srt/layers/attention/utils.py` modified +211/-0 (211 lines); hunks: -4,6 +4,8; -411,3 +413,212 @@ def concat_mla_absorb_q_general(q_nope, q_rope):; symbols: concat_mla_absorb_q_general, _correct_attn_cp_out_kernel, CPTritonContext, __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +91/-37 (128 lines); hunks: -7,6 +7,7; -28,6 +29,7; symbols: __init__, get_device_int32_arange, init_forward_metadata_replay_cuda_graph_from_precomputed, _save_kv_cache
  - `python/sglang/srt/models/deepseek_v2.py` modified +33/-0 (33 lines); hunks: -42,6 +42,7; -1126,6 +1127,9 @@ def __init__(; symbols: __init__, rebuild_cp_kv_cache, _filter_topk_indices_by_dcp, forward_absorb_prepare
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +20/-0 (20 lines); hunks: -113,6 +113,7 @@ class DecodeInputBuffers(ForwardInputBuffers):; -126,6 +127,7 @@ def create(; symbols: DecodeInputBuffers, create
  - `python/sglang/srt/layers/attention/nsa/transform_index.py` modified +13/-1 (14 lines); hunks: -20,6 +20,7 @@ def transform_index_page_table_decode_kernel(; -30,7 +31,9 @@ def transform_index_page_table_decode_kernel(; symbols: transform_index_page_table_decode_kernel, transform_index_page_table_decode_fast, transform_index_page_table_prefill_fast
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/utils.py
@@ -4,6 +4,8 @@
+from sglang.srt.distributed.parallel_state import GroupCoordinator
@@ -411,3 +413,212 @@ def concat_mla_absorb_q_general(q_nope, q_rope):
+# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/attention/ops/common.py
+@triton.jit
+def _correct_attn_cp_out_kernel(
+    outputs_ptr,
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -7,6 +7,7 @@
+from sglang.srt.distributed.parallel_state import get_dcp_group
@@ -28,6 +29,7 @@
+    is_nsa_enable_decode_cp,
@@ -37,6 +39,7 @@
+from sglang.srt.layers.attention.utils import cp_lse_ag_out_rs
@@ -358,6 +361,9 @@ def __init__(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -42,6 +42,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/utils.py` modified +211/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +91/-37; `python/sglang/srt/models/deepseek_v2.py` modified +33/-0; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +20/-0; `python/sglang/srt/layers/attention/nsa/transform_index.py` modified +13/-1; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +8/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18275 - [NPU] allgather after qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/18275
- Status/date: open / 2026-02-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +98/-10, 267 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[NPU] allgather after qlora for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation In DeepSeek-V3.2, due to the large hidden dimension of the model, performing all-gather on hidden_states before attention introduces a high-dimensional communicati...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +28/-3 (31 lines); hunks: -28,12 +28,21; -1137,6 +1146,7 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`; `python/sglang/srt/layers/communicator.py` modified +28/-2 (30 lines); hunks: -168,14 +168,15 @@ class AttnTpContext:; -233,6 +234,29 @@ def get_attn_tp_context():; symbols: AttnTpContext, __init__, init_context, get_attn_tp_context, touching `AttnTpContext, __init__, init_context`; `python/sglang/srt/models/deepseek_v2.py` modified +10/-1 (11 lines); hunks: -1353,13 +1353,15 @@ def forward(; -1370,6 +1372,7 @@ def forward_prepare(; symbols: forward, forward_prepare, touching `forward, forward_prepare`; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +27/-3 (30 lines); hunks: -13,7 +13,12; -285,6 +290,7 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu, touching `forward_dsa_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +28/-3 (31 lines); hunks: -28,12 +28,21; -1137,6 +1146,7 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/layers/communicator.py` modified +28/-2 (30 lines); hunks: -168,14 +168,15 @@ class AttnTpContext:; -233,6 +234,29 @@ def get_attn_tp_context():; symbols: AttnTpContext, __init__, init_context, get_attn_tp_context
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-1 (11 lines); hunks: -1353,13 +1353,15 @@ def forward(; -1370,6 +1372,7 @@ def forward_prepare(; symbols: forward, forward_prepare
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +27/-3 (30 lines); hunks: -13,7 +13,12; -285,6 +290,7 @@ def forward_dsa_prepare_npu(; symbols: forward_dsa_prepare_npu
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +5/-1 (6 lines); hunks: -292,6 +292,10 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -628,7 +632,7 @@ def forward_sparse(; symbols: init_forward_metadata, forward_sparse
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -28,12 +28,21 @@
+from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
+    scattered_to_tp_attn_full,
+)
+from sglang.srt.layers.communicator import (
+    ScatterMode,
+    delay_gather_for_dsa,
diff -- python/sglang/srt/layers/communicator.py
@@ -168,14 +168,15 @@ class AttnTpContext:
+        self.is_nsa = False
+        self.is_nsa = is_nsa
-            and _is_cuda
+            and (_is_cuda or _is_npu)
-            and not is_nsa
@@ -233,6 +234,29 @@ def get_attn_tp_context():
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1353,13 +1353,15 @@ def forward(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +28/-3; `python/sglang/srt/layers/communicator.py` modified +28/-2; `python/sglang/srt/models/deepseek_v2.py` modified +10/-1; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +27/-3; `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18297 - Deepseekv32 compatibility with transformers v5

- Link: https://github.com/sgl-project/sglang/pull/18297
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `e8a2c133807c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +33/-19, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Deepseekv32 compatibility with transformers v5". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation ## Modifications 1. Enhance Indexer and DeepseekV2AttentionMLA to support dynamic is_neox_style parameter. 2. Refactor rope parameter handling in DeepseekV2Decoder...
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

### PR #18542 - fix: fixed aux hidden state index out of range when using eagle3 with nsa cp

- Link: https://github.com/sgl-project/sglang/pull/18542
- Status/date: open / 2026-02-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +9/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "fix: fixed aux hidden state index out of range when using eagle3 with nsa cp". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation `python3 -m sglang.launch_server --model-path $MODEL_PATH --trust-remote-code \ --port 8000 --host 0.0.0.0 --attention-backend nsa \ --enable-metrics --mem-fractio...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -2708,7 +2708,15 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: -2708,7 +2708,15 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2708,7 +2708,15 @@ def forward(
-                    if self.enable_a2a_moe and i > self.first_k_dense_replace:
+                    if nsa_use_prefill_cp(forward_batch):
+                        aux_hidden_state = cp_all_gather_rerange_output(
+                            hidden_states + residual,
+                            self.cp_size,
+                            forward_batch,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +9/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18488 - Tilelang sparse decode fwd for dsv32 mi355

- Link: https://github.com/sgl-project/sglang/pull/18488
- Status/date: merged / 2026-02-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`; associated commits `4262f5259b94`
- Diff scope read: GitHub Pull Request files API returned 1 files, +257/-0, 271 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "Tilelang sparse decode fwd for dsv32 mi355". The diff centers on `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`. PR body context: cc @HaiShaw, @kkHuang-amd, @hubertlu-tw ## Motivation The current dsv32 attention implementation uses a single TileLang kernel for both prefill and decode. This kernel is optimi...
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
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix Bug on dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Fix two bugs on dsv32 when infering on npu ## Modifications 1. fix a bug: forward_npu method of nas_Indexer has a dual stream feature that can not be closed. May c...
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

### PR #18613 - [V3.2] Change default CP token split method to `--round-robin-split`

- Link: https://github.com/sgl-project/sglang/pull/18613
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `947927bdb55a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +5/-5, 45 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[V3.2] Change default CP token split method to `--round-robin-split`". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation Since the `--in-seq-split` one is out of maintenance ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -306,7 +306,7 @@ DeepSeek-V3.2-Speciale:; -326,7 +326,7 @@ Example:.
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -306,7 +306,7 @@ DeepSeek-V3.2-Speciale:; -326,7 +326,7 @@ Example:
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -306,7 +306,7 @@ DeepSeek-V3.2-Speciale:
-### In sequence splitting (default setting)
+### In sequence splitting
@@ -326,7 +326,7 @@ Example:
-### Round robin splitting
+### Round robin splitting (default setting)
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18733 - Add DeepSeek V32 PD disaggregation test

- Link: https://github.com/sgl-project/sglang/pull/18733
- Status/date: open / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +100/-0, 101 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add DeepSeek V32 PD disaggregation test". The diff centers on `test/registered/distributed/test_disaggregation_deepseek_v32.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/registered/distributed/test_disaggregation_deepseek_v32.py` added +100/-0 (100 lines); hunks: -0,0 +1,100; symbols: TestDisaggregationDeepseekV32, setUpClass, start_prefill, start_decode, touching `TestDisaggregationDeepseekV32, setUpClass, start_prefill`.
- Code diff details:
  - `test/registered/distributed/test_disaggregation_deepseek_v32.py` added +100/-0 (100 lines); hunks: -0,0 +1,100; symbols: TestDisaggregationDeepseekV32, setUpClass, start_prefill, start_decode
- Key code excerpts:

```diff
diff -- test/registered/distributed/test_disaggregation_deepseek_v32.py
@@ -0,0 +1,100 @@
+import unittest
+from types import SimpleNamespace
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
+from sglang.test.server_fixtures.disaggregation_fixture import (
+    PDDisaggregationServerBase,
```

- Reviewed files:
  - tests: `test/registered/distributed/test_disaggregation_deepseek_v32.py` added +100/-0
- Risk and verification: The diff ships test coverage in `test/registered/distributed/test_disaggregation_deepseek_v32.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17554 - Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels

- Link: https://github.com/sgl-project/sglang/pull/17554
- Status/date: merged / 2026-02-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `34132d6da50e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +2824/-54, 2945 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`. PR body context: ## Motivation Implement fused CUDA kernels to eliminate redundant metadata copies in Native Sparse Attention (NSA) backend during CUDA graph replay for speculative decoding. Thi...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: verify_single_backend_fused_metadata_copy, check_tensor_equal, verify_multi_backend_fused_metadata_copy, touching `verify_single_backend_fused_metadata_copy, check_tensor_equal, verify_multi_backend_fused_metadata_copy`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +307/-51 (358 lines); hunks: -16,6 +16,10; -63,6 +67,15; symbols: NSAFlashMLAMetadata, init_forward_metadata_replay_cuda_graph_from_precomputed, init_forward_metadata_replay_cuda_graph, touching `NSAFlashMLAMetadata, init_forward_metadata_replay_cuda_graph_from_precomputed, init_forward_metadata_replay_cuda_graph`; `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` modified +3/-3 (6 lines); hunks: -127,7 +127,7 @@ def _precompute_decode_mode(; -187,7 +187,7 @@ def _precompute_target_verify_mode(; symbols: _precompute_decode_mode, _precompute_target_verify_mode, _precompute_draft_extend_mode, touching `_precompute_decode_mode, _precompute_target_verify_mode, _precompute_draft_extend_mode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` added +407/-0 (407 lines); hunks: -0,0 +1,407; symbols: verify_single_backend_fused_metadata_copy, check_tensor_equal, verify_multi_backend_fused_metadata_copy
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +307/-51 (358 lines); hunks: -16,6 +16,10; -63,6 +67,15; symbols: NSAFlashMLAMetadata, init_forward_metadata_replay_cuda_graph_from_precomputed, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` modified +3/-3 (6 lines); hunks: -127,7 +127,7 @@ def _precompute_decode_mode(; -187,7 +187,7 @@ def _precompute_target_verify_mode(; symbols: _precompute_decode_mode, _precompute_target_verify_mode, _precompute_draft_extend_mode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py
@@ -0,0 +1,407 @@
+"""
+Verification utilities for NSA backend fused metadata copy operations.
+This module contains verification code to ensure that fused metadata copy kernels
+produce the same results as individual copy operations.
+"""
+import torch
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -16,6 +16,10 @@
+from sglang.srt.layers.attention.nsa.nsa_mtp_verification import (
+    verify_multi_backend_fused_metadata_copy,
+    verify_single_backend_fused_metadata_copy,
+)
@@ -63,6 +67,15 @@
+# Control whether to use fused metadata copy kernel (default: enabled)
diff -- python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py
@@ -127,7 +127,7 @@ def _precompute_decode_mode(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` added +407/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +307/-51; `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18126 - Fix dsv32 encode_messages

- Link: https://github.com/sgl-project/sglang/pull/18126
- Status/date: merged / 2026-02-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +30/-5, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix dsv32 encode_messages". The diff centers on `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/jinja_template_utils.py`. PR body context: Fix #18125
- Key implementation: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +14/-0 (14 lines); hunks: -393,6 +393,20 @@ def _apply_jinja_template(; symbols: _apply_jinja_template, touching `_apply_jinja_template`; `python/sglang/srt/parser/jinja_template_utils.py` modified +16/-5 (21 lines); hunks: -127,6 +127,7 @@ def process_content_for_template_format(; -138,6 +139,7 @@ def process_content_for_template_format(; symbols: process_content_for_template_format, touching `process_content_for_template_format`.
- Code diff details:
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +14/-0 (14 lines); hunks: -393,6 +393,20 @@ def _apply_jinja_template(; symbols: _apply_jinja_template
  - `python/sglang/srt/parser/jinja_template_utils.py` modified +16/-5 (21 lines); hunks: -127,6 +127,7 @@ def process_content_for_template_format(; -138,6 +139,7 @@ def process_content_for_template_format(; symbols: process_content_for_template_format
- Key code excerpts:

```diff
diff -- python/sglang/srt/entrypoints/openai/serving_chat.py
@@ -393,6 +393,20 @@ def _apply_jinja_template(
+            for msg in messages:
+                if msg.get("content") is None:
+                    msg["content"] = ""
+                processed_msg = process_content_for_template_format(
+                    msg,
+                    template_content_format,
diff -- python/sglang/srt/parser/jinja_template_utils.py
@@ -127,6 +127,7 @@ def process_content_for_template_format(
+    use_dpsk_v32_encoding: bool = False,
@@ -138,6 +139,7 @@ def process_content_for_template_format(
+        use_dpsk_v32_encoding: If True, extract multimodal data and convert content to string (for DeepSeek-V3.2 encoding)
@@ -146,9 +148,11 @@ def process_content_for_template_format(
-    if content_format == "openai":
+    if content_format == "openai" or use_dpsk_v32_encoding:
```

- Reviewed files:
  - runtime: `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +14/-0; `python/sglang/srt/parser/jinja_template_utils.py` modified +16/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/entrypoints/openai/serving_chat.py`, `python/sglang/srt/parser/jinja_template_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16907 - Fix model loading for DeepSeek-V3.2-AWQ

- Link: https://github.com/sgl-project/sglang/pull/16907
- Status/date: merged / 2026-02-15
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_v2.py`; associated commits `190fa8246fbc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-4, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix model loading for DeepSeek-V3.2-AWQ". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation Got the following error when using DeepSeek-V3.2-AWQ (https://huggingface.co/QuantTrio/DeepSeek-V3.2-AWQ/) ## Modifications Add check of whether `weight_packed` ex...
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
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `0ffd0a3995e5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +352/-183, 914 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation #17655 - support Deepseek v3.2 NVFP4 with trtllm mla sparse fp8 attention backend ## Modifications - update the nsa backend to support trtllm sparse fp8 attention...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunks: -33,7 +33,10; -340,6 +343,7 @@ def __init__(; symbols: __init__, forward_extend, touching `__init__, forward_extend`; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -1488,6 +1488,12 @@ def _fuse_rope_for_trtllm_mla(self, forward_batch: Forwar...; symbols: _fuse_rope_for_trtllm_mla, touching `_fuse_rope_for_trtllm_mla`; `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunks: -66,9 +66,13 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-...; `test/registered/kernels/test_nsa_indexer.py` modified +1/-0 (1 lines); hunks: -232,6 +232,7 @@ def __init__(self, config=None):; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunks: -33,7 +33,10; -340,6 +343,7 @@ def __init__(; symbols: __init__, forward_extend
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: -1488,6 +1488,12 @@ def _fuse_rope_for_trtllm_mla(self, forward_batch: Forwar...; symbols: _fuse_rope_for_trtllm_mla
  - `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunks: -66,9 +66,13 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-...
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
diff -- docs/basic_usage/deepseek_v32.md
@@ -66,9 +66,13 @@ python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --n
+  - `trtllm`: `trtllm-mla` sparse kernel from flashinfer library. Only run on blackwell GPUs. It requires QKV bf16 or QKV fp8.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66; `python/sglang/srt/models/deepseek_v2.py` modified +6/-0
  - docs: `docs/basic_usage/deepseek_v32.md` modified +4/-0
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/hicache/test_nsa_pool_host_unit.py`, `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18978 - [AMD] Fix mi35x dsv32 mtp nightly

- Link: https://github.com/sgl-project/sglang/pull/18978
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `462267982bd8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Fix mi35x dsv32 mtp nightly". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation PR #17554 introduced `fused_metadata_copy`, a JIT-compiled CUDA kernel enabled by default. This kernel cannot run on ROCm/HIP — it fails on every CUDA graph replay...
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

### PR #18931 - Fix NSA FP8 KV cache path for both-trtllm MHA one-shot

- Link: https://github.com/sgl-project/sglang/pull/18931
- Status/date: merged / 2026-02-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; associated commits `f23a23cc05fc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix NSA FP8 KV cache path for both-trtllm MHA one-shot". The diff centers on `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`. PR body context: ## Motivation This is a follow-up to #18389 for the NSA FP8 KV cache path case when both prefill and decode backends are trtllm. In this setup, the MHA one-shot prefix path coul...
- Key implementation: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +8/-1 (9 lines); hunks: -215,7 +215,14 @@ def forward_normal_prepare(; symbols: forward_normal_prepare, touching `forward_normal_prepare`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +8/-1 (9 lines); hunks: -215,7 +215,14 @@ def forward_normal_prepare(; symbols: forward_normal_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
@@ -215,7 +215,14 @@ def forward_normal_prepare(
-            if self.use_nsa and self.kv_cache_dtype == "fp8_e4m3":
+            if (
+                self.use_nsa
+                and self.kv_cache_dtype == "fp8_e4m3"
+                and (
+                    not get_global_server_args().nsa_decode_backend == "trtllm"
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19062 - [DSv32] Fix MTP and CP compatability

- Link: https://github.com/sgl-project/sglang/pull/19062
- Status/date: merged / 2026-02-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-5, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSv32] Fix MTP and CP compatability". The diff centers on `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Motivation PR https://github.com/sgl-project/sglang/pull/17213 introduced separate `get_attention_cp_size()` / `get_attention_cp_rank()` APIs and migrated the main model in `...
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +5/-5 (10 lines); hunks: -34,8 +34,8; -123,7 +123,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +5/-5 (10 lines); hunks: -34,8 +34,8; -123,7 +123,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -34,8 +34,8 @@
-    get_attention_tp_rank,
-    get_attention_tp_size,
+    get_attention_cp_rank,
+    get_attention_cp_size,
@@ -123,7 +123,7 @@ def __init__(
-            self.cp_size = get_attention_tp_size()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +5/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19134 - Fix spec v2+dp attention in nsa backend

- Link: https://github.com/sgl-project/sglang/pull/19134
- Status/date: merged / 2026-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/utils.py`; associated commits `8cf003c44b1b`
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix spec v2+dp attention in nsa backend". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`. PR body context: ## Motivation Close #18943 This bug is introduced in #17213 ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the...
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-1 (6 lines); hunks: -112,7 +112,11 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens, pad_nsa_cache_seqlens, touching `cal_padded_tokens, pad_nsa_cache_seqlens`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-1 (6 lines); hunks: -112,7 +112,11 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens, pad_nsa_cache_seqlens
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -112,7 +112,11 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):
-    if attn_cp_size == 1 or not can_nsa_prefill_cp_round_robin_split(forward_batch):
+    needs_cp_pad = attn_cp_size > 1 and can_nsa_prefill_cp_round_robin_split(
+        forward_batch
+    )
+    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
+    if not needs_cp_pad and not needs_dp_pad:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19041 - [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`

- Link: https://github.com/sgl-project/sglang/pull/19041
- Status/date: merged / 2026-02-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `eddf193292d3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +48/-9, 121 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`. PR body context: @humansand ## Motivation From recent GLM-5 tech report: > Compared with the non-deterministic CUDA-based top‑k implementation used in SGLang’s DSA Indexer, directly using the na...
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

### PR #19211 - [Refactor][DeepSeek-V3.2] Extract V3.2/NSA logic into `DeepseekV32Mixin`

- Link: https://github.com/sgl-project/sglang/pull/19211
- Status/date: open / 2026-02-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +298/-169, 669 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "[Refactor][DeepSeek-V3.2] Extract V3.2/NSA logic into `DeepseekV32Mixin`". The diff centers on `python/sglang/srt/models/deepseek_common/v32_mixin.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Motivation #16815 - part of #16255 cc @Fridge003 ## Accuracy Tests ### GSMK ### GPQA ### AIME ## Benchmark
- Key implementation: `python/sglang/srt/models/deepseek_common/v32_mixin.py` added +241/-0 (241 lines); hunks: -0,0 +1,241; symbols: DeepseekV32Mixin, init_v32_attention, _init_nsa_indexer, _fuse_rope_for_trtllm_mla, touching `DeepseekV32Mixin, init_v32_attention, _init_nsa_indexer`; `python/sglang/srt/models/deepseek_v2.py` modified +42/-120 (162 lines); hunks: -32,12 +32,6; -53,26 +47,16; symbols: DeepseekV2AttentionMLA, __init__, touching `DeepseekV2AttentionMLA, __init__`; `python/sglang/srt/models/deepseek_nextn.py` modified +15/-49 (64 lines); hunks: -21,24 +21,10; -49,6 +35,7; symbols: DeepseekModelNextN, __init__, forward, touching `DeepseekModelNextN, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/v32_mixin.py` added +241/-0 (241 lines); hunks: -0,0 +1,241; symbols: DeepseekV32Mixin, init_v32_attention, _init_nsa_indexer, _fuse_rope_for_trtllm_mla
  - `python/sglang/srt/models/deepseek_v2.py` modified +42/-120 (162 lines); hunks: -32,12 +32,6; -53,26 +47,16; symbols: DeepseekV2AttentionMLA, __init__
  - `python/sglang/srt/models/deepseek_nextn.py` modified +15/-49 (64 lines); hunks: -21,24 +21,10; -49,6 +35,7; symbols: DeepseekModelNextN, __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/v32_mixin.py
@@ -0,0 +1,241 @@
+# Copyright 2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -32,12 +32,6 @@
-from sglang.srt.configs.model_config import (
-    get_nsa_index_head_dim,
-    get_nsa_index_n_heads,
-    get_nsa_index_topk,
-    is_deepseek_nsa,
-)
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -21,24 +21,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/v32_mixin.py` added +241/-0; `python/sglang/srt/models/deepseek_v2.py` modified +42/-120; `python/sglang/srt/models/deepseek_nextn.py` modified +15/-49
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_common/v32_mixin.py`, `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19299 - [Perf] O(1) expert weight matching in DeepSeek weight loader

- Link: https://github.com/sgl-project/sglang/pull/19299
- Status/date: open / 2026-02-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +344/-29, 403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf] O(1) expert weight matching in DeepSeek weight loader". The diff centers on `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `test/unit/test_deepseek_weight_loader.py`. PR body context: ## Motivation Large MoE models like GLM-5 with 256 routed experts generate hundreds of thousands of weight tensors. The original weight loading code used an O(N*M) algorithm (li...
- Key implementation: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +67/-29 (96 lines); hunks: -14,8 +14,9; -84,6 +85,34 @@ class NextNDisabledConfig:; symbols: NextNDisabledConfig, _build_expert_proj_map, DeepseekV2WeightLoaderMixin, do_load_weights, touching `NextNDisabledConfig, _build_expert_proj_map, DeepseekV2WeightLoaderMixin`; `test/unit/test_deepseek_weight_loader.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _download_index, _download_config, _make_expert_params_mapping, _linear_scan_match, touching `_download_index, _download_config, _make_expert_params_mapping`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +67/-29 (96 lines); hunks: -14,8 +14,9; -84,6 +85,34 @@ class NextNDisabledConfig:; symbols: NextNDisabledConfig, _build_expert_proj_map, DeepseekV2WeightLoaderMixin, do_load_weights
  - `test/unit/test_deepseek_weight_loader.py` added +277/-0 (277 lines); hunks: -0,0 +1,277; symbols: _download_index, _download_config, _make_expert_params_mapping, _linear_scan_match
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py
@@ -14,8 +14,9 @@
+import re
-from typing import Iterable, List, Optional, Tuple
+from typing import Dict, Iterable, List, Optional, Tuple
@@ -84,6 +85,34 @@ class NextNDisabledConfig:
+# Regex to extract expert_id and projection name from weight names like
+# "model.layers.5.mlp.experts.42.gate_proj.weight" or "...experts.42.w1.input_scale"
diff -- test/unit/test_deepseek_weight_loader.py
@@ -0,0 +1,277 @@
+"""Unit tests for the O(1) expert weight matching optimization in deepseek_weight_loader.
+Downloads model.safetensors.index.json from real HuggingFace model repos and
+verifies that the new dict-based lookup produces identical results to the
+original linear scan for every weight tensor name.
+"""
+import json
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +67/-29
  - tests: `test/unit/test_deepseek_weight_loader.py` added +277/-0
- Risk and verification: The diff ships test coverage in `test/unit/test_deepseek_weight_loader.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19148 - [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache

- Link: https://github.com/sgl-project/sglang/pull/19148
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `4e843f121657`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +307/-21, 386 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`. PR body context: ## Motivation In DeepSeek v3.2, after the Indexer produces key in bf16 (roughly (N, 128)), it needs to populate NSA’s index_k_with_scale_buffer. The previous implementation used...
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

### PR #19367 - Fix NSA CP positions mismatch in eagle NextN model

- Link: https://github.com/sgl-project/sglang/pull/19367
- Status/date: merged / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix NSA CP positions mismatch in eagle NextN model". The diff centers on `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Summary - When context parallelism (CP) is enabled, the eagle NextN model (`deepseek_nextn.py`) splits `hidden_states` across CP ranks but does **not** split `positions`, cau...
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +2/-0 (2 lines); hunks: -29,6 +29,7; -160,6 +161,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-0 (2 lines); hunks: -29,6 +29,7; -160,6 +161,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -29,6 +29,7 @@
+    cp_split_and_rebuild_position,
@@ -160,6 +161,7 @@ def forward(
+            positions = cp_split_and_rebuild_position(forward_batch, positions)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17199 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/17199
- Status/date: closed / 2026-02-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +191/-82, 650 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Feature] add feature mla_ag_after_qlora for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/communicator.py`. PR body context: ## Motivation In DeepSeek-V3.2, due to the large hidden dimension of the model, performing all-gather on hidden_states before attention introduces a high-dimensional communicati...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-3 (22 lines); hunks: -35,7 +35,11; -968,6 +972,7 @@ def forward_npu(; symbols: forward_npu, touching `forward_npu`; `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -170,7 +170,7; -1566,6 +1566,7 @@ def forward(; symbols: forward, forward_prepare, __init__, touching `forward, forward_prepare, __init__`; `python/sglang/srt/layers/communicator.py` modified +4/-1 (5 lines); hunks: -68,6 +68,7; -151,7 +152,7 @@ def __init__(self):; symbols: __init__, init_context, get_fn, touching `__init__, init_context, get_fn`; `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +119/-70 (189 lines); hunks: -7,18 +7,17; -240,12 +239,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, init_forward_metadata, _generate_alibi_bias, generate_alibi_bias, touching `__init__, init_forward_metadata, _generate_alibi_bias`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-3 (22 lines); hunks: -35,7 +35,11; -968,6 +972,7 @@ def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunks: -170,7 +170,7; -1566,6 +1566,7 @@ def forward(; symbols: forward, forward_prepare, __init__
  - `python/sglang/srt/layers/communicator.py` modified +4/-1 (5 lines); hunks: -68,6 +68,7; -151,7 +152,7 @@ def __init__(self):; symbols: __init__, init_context, get_fn
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +119/-70 (189 lines); hunks: -7,18 +7,17; -240,12 +239,17 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, init_forward_metadata, _generate_alibi_bias, generate_alibi_bias
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-6 (40 lines); hunks: -12,12 +12,15; -76,7 +79,9 @@ def forward_mha_prepare_npu(; symbols: forward_mha_prepare_npu, forward_dsa_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -35,7 +35,11 @@
+from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import scattered_to_tp_attn_full
+from sglang.srt.utils import get_bool_env_var
+from sglang.srt.layers.communicator import ScatterMode
+_use_ag_after_qlora = get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA")
@@ -968,6 +972,7 @@ def forward_npu(
+        layer_scatter_modes,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -170,7 +170,7 @@
+_use_ag_after_qlora = get_bool_env_var("SGLANG_USE_AG_AFTER_QLORA")
@@ -1566,6 +1566,7 @@ def forward(
+        layer_scatter_modes: LayerScatterModes,
@@ -1574,6 +1575,7 @@ def forward(
+            layer_scatter_modes=layer_scatter_modes,
@@ -1583,6 +1585,7 @@ def forward_prepare(
diff -- python/sglang/srt/layers/communicator.py
@@ -68,6 +68,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-3; `python/sglang/srt/models/deepseek_v2.py` modified +15/-2; `python/sglang/srt/layers/communicator.py` modified +4/-1; `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +119/-70; `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18319 - [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct

- Link: https://github.com/sgl-project/sglang/pull/18319
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `9496bbd7b11c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 23 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: As per title. This PR adds a default on Instinct to `tilelang` for NSA attention backend. "flashmla_sparse" and "fa3" are not supported on ROCm afaik. Seems like part of this wa...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: -1477,7 +1477,9 @@ def forward_extend(; symbols: forward_extend, forward_decode, touching `forward_extend, forward_decode`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: -1477,7 +1477,9 @@ def forward_extend(; symbols: forward_extend, forward_decode
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1477,7 +1477,9 @@ def forward_extend(
-            raise ValueError(f"Unsupported {nsa_impl = }")
+            raise ValueError(
+                f"Unsupported {nsa_impl = } for forward_extend. Consider using an other attention backend."
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18526 - [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr…

- Link: https://github.com/sgl-project/sglang/pull/18526
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `7e46aafebb6d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +130/-3, 176 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr…". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`. PR body context: ## Motivation Bool mask produced dynamic shape that could not be captured into cuda graph. Capture NSA decode implemented by aiter backend into cuda graph for better performance...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +70/-3 (73 lines); hunks: -52,6 +52,8; -333,6 +335,12 @@ def __init__(; symbols: __init__, forward_extend, _forward_aiter, _forward_aiter_extend, touching `__init__, forward_extend, _forward_aiter`; `python/sglang/srt/layers/attention/nsa/triton_kernel.py` modified +60/-0 (60 lines); hunks: -134,3 +134,63 @@ def act_quant(; symbols: act_quant, _get_valid_kv_indices_kernel, get_valid_kv_indices, touching `act_quant, _get_valid_kv_indices_kernel, get_valid_kv_indices`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +70/-3 (73 lines); hunks: -52,6 +52,8; -333,6 +335,12 @@ def __init__(; symbols: __init__, forward_extend, _forward_aiter, _forward_aiter_extend
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` modified +60/-0 (60 lines); hunks: -134,3 +134,63 @@ def act_quant(; symbols: act_quant, _get_valid_kv_indices_kernel, get_valid_kv_indices
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -52,6 +52,8 @@
+    from sglang.srt.layers.attention.nsa.triton_kernel import get_valid_kv_indices
@@ -333,6 +335,12 @@ def __init__(
+            self.kv_indices = torch.zeros(
+                max_bs * self.nsa_index_topk,
+                dtype=torch.int32,
+                device=self.device,
diff -- python/sglang/srt/layers/attention/nsa/triton_kernel.py
@@ -134,3 +134,63 @@ def act_quant(
+@triton.jit
+def _get_valid_kv_indices_kernel(
+    page_table_ptr,  # [bs, topk]
+    kv_indptr_ptr,  # [bs + 1]
+    kv_indices_ptr,  # [bs * topk] output buffer
+    bs: tl.constexpr,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +70/-3; `python/sglang/srt/layers/attention/nsa/triton_kernel.py` modified +60/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19122 - [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py

- Link: https://github.com/sgl-project/sglang/pull/19122
- Status/date: merged / 2026-02-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +906/-818, 1896 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`. PR body context: ## Motivation Part of #16255 Split different MLA forward methods into separate files, including: - `forward_mla.py` for Cuda/HIP mla aborption - `forward_mla_fused_rope_cpu.py‎`...
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

### PR #19609 - TP indexer weight in NSA attention

- Link: https://github.com/sgl-project/sglang/pull/19609
- Status/date: open / 2026-03-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-9, 96 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR extends deployment docs, tests, or CI coverage. Title: "TP indexer weight in NSA attention". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Tested at BS = 4, ISL=2048 OSL=1024 Before: After: Only seems to be around 2%. Currently, the all-reduce in TP=4 will take the same time as the size of the indexer...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +41/-9 (50 lines); hunks: -36,6 +36,9; -45,7 +48,8; symbols: __init__, _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, touching `__init__, _with_real_sm_count, _weights_proj_bf16_in_fp32_out`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +41/-9 (50 lines); hunks: -36,6 +36,9; -45,7 +48,8; symbols: __init__, _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -36,6 +36,9 @@
+    get_tensor_model_parallel_rank,
+    get_tensor_model_parallel_world_size,
+    tensor_model_parallel_all_reduce,
@@ -45,7 +48,8 @@
-from sglang.srt.layers.linear import ReplicatedLinear
+from sglang.srt.layers.dp_attention import is_dp_attention_enabled
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +41/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19536 - [Perf] Optimize NSA backend metadata under MTP

- Link: https://github.com/sgl-project/sglang/pull/19536
- Status/date: merged / 2026-03-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `80a6b32703db`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +85/-64, 191 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf] Optimize NSA backend metadata under MTP". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation This PR is authored by @Baidu-AIAK in https://github.com/sgl-project/sglang/pull/17647. I just tested the code, since the author hasn't replied in a while. ## Accu...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-64 (88 lines); hunks: -36,6 +36,7; -434,24 +435,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph, touching `init_forward_metadata, init_forward_metadata_replay_cuda_graph`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-64 (88 lines); hunks: -36,6 +36,7; -434,24 +435,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -36,6 +36,7 @@
+    seqlens_expand_triton,
@@ -434,24 +435,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            seqlens_int32_cpu = [
-                self.speculative_num_draft_tokens + kv_len
-                for kv_len in forward_batch.seq_lens_cpu.tolist()
-            ]
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-64
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17647 - [Perf] opt nsa backend init forward metada

- Link: https://github.com/sgl-project/sglang/pull/17647
- Status/date: closed / 2026-03-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +88/-64, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Perf] opt nsa backend init forward metada". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`. PR body context: ## Motivation During Dsa backend init forward metadate, it use for-loop to get the Seqlen_expanded. when BS is large, It's very time-consuming. ## Modifications We replaced the...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +27/-64 (91 lines); hunks: -30,6 +30,7; -404,24 +405,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph, touching `init_forward_metadata, init_forward_metadata_replay_cuda_graph`; `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunks: -279,6 +279,67 @@ def pad_sequence_with_mask(; symbols: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton, touching `pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +27/-64 (91 lines); hunks: -30,6 +30,7; -404,24 +405,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunks: -279,6 +279,67 @@ def pad_sequence_with_mask(; symbols: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -30,6 +30,7 @@
+from sglang.srt.layers.attention.utils import seqlens_expand_triton
@@ -404,24 +405,11 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
-            seqlens_int32_cpu = [
-                self.speculative_num_draft_tokens + kv_len
-                for kv_len in forward_batch.seq_lens_cpu.tolist()
-            ]
diff -- python/sglang/srt/layers/attention/utils.py
@@ -279,6 +279,67 @@ def pad_sequence_with_mask(
+@triton.jit
+def seqlens_expand_kernel(
+    extend_seq_lens_ptr,  # [N]
+    seq_lens_ptr,  # [N]
+    offsets_ptr,  # [N+1]
+    output_ptr,  # [sum(extend_seq_lens)]
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +27/-64; `python/sglang/srt/layers/attention/utils.py` modified +61/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19428 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- Link: https://github.com/sgl-project/sglang/pull/19428
- Status/date: merged / 2026-03-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `b3718982a1b4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +101/-9, 316 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Feature] add feature mla_ag_after_qlora for dsv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: ## Motivation In DeepSeek-V3.2, due to the large hidden dimension of the model, performing all-gather on hidden_states before attention introduces a high-dimensional communicati...
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
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation LLMs may generate tool call content that are not correct json format, we must catch this error for normal streaming output. ## Modifications ## Accuracy Tests ## B...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format.". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation The previous implementation only supported streaming function call parameters in XML format. ## Modifications 1. Convert `_parse_parameters_from_xml` to return `st...
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

### PR #19829 - [NSA] Fix line-too-long lint in `can_nsa_prefill_cp_round_robin_split`

- Link: https://github.com/sgl-project/sglang/pull/19829
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/utils.py`; associated commits `b7f7df7ee6c2`
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-1, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[NSA] Fix line-too-long lint in `can_nsa_prefill_cp_round_robin_split`". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`. PR body context: - Reformat long boolean return expression in `can_nsa_prefill_cp_round_robin_split` to comply with line-length lint rule Made with Cursor
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +6/-1 (7 lines); hunks: -54,7 +54,12 @@ def can_nsa_prefill_cp_round_robin_split(forward_batch: "Forw...; symbols: can_nsa_prefill_cp_round_robin_split, nsa_cp_round_robin_split_data, touching `can_nsa_prefill_cp_round_robin_split, nsa_cp_round_robin_split_data`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +6/-1 (7 lines); hunks: -54,7 +54,12 @@ def can_nsa_prefill_cp_round_robin_split(forward_batch: "Forw...; symbols: can_nsa_prefill_cp_round_robin_split, nsa_cp_round_robin_split_data
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -54,7 +54,12 @@ def can_nsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
-    return is_nsa_prefill_cp_round_robin_split() and seq_len > 0 and seq_len >= cp_size and cp_size > 1
+    return (
+        is_nsa_prefill_cp_round_robin_split()
+        and seq_len > 0
+        and seq_len >= cp_size
+        and cp_size > 1
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +6/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19975 - [AMD] Support context parallel for DeepSeek-V3.2 on AMD GPUs and add its test to nightly CI

- Link: https://github.com/sgl-project/sglang/pull/19975
- Status/date: open / 2026-03-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +119/-38, 232 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Support context parallel for DeepSeek-V3.2 on AMD GPUs and add its test to nightly CI". The diff centers on `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py`, `.github/workflows/nightly-test-amd-rocm720.yml`. PR body context: ## Motivation This PR adds and validates context parallel support for DeepSeek-V3.2 on AMD GPUs, focusing on the `round-robin-split` CP mode with AITER NSA backends. The goal is...
- Key implementation: `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` modified +49/-21 (70 lines); hunks: -1,12 +1,15; -32,6 +35,19; symbols: TestDeepseekV32CPSingleNode, test_deepseek_v32_cp_variants, touching `TestDeepseekV32CPSingleNode, test_deepseek_v32_cp_variants`; `python/sglang/srt/server_args.py` modified +26/-16 (42 lines); hunks: -1453,24 +1453,34 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `.github/workflows/nightly-test-amd-rocm720.yml` modified +31/-0 (31 lines); hunks: -457,6 +457,37 @@ jobs:; `docs/basic_usage/deepseek_v32.md` modified +13/-1 (14 lines); hunks: -54,6 +54,8 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...; -306,7 +308,7 @@ DeepSeek-V3.2-Speciale:.
- Code diff details:
  - `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` modified +49/-21 (70 lines); hunks: -1,12 +1,15; -32,6 +35,19; symbols: TestDeepseekV32CPSingleNode, test_deepseek_v32_cp_variants
  - `python/sglang/srt/server_args.py` modified +26/-16 (42 lines); hunks: -1453,24 +1453,34 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +31/-0 (31 lines); hunks: -457,6 +457,37 @@ jobs:
  - `docs/basic_usage/deepseek_v32.md` modified +13/-1 (14 lines); hunks: -54,6 +54,8 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3...; -306,7 +308,7 @@ DeepSeek-V3.2-Speciale:
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py
@@ -1,12 +1,15 @@
-from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
-from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
+from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system, is_hip
+is_hip = is_hip()
+register_amd_ci(
diff -- python/sglang/srt/server_args.py
@@ -1453,24 +1453,34 @@ def _handle_model_specific_adjustments(self):
-                        logger.warning(
-                            "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
-                        )
-                        if self.nsa_prefill_cp_mode == "in-seq-split":
-                            # TODO Supports moe_dense_tp_size != 1, kv cache dtype = "fp8",moe_a2a_backend non-deepep and cross-machine operation .
-                            self.enable_dp_attention = True
diff -- .github/workflows/nightly-test-amd-rocm720.yml
@@ -457,6 +457,37 @@ jobs:
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` modified +49/-21
  - runtime: `python/sglang/srt/server_args.py` modified +26/-16
  - ci: `.github/workflows/nightly-test-amd-rocm720.yml` modified +31/-0
  - docs: `docs/basic_usage/deepseek_v32.md` modified +13/-1
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19987 - [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture

- Link: https://github.com/sgl-project/sglang/pull/19987
- Status/date: closed / 2026-03-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 14 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation PR #19263 removed the `forward_native` override for rotary embeddings in DeepSeek V3.2 (`self.rotary_emb.forward = self.rotary_emb.forward_native`). Previously, `f...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-0 (7 lines); hunks: -313,6 +313,13 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16, touching `_get_q_k_bf16`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-0 (7 lines); hunks: -313,6 +313,13 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -313,6 +313,13 @@ def _get_q_k_bf16(
+        # On ROCm, CUDA graph capture rejects in-place writes to aliased
+        # tensors (q_rope / k_rope are torch.split views of query / key).
+        # Cloning breaks the alias so the write-back succeeds.
+        if _is_hip:
+            q_rope = q_rope.clone()
+            k_rope = k_rope.clone()
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19016 - [FIX] NSA backend page_table overflow in speculative decoding target_verify

- Link: https://github.com/sgl-project/sglang/pull/19016
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `9ebffef1ef95`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[FIX] NSA backend page_table overflow in speculative decoding target_verify". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation This PR may address #18980 . When speculative decoding is enabled with the NSA attention backend, the decode server crashes with: Root cause: - init_cuda_graph_sta...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: -766,9 +766,11 @@ def init_cuda_graph_state(self, max_bs: int, max_num_tokens...; symbols: init_cuda_graph_state, touching `init_cuda_graph_state`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunks: -766,9 +766,11 @@ def init_cuda_graph_state(self, max_bs: int, max_num_tokens...; symbols: init_cuda_graph_state
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -766,9 +766,11 @@ def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
+            # Add extra columns for speculative draft tokens to avoid
+            # overflow during target_verify when max_seqlen_k = seq_len + num_draft_tokens
-                self.max_context_len,
+                self.max_context_len + (self.speculative_num_draft_tokens or 0),
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19985 - [V32] Enhance deepseek v32 related tests

- Link: https://github.com/sgl-project/sglang/pull/19985
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; associated commits `04e364d538bb`
- Diff scope read: GitHub Pull Request files API returned 4 files, +477/-12, 572 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[V32] Enhance deepseek v32 related tests". The diff centers on `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`. PR body context: ## Motivation - Add spec v2 tests - Add mtp tests for fp4 model ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to...
- Key implementation: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass`; `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +86/-4 (90 lines); hunks: -4,6 +4,7; -12,13 +13,13; symbols: TestDeepseekV32FP4, TestDeepseekV32FP4DP, setUpClass, test_a_gsm8k, touching `TestDeepseekV32FP4, TestDeepseekV32FP4DP, setUpClass`.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` added +213/-0 (213 lines); hunks: -0,0 +1,213; symbols: TestDeepseekV32FP4DPSpecV2, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +86/-4 (90 lines); hunks: -4,6 +4,7; -12,13 +13,13; symbols: TestDeepseekV32FP4, TestDeepseekV32FP4DP, setUpClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -0,0 +1,213 @@
+import unittest
+from types import SimpleNamespace
+import requests
+from sglang.srt.environ import envs
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -4,6 +4,7 @@
+from sglang.test.send_one import BenchArgs, send_one_prompt
@@ -12,13 +13,13 @@
-register_cuda_ci(est_time=600, suite="stage-c-test-4-gpu-b200")
+register_cuda_ci(est_time=1200, suite="stage-c-test-4-gpu-b200")
-class TestDeepseekV32FP4(CustomTestCase):
+class TestDeepseekV32FP4DP(CustomTestCase):
```

- Reviewed files:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` added +213/-0; `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +86/-4
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_basic.py`, `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20086 - [V32/GLM5] Change default setting of V32 nvfp4 on TP4

- Link: https://github.com/sgl-project/sglang/pull/20086
- Status/date: merged / 2026-03-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +15/-6, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[V32/GLM5] Change default setting of V32 nvfp4 on TP4". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation After the flashmla is rebased to latest version, DeepSeekV32 fp4+tp4 will break on flashmla decode kernel, since the latest code doesn't support q_head=32 error lo...
- Key implementation: `python/sglang/srt/server_args.py` modified +15/-6 (21 lines); hunks: -1271,7 +1271,9 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; -1290,11 +1292,18 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str,...; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends, touching `_set_default_nsa_kv_cache_dtype, _set_default_nsa_backends`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +15/-6 (21 lines); hunks: -1271,7 +1271,9 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; -1290,11 +1292,18 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str,...; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -1271,7 +1271,9 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:
-            self.kv_cache_dtype = "fp8_e4m3" if major >= 10 else "bfloat16"
+            self.kv_cache_dtype = (
+                "fp8_e4m3" if (major >= 10 and self.dp_size > 1) else "bfloat16"
+            )
@@ -1290,11 +1292,18 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:
-            # flashmla_auto dispatches to flashmla_sparse/flashmla_kv based on hardware and heuristics
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +15/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20062 - [V32/GLM5] Control the threshold of applying dense attention with an environ

- Link: https://github.com/sgl-project/sglang/pull/20062
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; associated commits `be63f982b7b7`
- Diff scope read: GitHub Pull Request files API returned 6 files, +32/-59, 200 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[V32/GLM5] Control the threshold of applying dense attention with an environ". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`. PR body context: ## Motivation - Add an environ `SGLANG_NSA_DENSE_ATTN_KV_LEN_THRESHOLD`, for controlling whether to use dense MHA or sparse MLA kernel. It's set to index.topk by default, thus n...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl, touching `NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed`; `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46 (49 lines); hunks: -16,10 +16,6; -71,15 +67,10; symbols: NSAFlashMLAMetadata, __init__, init_forward_metadata_replay_cuda_graph_from_precomputed, set_nsa_prefill_impl
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4 (4 lines); hunks: -34,8 +34,6 @@ def setUpClass(cls):; -103,8 +101,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4 (4 lines); hunks: -39,8 +39,6 @@ def setUpClass(cls):; -131,8 +129,6 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -16,10 +16,6 @@
-from sglang.srt.layers.attention.nsa.nsa_mtp_verification import (
-    verify_multi_backend_fused_metadata_copy,
-    verify_single_backend_fused_metadata_copy,
-)
@@ -71,15 +67,10 @@
-# Control whether to use fused metadata copy kernel (default: enabled)
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -34,8 +34,6 @@ def setUpClass(cls):
-            "--kv-cache-dtype",
-            "fp8_e4m3",
@@ -103,8 +101,6 @@ def setUpClass(cls):
-            "--kv-cache-dtype",
-            "fp8_e4m3",
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -39,8 +39,6 @@ def setUpClass(cls):
-            "--kv-cache-dtype",
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-46
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-4; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-4
- Risk and verification: The diff ships test coverage in `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18876 - Add DeepSeek3.2 and GlmMoeDsa into moe tune

- Link: https://github.com/sgl-project/sglang/pull/18876
- Status/date: merged / 2026-03-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-0, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add DeepSeek3.2 and GlmMoeDsa into moe tune". The diff centers on `benchmark/kernels/fused_moe_triton/common_utils.py`. PR body context: ## Motivation As title. tuning_fused_moe_triton_sep.py uses `get_model_config` in common_utils to choose the moe model. This PR is to count DeepSeek V3.2 and GLM-5(which uses Gl...
- Key implementation: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +4/-0 (4 lines); hunks: -74,7 +74,9 @@ def get_model_config(; -83,7 +85,9 @@ def get_model_config(; symbols: get_model_config, touching `get_model_config`.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +4/-0 (4 lines); hunks: -74,7 +74,9 @@ def get_model_config(; -83,7 +85,9 @@ def get_model_config(; symbols: get_model_config
- Key code excerpts:

```diff
diff -- benchmark/kernels/fused_moe_triton/common_utils.py
@@ -74,7 +74,9 @@ def get_model_config(
+        "DeepseekV32ForCausalLM",
+        "GlmMoeDsaForCausalLM",
@@ -83,7 +85,9 @@ def get_model_config(
+                "DeepseekV32ForCausalLM",
+                "GlmMoeDsaForCausalLM",
```

- Reviewed files:
  - other: `benchmark/kernels/fused_moe_triton/common_utils.py` modified +4/-0
- Risk and verification: No explicit test file appears in the diff; future edits should add or run model loading, short generation, and parser/multimodal regression checks.

### PR #20360 - [AMD][Bug fix] Fix NSA context parallelism (round-robin-split) producing garbage output

- Link: https://github.com/sgl-project/sglang/pull/20360
- Status/date: open / 2026-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +31/-5, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD][Bug fix] Fix NSA context parallelism (round-robin-split) producing garbage output". The diff centers on `python/sglang/srt/layers/communicator_nsa_cp.py`. PR body context: ## Summary Fix garbage output when using `--enable-nsa-prefill-context-parallel` with `--attn-cp-size 2` on multi-GPU setups (observed on 8xMI300X with GLM-5-FP8 / DeepSeek-V3.2...
- Key implementation: `python/sglang/srt/layers/communicator_nsa_cp.py` modified +31/-5 (36 lines); hunks: -20,6 +20,7; -34,6 +35,7; symbols: __init__, should_use_reduce_scatter, _post_init_communicate, get_fn, touching `__init__, should_use_reduce_scatter, _post_init_communicate`.
- Code diff details:
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +31/-5 (36 lines); hunks: -20,6 +20,7; -34,6 +35,7; symbols: __init__, should_use_reduce_scatter, _post_init_communicate, get_fn
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator_nsa_cp.py
@@ -20,6 +20,7 @@
+    is_nsa_prefill_cp_round_robin_split,
@@ -34,6 +35,7 @@
+    attn_tp_all_reduce,
@@ -66,6 +68,11 @@ def __init__(
+    def should_use_reduce_scatter(self, forward_batch) -> bool:
+        if is_nsa_prefill_cp_round_robin_split():
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator_nsa_cp.py` modified +31/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator_nsa_cp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20326 - [Doc] Add DSA/NSA attention backend to support matrix

- Link: https://github.com/sgl-project/sglang/pull/20326
- Status/date: merged / 2026-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-1, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Doc] Add DSA/NSA attention backend to support matrix". The diff centers on `docs/advanced_features/attention_backend.md`. PR body context: ## Summary - Adds NSA (DSA) row to the MLA backends table with its capabilities (FP8 KV cache, speculative decoding topk=1, chunked prefix cache) - Adds a new "DSA Attention Bac...
- Key implementation: `docs/advanced_features/attention_backend.md` modified +19/-1 (20 lines); hunks: -50,7 +50,7 @@ Multimodal attention is selected by `--mm-attention-backend`....; -107,6 +107,24 @@ GDN models are hybrid: the full-attention layers still requ....
- Code diff details:
  - `docs/advanced_features/attention_backend.md` modified +19/-1 (20 lines); hunks: -50,7 +50,7 @@ Multimodal attention is selected by `--mm-attention-backend`....; -107,6 +107,24 @@ GDN models are hybrid: the full-attention layers still requ...
- Key code excerpts:

```diff
diff -- docs/advanced_features/attention_backend.md
@@ -50,7 +50,7 @@ Multimodal attention is selected by `--mm-attention-backend`. The "MultiModal" c
-- NSA is specifically designed for [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/).
+- NSA is specifically designed for [DeepSeek V3.2 DSA](https://lmsys.org/blog/2025-09-29-deepseek-V32/). See the [DSA Attention Backend (NSA)](#dsa-attention-backend-nsa) section
@@ -107,6 +107,24 @@ GDN models are hybrid: the full-attention layers still require a standard `--att
+### DSA Attention Backend (NSA)
+DSA (Deepseek Sparse Attention) is a native sparse attention mechanism used by [DeepSeek V3.2](https://lmsys.org/blog/2025-09-29-deepseek-V32/). It is activated automatically when
+Internally, the NSA backend dispatches to different sub-backends for prefill and decode phases. You can override these with `--nsa-prefill-backend` and `--nsa-decode-backend`:
```

- Reviewed files:
  - docs: `docs/advanced_features/attention_backend.md` modified +19/-1
- Risk and verification: This is mostly docs/examples in `docs/advanced_features/attention_backend.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #19319 - [deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug

- Link: https://github.com/sgl-project/sglang/pull/19319
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`; associated commits `006bd44cf920`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +380/-81, 740 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[deepseekv3.2] fix get_k_and_s_triton kenel for 128K seqlen case bug". The diff centers on `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`. PR body context: ## Motivation fix get_k_and_s_triton kenel for 128K seqlen case bug ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [x] Format your code accordin...
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

### PR #20531 - [bugfix] Fix NSA indexer ragged gather batch-view mismatch in CP round-robin split

- Link: https://github.com/sgl-project/sglang/pull/20531
- Status/date: open / 2026-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-3, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[bugfix] Fix NSA indexer ragged gather batch-view mismatch in CP round-robin split". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation This PR fixes a crash in the NSA indexer prefill path when context parallelism is enabled with round-robin-split. The issue was caused by mixing two different batc...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-3 (10 lines); hunks: -538,11 +538,15 @@ def _get_topk_ragged(; symbols: _get_topk_ragged, touching `_get_topk_ragged`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-3 (10 lines); hunks: -538,11 +538,15 @@ def _get_topk_ragged(; symbols: _get_topk_ragged
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -538,11 +538,15 @@ def _get_topk_ragged(
-        seq_len_sum = forward_batch.seq_lens_sum
-        max_seq_len = torch.max(forward_batch.seq_lens_cpu).item()
+        # In CP round-robin-split, page tables may be filtered by metadata. The KV
+        # gather inputs must use the same filtered batch view to stay aligned.
+        seq_lens = metadata.get_seqlens_int32()
+        seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20534 - Transfer FP8 K/K_scale for CP indexer prefill gather

- Link: https://github.com/sgl-project/sglang/pull/20534
- Status/date: open / 2026-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +35/-8, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "Transfer FP8 K/K_scale for CP indexer prefill gather". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation This PR optimizes the NSA indexer CP prefill path in round-robin-split mode. Instead of: - all-gathering reordered bf16 indexer keys - then quantizing and storing...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +35/-8 (43 lines); hunks: -333,14 +333,6 @@ def _get_q_k_bf16(; -958,6 +950,41 @@ def _store_index_k_cache(; symbols: _get_q_k_bf16, _get_k_bf16, _store_index_k_cache, touching `_get_q_k_bf16, _get_k_bf16, _store_index_k_cache`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +35/-8 (43 lines); hunks: -333,14 +333,6 @@ def _get_q_k_bf16(; -958,6 +950,41 @@ def _store_index_k_cache(; symbols: _get_q_k_bf16, _get_k_bf16, _store_index_k_cache
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -333,14 +333,6 @@ def _get_q_k_bf16(
-        # allgather+rerrange
-        if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
-            key = cp_all_gather_rerange_output(
-                key.contiguous(),
-                self.cp_size,
-                forward_batch,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +35/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18280 - [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp

- Link: https://github.com/sgl-project/sglang/pull/18280
- Status/date: merged / 2026-03-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`; associated commits `17031120b8f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-4, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`. PR body context: ## Motivation Fix PR #16043 #19319 Prefill CP scenario bug. In the DeepSeek V3.2 prefill with CP (context parallel) scenario, `forward_batch.seq_lens` and `forward_batch.seq_len...
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

### PR #20809 - [Bugfix] Add DeepseekV32ForCausalLM to MTP draft model mapping

- Link: https://github.com/sgl-project/sglang/pull/20809
- Status/date: open / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bugfix] Add DeepseekV32ForCausalLM to MTP draft model mapping". The diff centers on `python/sglang/srt/configs/model_config.py`. PR body context: ## Motivation Speculative decoding (EAGLE/NEXTN MTP) does not work with DeepSeek-V3.2 models (`DeepseekV32ForCausalLM`). The `_config_draft_model()` function only maps `Deepseek...
- Key implementation: `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -279,6 +279,7 @@ def _config_draft_model(self):; symbols: _config_draft_model, touching `_config_draft_model`.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunks: -279,6 +279,7 @@ def _config_draft_model(self):; symbols: _config_draft_model
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/model_config.py
@@ -279,6 +279,7 @@ def _config_draft_model(self):
+            "DeepseekV32ForCausalLM",
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/model_config.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20840 - [AMD] Fix dpsk-v32 accuracy issue on mi355

- Link: https://github.com/sgl-project/sglang/pull/20840
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[AMD] Fix dpsk-v32 accuracy issue on mi355". The diff centers on `python/sglang/srt/layers/quantization/fp8_utils.py`. PR body context: ## Motivation DeepSeek-V3.2 accuracy drops from ~95% to 0% on MI355. nightly CI test Root Cause: - PR#19826 changed the fp8 block-scaled gemm dispatch on gfx950 — for shapes not...
- Key implementation: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunks: -68,6 +68,7 @@ def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) ->...; symbols: use_aiter_triton_gemm_w8a8_tuned_gfx950, touching `use_aiter_triton_gemm_w8a8_tuned_gfx950`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunks: -68,6 +68,7 @@ def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) ->...; symbols: use_aiter_triton_gemm_w8a8_tuned_gfx950
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -68,6 +68,7 @@ def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) -> bool:
+        (7168, 2304),
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/fp8_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20880 - Reject HiCache L3 storage backend for NSA models at init time

- Link: https://github.com/sgl-project/sglang/pull/20880
- Status/date: open / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-1, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Reject HiCache L3 storage backend for NSA models at init time". The diff centers on `python/sglang/srt/mem_cache/hiradix_cache.py`. PR body context: ## Summary - Raise `ValueError` at startup when `--hicache-storage-backend` is used with NSA (Native Sparse Attention) models (e.g. GLM-5, DeepSeek-V3.2) - The L3 storage layer...
- Key implementation: `python/sglang/srt/mem_cache/hiradix_cache.py` modified +13/-1 (14 lines); hunks: -82,6 +82,15 @@ def __init__(self, params: CacheInitParams, server_args: Serv...; -100,7 +109,10 @@ def __init__(self, params: CacheInitParams, server_args: Se...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/mem_cache/hiradix_cache.py` modified +13/-1 (14 lines); hunks: -82,6 +82,15 @@ def __init__(self, params: CacheInitParams, server_args: Serv...; -100,7 +109,10 @@ def __init__(self, params: CacheInitParams, server_args: Se...; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/hiradix_cache.py
@@ -82,6 +82,15 @@ def __init__(self, params: CacheInitParams, server_args: ServerArgs):
+            if server_args.hicache_storage_backend is not None:
+                raise ValueError(
+                    "HiCache L3 storage backend (e.g. mooncake) is not yet supported "
+                    "for models with NSA (Native Sparse Attention). The L3 storage "
+                    "layer does not handle the NSA indexer cache "
+                    "(index_k_with_scale_buffer). Please remove "
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/hiradix_cache.py` modified +13/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/hiradix_cache.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20492 - [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode

- Link: https://github.com/sgl-project/sglang/pull/20492
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation The current DeepSeek Eagle3 implementation has an issue in DP scenarios when using tensor_model_parallel_all_gather. The gather should be performed across the atte...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -43,7 +43,6; -73,6 +72,7; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunks: -43,7 +43,6; -73,6 +72,7; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -43,7 +43,6 @@
-    tensor_model_parallel_all_gather,
@@ -73,6 +72,7 @@
+    get_attention_tp_group,
@@ -1979,7 +1979,7 @@ def forward(
-                        aux_hidden_state = tensor_model_parallel_all_gather(
+                        aux_hidden_state = get_attention_tp_group().all_gather(
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17024 - [PD] Fix DeepSeek V3.2 indexer cache transfer

- Link: https://github.com/sgl-project/sglang/pull/17024
- Status/date: closed / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-10, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[PD] Fix DeepSeek V3.2 indexer cache transfer". The diff centers on `python/sglang/srt/disaggregation/prefill.py`. PR body context: ## Motivation A bugfix by @whybeyoung , we will verify it together tomorrow. ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code...
- Key implementation: `python/sglang/srt/disaggregation/prefill.py` modified +6/-10 (16 lines); hunks: -677,6 +677,12 @@ def send_kv_chunk(; -711,17 +717,7 @@ def send_kv_chunk(; symbols: send_kv_chunk, touching `send_kv_chunk`.
- Code diff details:
  - `python/sglang/srt/disaggregation/prefill.py` modified +6/-10 (16 lines); hunks: -677,6 +677,12 @@ def send_kv_chunk(; -711,17 +717,7 @@ def send_kv_chunk(; symbols: send_kv_chunk
- Key code excerpts:

```diff
diff -- python/sglang/srt/disaggregation/prefill.py
@@ -677,6 +677,12 @@ def send_kv_chunk(
+        page_indices = kv_to_page_indices(kv_indices, page_size)
+        # For NSA, state_indices (Indexer Cache) maps 1:1 to page_indices, so we send the Indexer Cache for each chunk
+        if isinstance(self.token_to_kv_pool_allocator.get_kvcache(), NSATokenToKVPool):
+            state_indices = page_indices
@@ -711,17 +717,7 @@ def send_kv_chunk(
-            elif isinstance(
```

- Reviewed files:
  - runtime: `python/sglang/srt/disaggregation/prefill.py` modified +6/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/prefill.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20984 - Fix DeepSeek V32 FP4 test

- Link: https://github.com/sgl-project/sglang/pull/20984
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; associated commits `c82d20d48ecc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +20/-1, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix DeepSeek V32 FP4 test". The diff centers on `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +9/-0 (9 lines); hunks: -1,3 +1,4; -46,6 +47,10 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +9/-0 (9 lines); hunks: -1,3 +1,4; -60,6 +61,10 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +9/-0 (9 lines); hunks: -1,3 +1,4; -46,6 +47,10 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +9/-0 (9 lines); hunks: -1,3 +1,4; -60,6 +61,10 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -1,3 +1,4 @@
+import os
@@ -46,6 +47,10 @@ def setUpClass(cls):
+            env={
+                **os.environ,
+                "HF_HUB_OFFLINE": "0",
+            },
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,3 +1,4 @@
+import os
@@ -60,6 +61,10 @@ def setUpClass(cls):
+                env={
+                    **os.environ,
+                    "HF_HUB_OFFLINE": "0",
+                },
```

- Reviewed files:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +9/-0; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +9/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21003 - Revert "Fix DeepSeek V32 FP4 test"

- Link: https://github.com/sgl-project/sglang/pull/21003
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; associated commits `a0a4dae67f5f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +1/-20, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Revert "Fix DeepSeek V32 FP4 test"". The diff centers on `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`. PR body context: Reverts sgl-project/sglang#20984 because it breaks ci
- Key implementation: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-9 (9 lines); hunks: -1,4 +1,3; -47,10 +46,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-9 (9 lines); hunks: -1,4 +1,3; -61,10 +60,6 @@ def setUpClass(cls):; symbols: setUpClass, touching `setUpClass`.
- Code diff details:
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-9 (9 lines); hunks: -1,4 +1,3; -47,10 +46,6 @@ def setUpClass(cls):; symbols: setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-9 (9 lines); hunks: -1,4 +1,3; -61,10 +60,6 @@ def setUpClass(cls):; symbols: setUpClass
- Key code excerpts:

```diff
diff -- test/registered/quant/test_deepseek_v32_fp4_4gpu.py
@@ -1,4 +1,3 @@
-import os
@@ -47,10 +46,6 @@ def setUpClass(cls):
-            env={
-                **os.environ,
-                "HF_HUB_OFFLINE": "0",
-            },
diff -- test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py
@@ -1,4 +1,3 @@
-import os
@@ -61,10 +60,6 @@ def setUpClass(cls):
-                env={
-                    **os.environ,
-                    "HF_HUB_OFFLINE": "0",
-                },
```

- Reviewed files:
  - tests: `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-9; `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-9
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21179 - [Bug] Preserve DeepSeek-V3.2 tool-call markers in reasoning parsing

- Link: https://github.com/sgl-project/sglang/pull/21179
- Status/date: open / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +179/-13, 295 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Bug] Preserve DeepSeek-V3.2 tool-call markers in reasoning parsing". The diff centers on `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`. PR body context: ## Summary - preserve DeepSeek-V3.2 DSML tool-call markers when `deepseek-v3` reasoning output switches into tool calls - add DeepSeek-V3-specific reasoning detector coverage fo...
- Key implementation: `test/registered/unit/parser/test_reasoning_parser.py` modified +106/-0 (106 lines); hunks: -1,8 +1,10; -17,6 +19,10; symbols: TestStreamingParseResult, test_init_default, test_detect_and_parse_without_thinking, TestDeepSeekV3Detector, touching `TestStreamingParseResult, test_init_default, test_detect_and_parse_without_thinking`; `python/sglang/srt/parser/reasoning_parser.py` modified +73/-13 (86 lines); hunks: -1,5 +1,6; -25,7 +26,7 @@ def __init__(; symbols: __init__, _tool_start_tokens, _find_tool_start, _find_partial_tool_start_suffix_len, touching `__init__, _tool_start_tokens, _find_tool_start`.
- Code diff details:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +106/-0 (106 lines); hunks: -1,8 +1,10; -17,6 +19,10; symbols: TestStreamingParseResult, test_init_default, test_detect_and_parse_without_thinking, TestDeepSeekV3Detector
  - `python/sglang/srt/parser/reasoning_parser.py` modified +73/-13 (86 lines); hunks: -1,5 +1,6; -25,7 +26,7 @@ def __init__(; symbols: __init__, _tool_start_tokens, _find_tool_start, _find_partial_tool_start_suffix_len
- Key code excerpts:

```diff
diff -- test/registered/unit/parser/test_reasoning_parser.py
@@ -1,8 +1,10 @@
+from sglang.srt.entrypoints.openai.encoding_dsv32 import dsml_token
+    DeepSeekV3Detector,
@@ -17,6 +19,10 @@
+DEEPSEEK_V32_TOOL_START = f"<{dsml_token}function_calls>"
+DEEPSEEK_V32_FORMATTED_TOOL_START = f"\n\n<{dsml_token}function_calls>"
@@ -221,6 +227,59 @@ def test_detect_and_parse_without_thinking(self):
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -1,5 +1,6 @@
-from typing import Dict, Optional, Tuple, Type
+from typing import Dict, Optional, Tuple, Type, Union
+from sglang.srt.entrypoints.openai.encoding_dsv32 import dsml_token
@@ -25,7 +26,7 @@ def __init__(
-        tool_start_token: Optional[str] = None,
+        tool_start_token: Optional[Union[str, Tuple[str, ...]]] = None,
```

- Reviewed files:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +106/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +73/-13
- Risk and verification: The diff ships test coverage in `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20343 - HiSparse for Sparse Attention

- Link: https://github.com/sgl-project/sglang/pull/20343
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +1692/-59, 2094 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "HiSparse for Sparse Attention". The diff centers on `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation This PR introduces HiSparse, which leverages CPU memory to store idle KV cache during decoding, thereby increasing batch size and improving throughput for models t...
- Key implementation: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +34/-3 (37 lines); hunks: -13,6 +13,10; -481,8 +485,8 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools, touching `_init_pools`; `python/sglang/srt/model_executor/model_runner.py` modified +28/-0 (28 lines); hunks: -345,6 +345,7 @@ def __init__(; -418,6 +419,9 @@ def __init__(; symbols: __init__, initialize, _forward_raw, touching `__init__, initialize, _forward_raw`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-2 (26 lines); hunks: -177,6 +177,7 @@ class NSAIndexerMetadata(BaseIndexerMetadata):; -246,7 +247,7 @@ def topk_transform(; symbols: NSAIndexerMetadata, get_seqlens_int32, topk_transform, forward_extend, touching `NSAIndexerMetadata, get_seqlens_int32, topk_transform`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0 (9 lines); hunks: -953,6 +953,12 @@ def capture_one_batch_size(; -1119,6 +1125,9 @@ def replay_prepare(; symbols: capture_one_batch_size, replay_prepare, replay, touching `capture_one_batch_size, replay_prepare, replay`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +34/-3 (37 lines); hunks: -13,6 +13,10; -481,8 +485,8 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools
  - `python/sglang/srt/model_executor/model_runner.py` modified +28/-0 (28 lines); hunks: -345,6 +345,7 @@ def __init__(; -418,6 +419,9 @@ def __init__(; symbols: __init__, initialize, _forward_raw
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-2 (26 lines); hunks: -177,6 +177,7 @@ class NSAIndexerMetadata(BaseIndexerMetadata):; -246,7 +247,7 @@ def topk_transform(; symbols: NSAIndexerMetadata, get_seqlens_int32, topk_transform, forward_extend
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0 (9 lines); hunks: -953,6 +953,12 @@ def capture_one_batch_size(; -1119,6 +1125,9 @@ def replay_prepare(; symbols: capture_one_batch_size, replay_prepare, replay
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0 (4 lines); hunks: -62,6 +62,7; -415,6 +416,9 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: ForwardBatch
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
@@ -13,6 +13,10 @@
+from sglang.srt.mem_cache.hisparse_memory_pool import (
+    HiSparseNSATokenToKVPool,
+    HiSparseTokenToKVPoolAllocator,
+)
@@ -481,8 +485,8 @@ def _init_pools(self: ModelRunner):
-            self.token_to_kv_pool = NSATokenToKVPool(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -345,6 +345,7 @@ def __init__(
+        self.enable_hisparse = server_args.enable_hisparse
@@ -418,6 +419,9 @@ def __init__(
+        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
+        self.hisparse_coordinator = None
@@ -611,6 +615,26 @@ def initialize(self, pre_model_load_memory: float):
+        # Init hisparse coordinator (must happen before CUDA graph capture)
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -177,6 +177,7 @@ class NSAIndexerMetadata(BaseIndexerMetadata):
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +34/-3; `python/sglang/srt/model_executor/model_runner.py` modified +28/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-2; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0; `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0; `python/sglang/srt/managers/hisparse_coordinator.py` added +596/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/jit_kernel/hisparse.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21194 - [bugfix][AMD] Fix PPMissingLayer AttributeError for deepseek v2/v3 in aiter_gfx95 code path

- Link: https://github.com/sgl-project/sglang/pull/21194
- Status/date: open / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-4, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[bugfix][AMD] Fix PPMissingLayer AttributeError for deepseek v2/v3 in aiter_gfx95 code path". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation When running DeepSeek-V2/V3 with Pipeline Parallelism (PP) on AMD gfx950 GPUs (MI300/MI350), the server crashes during model initialization with: `AttributeError:...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -1867,19 +1867,19 @@ def __init__(; -1904,7 +1904,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -1867,19 +1867,19 @@ def __init__(; -1904,7 +1904,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1867,19 +1867,19 @@ def __init__(
-            and self.embed_tokens.embedding_dim == 7168
+            and getattr(self.embed_tokens, "embedding_dim", None) == 7168
-                    if isinstance(self.layers[i].mlp, DeepseekV2MoE)
+                    if isinstance(getattr(self.layers[i], "mlp", None), DeepseekV2MoE)
-                if isinstance(self.layers[i].mlp, DeepseekV2MoE):
+                if isinstance(getattr(self.layers[i], "mlp", None), DeepseekV2MoE):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +4/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15807 - [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side.

- Link: https://github.com/sgl-project/sglang/pull/15807
- Status/date: closed / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +516/-39, 827 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side.". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation In the previous implementation, NSA's indexer_k pool and KV cache pool shared a single set of index mappings (req_to_token_pool, token_to_kv_pool_allocator). In up...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +89/-0 (89 lines); hunks: -30,6 +30,7; -110,6 +111,9 @@ class NSAMetadata:; symbols: NSAMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded, touching `NSAMetadata, get_seqlens_int32, get_page_table_64`; `python/sglang/srt/model_executor/model_runner.py` modified +48/-17 (65 lines); hunks: -1913,6 +1913,7 @@ def init_memory_pool(; -1942,14 +1943,28 @@ def init_memory_pool(; symbols: init_memory_pool, touching `init_memory_pool`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +31/-12 (43 lines); hunks: -7,6 +7,7; -549,11 +550,10 @@ def _forward_cuda_k_only(; symbols: _forward_cuda_k_only, forward_indexer, forward_cuda, forward_npu, touching `_forward_cuda_k_only, forward_indexer, forward_cuda`; `python/sglang/srt/mem_cache/allocator.py` modified +156/-0 (156 lines); hunks: -584,3 +584,159 @@ def get_cpu_copy(self, indices):; symbols: get_cpu_copy, load_cpu_copy, NSAHybridTokenToKVPoolAllocator, __init__, touching `get_cpu_copy, load_cpu_copy, NSAHybridTokenToKVPoolAllocator`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +89/-0 (89 lines); hunks: -30,6 +30,7; -110,6 +111,9 @@ class NSAMetadata:; symbols: NSAMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/model_executor/model_runner.py` modified +48/-17 (65 lines); hunks: -1913,6 +1913,7 @@ def init_memory_pool(; -1942,14 +1943,28 @@ def init_memory_pool(; symbols: init_memory_pool
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +31/-12 (43 lines); hunks: -7,6 +7,7; -549,11 +550,10 @@ def _forward_cuda_k_only(; symbols: _forward_cuda_k_only, forward_indexer, forward_cuda, forward_npu
  - `python/sglang/srt/mem_cache/allocator.py` modified +156/-0 (156 lines); hunks: -584,3 +584,159 @@ def get_cpu_copy(self, indices):; symbols: get_cpu_copy, load_cpu_copy, NSAHybridTokenToKVPoolAllocator, __init__
  - `python/sglang/srt/mem_cache/common.py` modified +68/-5 (73 lines); hunks: -7,7 +7,10; -470,9 +473,41 @@ def alloc_for_decode(batch: ScheduleBatch, token_per_req: i...; symbols: alloc_for_decode, _alloc_for_nsa_index_k, release_kv_cache, available_and_evictable_str
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -30,6 +30,7 @@
+from sglang.srt.mem_cache.common import enable_nsa_hybrid_indexer_pool
@@ -110,6 +111,9 @@ class NSAMetadata:
+    # Separated page table for index_k
+    indexer_real_page_table: Optional[torch.Tensor] = None
@@ -164,6 +168,8 @@ def get_seqlens_int32(self) -> torch.Tensor:
+        if self.attn_metadata.indexer_real_page_table is not None:
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -1913,6 +1913,7 @@ def init_memory_pool(
+        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
@@ -1942,14 +1943,28 @@ def init_memory_pool(
-                    self.req_to_token_pool = DecodeReqToTokenPool(
-                        size=max_num_reqs,
-                        max_context_len=self.model_config.context_len
-                        + extra_max_context_len,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -7,6 +7,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +89/-0; `python/sglang/srt/model_executor/model_runner.py` modified +48/-17; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +31/-12; `python/sglang/srt/mem_cache/allocator.py` modified +156/-0; `python/sglang/srt/mem_cache/common.py` modified +68/-5; `python/sglang/srt/disaggregation/decode.py` modified +56/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/decode.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #14619 - [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32.

- Link: https://github.com/sgl-project/sglang/pull/14619
- Status/date: closed / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 31 files, +3077/-118, 3804 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32.". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`. PR body context: ## Motivation Previously, Sparse Attention was introduced in DeepSeekV32 Mode, which selects only the top-2048 tokens to participate in attention computation, significantly impr...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +98/-8 (106 lines); hunks: -92,6 +92,7; -486,6 +487,7 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize, init_memory_pool, touching `initialize, init_memory_pool`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +55/-5 (60 lines); hunks: -7,6 +7,7; -22,6 +23,7; symbols: NSAMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded, touching `NSAMetadata, get_seqlens_int32, get_page_table_64`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +33/-0 (33 lines); hunks: -11,6 +11,7; -362,6 +363,12 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_extend, forward_decode, touching `__init__, init_forward_metadata, forward_extend`; `python/sglang/srt/models/deepseek_v2.py` modified +26/-7 (33 lines); hunks: -109,6 +109,7; -1768,13 +1769,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, touching `forward_absorb_prepare`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +98/-8 (106 lines); hunks: -92,6 +92,7; -486,6 +487,7 @@ def initialize(self, min_per_gpu_memory: float):; symbols: initialize, init_memory_pool
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +55/-5 (60 lines); hunks: -7,6 +7,7; -22,6 +23,7; symbols: NSAMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +33/-0 (33 lines); hunks: -11,6 +11,7; -362,6 +363,12 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_extend, forward_decode
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-7 (33 lines); hunks: -109,6 +109,7; -1768,13 +1769,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +21/-6 (27 lines); hunks: -440,11 +440,10 @@ def _forward_cuda_k_only(; -621,11 +620,10 @@ def forward_cuda(; symbols: _forward_cuda_k_only, forward_cuda, _get_index_cache_loc, forward_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -92,6 +92,7 @@
+    NSAHybridTokenToKVPoolAllocator,
@@ -486,6 +487,7 @@ def initialize(self, min_per_gpu_memory: float):
+            self.init_sparse_coordinator()
@@ -1651,6 +1653,7 @@ def init_memory_pool(
+                    NSADecodeReqToTokenPool,
@@ -1667,6 +1670,18 @@ def init_memory_pool(
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -7,6 +7,7 @@
+from sglang.srt.disaggregation.decode import NSADecodeReqToTokenPool
@@ -22,6 +23,7 @@
+from sglang.srt.mem_cache.allocator import is_enable_hierarchical_nsa
@@ -102,6 +104,9 @@ class NSAMetadata:
+    # Separate page table for indexer_k (when enable hierarchical NSA)
+    index_real_page_table: Optional[torch.Tensor] = None
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -11,6 +11,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +98/-8; `python/sglang/srt/layers/attention/nsa_backend.py` modified +55/-5; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +33/-0; `python/sglang/srt/models/deepseek_v2.py` modified +26/-7; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +21/-6; `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py` added +622/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/disaggregation/decode.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21192 - Fix CP in-seq-split method for DeepSeek V32 and update related tests

- Link: https://github.com/sgl-project/sglang/pull/21192
- Status/date: merged / 2026-03-23
- Trace source: `git log --name-only -- <model-files>` found it through `test/manual/nightly/test_deepseek_v32_perf.py`, `test/registered/cp/test_deepseek_v32_cp_single_node.py`; associated commits `ed316a26efa4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +162/-97, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Fix CP in-seq-split method for DeepSeek V32 and update related tests". The diff centers on `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `test/manual/nightly/test_deepseek_v32_perf.py`. PR body context: ## Motivation - Fix in-seq-split for CP of V32 - Remove all the V32-Exp models in CI testcases - Move V32 CP tests to cp folder, and mark it as pr-test ## Modifications ## Accur...
- Key implementation: `test/registered/cp/test_deepseek_v32_cp_single_node.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: TestDeepseekV32CPInSeqSplit, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32CPInSeqSplit, setUpClass, tearDownClass`; `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` removed +0/-92 (92 lines); hunks: -1,92 +0,0; symbols: TestDeepseekV32CPSingleNode, for, test_deepseek_v32_cp_variants, touching `TestDeepseekV32CPSingleNode, for, test_deepseek_v32_cp_variants`; `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-1 (2 lines); hunks: -3,7 +3,7.
- Code diff details:
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` added +157/-0 (157 lines); hunks: -0,0 +1,157; symbols: TestDeepseekV32CPInSeqSplit, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` removed +0/-92 (92 lines); hunks: -1,92 +0,0; symbols: TestDeepseekV32CPSingleNode, for, test_deepseek_v32_cp_variants
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-1 (2 lines); hunks: -3,7 +3,7
- Key code excerpts:

```diff
diff -- test/registered/cp/test_deepseek_v32_cp_single_node.py
@@ -0,0 +1,157 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.environ import envs
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
diff -- test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py
@@ -1,92 +0,0 @@
-import unittest
-from sglang.test.accuracy_test_runner import AccuracyTestParams
-from sglang.test.ci.ci_register import register_cuda_ci
-from sglang.test.run_combined_tests import run_combined_tests
-from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system
-register_cuda_ci(est_time=5400, suite="nightly-8-gpu-common", nightly=True)
diff -- test/manual/nightly/test_deepseek_v32_perf.py
@@ -3,7 +3,7 @@
```

- Reviewed files:
  - tests: `test/registered/cp/test_deepseek_v32_cp_single_node.py` added +157/-0; `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` removed +0/-92; `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/manual/nightly/test_deepseek_v32_perf.py`, `test/registered/8-gpu-models/test_deepseek_v32_basic.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20438 - [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2

- Link: https://github.com/sgl-project/sglang/pull/20438
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `649172879778`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +19/-0, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation In the current Native Sparse Attention (NSA) implementation, specifically when Context Parallelism (CP) is enabled via `--enable-nsa-prefill-context-parallel`, the...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Tilelang sparse fwd for dsv32 mi355/mi300". The diff centers on `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`. PR body context: ## Motivation - Enable the faster/new tilelang kernel on MI300. - Improve longer-context kernel performance on MI355. ## Modifications - Enable the new tilelang partial+combine...
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

### PR #21337 - Workaround of DSA performance drop on B200 + DP

- Link: https://github.com/sgl-project/sglang/pull/21337
- Status/date: merged / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-5, 37 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "Workaround of DSA performance drop on B200 + DP". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation #21291 Route the config of glmfp8+B200+DP to bf16 kvcache+flashmla sparse prefill+trtllm decode Note that this is just a workaround, not root fix. #21011 can be th...
- Key implementation: `python/sglang/srt/server_args.py` modified +11/-5 (16 lines); hunks: -1386,7 +1386,7 @@ def _generate_piecewise_cuda_graph_tokens(self):; -1400,9 +1400,15 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) ->...; symbols: _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _handle_model_specific_adjustments, touching `_generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +11/-5 (16 lines); hunks: -1386,7 +1386,7 @@ def _generate_piecewise_cuda_graph_tokens(self):; -1400,9 +1400,15 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) ->...; symbols: _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -1386,7 +1386,7 @@ def _generate_piecewise_cuda_graph_tokens(self):
-    def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:
+    def _set_default_nsa_kv_cache_dtype(self, major: int, quantization: str) -> str:
@@ -1400,9 +1400,15 @@ def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:
-            self.kv_cache_dtype = (
-                "fp8_e4m3" if (major >= 10 and self.dp_size > 1) else "bfloat16"
-            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +11/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16079 - [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2

- Link: https://github.com/sgl-project/sglang/pull/16079
- Status/date: closed / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 27 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[Performance] Change sparse MLA and dense MHA switching threshold DSv3.2". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Command `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 SGLANG_DISAGGREGATION_QUEUE_SIZE=128 SGLANG_PP_LAYER_PARTITION="8,8,8,8,7,7,7,...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-2 (6 lines); hunks: -26,7 +26,7; -257,6 +257,8 @@ def __init__(; symbols: __init__, set_nsa_prefill_impl, touching `__init__, set_nsa_prefill_impl`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-2 (6 lines); hunks: -26,7 +26,7; -257,6 +257,8 @@ def __init__(; symbols: __init__, set_nsa_prefill_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -26,7 +26,7 @@
+from sglang.srt.utils.common import get_int_env_var
@@ -257,6 +257,8 @@ def __init__(
+        mha_nsa_index_topk_value = self.nsa_index_topk*6 if self.nsa_kv_cache_store_fp8 else self.nsa_index_topk*3
+        self.mha_nsa_index_topk = get_int_env_var("SGLANG_MHA_NSA_INDEX_TOPK", mha_nsa_index_topk_value)
@@ -1445,7 +1447,7 @@ def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
-                and max_kv_len <= self.nsa_index_topk  # Short enough for MHA
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20606 - FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache

- Link: https://github.com/sgl-project/sglang/pull/20606
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa_backend.py`; associated commits `4b5f63e1b8ba`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +20/-4, 66 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`. PR body context: ## Motivation When using the flashmla_sparse NSA prefill backend with FP8 KV cache, topk_indices_offset is never computed outside the normal EXTEND forward mode, causing a crash...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +20/-4 (24 lines); hunks: -260,6 +260,11 @@ def topk_transform(; -402,7 +407,9 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: topk_transform, init_forward_metadata, forward_extend, set_nsa_prefill_impl, touching `topk_transform, init_forward_metadata, forward_extend`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +20/-4 (24 lines); hunks: -260,6 +260,11 @@ def topk_transform(; -402,7 +407,9 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: topk_transform, init_forward_metadata, forward_extend, set_nsa_prefill_impl
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -260,6 +260,11 @@ def topk_transform(
+            if cu_topk_indices_offset is None:
+                raise RuntimeError(
+                    "RAGGED topk_transform requires topk_indices_offset; "
+                    "expected extend-without-speculative metadata."
+                )
@@ -402,7 +407,9 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +20/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21506 - [WIP][NPU] DeepSeek-V3.2 adapt enable-torch-compile

- Link: https://github.com/sgl-project/sglang/pull/21506
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +70/-17, 164 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[WIP][NPU] DeepSeek-V3.2 adapt enable-torch-compile". The diff centers on `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` modified +37/-12 (49 lines); hunks: -16,6 +16,7; -66,21 +67,45 @@ def __init__(; symbols: __init__, dispatch, combine, touching `__init__, dispatch, combine`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` modified +18/-0 (18 lines); hunks: -132,3 +132,21 @@ def apply_weights(; symbols: apply_weights, apply_without_routing_weights, touching `apply_weights, apply_without_routing_weights`; `python/sglang/srt/models/deepseek_v2.py` modified +8/-5 (13 lines); hunks: -1348,9 +1348,6 @@ def forward_prepare(; -1992,7 +1989,8 @@ def forward(; symbols: forward_prepare, forward, touching `forward_prepare, forward`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +1/-0 (1 lines); hunks: -202,6 +202,7 @@ def _add_fused_moe_to_target_scheme_map(self):; symbols: _add_fused_moe_to_target_scheme_map, weight_block_size, touching `_add_fused_moe_to_target_scheme_map, weight_block_size`.
- Code diff details:
  - `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` modified +37/-12 (49 lines); hunks: -16,6 +16,7; -66,21 +67,45 @@ def __init__(; symbols: __init__, dispatch, combine
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` modified +18/-0 (18 lines); hunks: -132,3 +132,21 @@ def apply_weights(; symbols: apply_weights, apply_without_routing_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-5 (13 lines); hunks: -1348,9 +1348,6 @@ def forward_prepare(; -1992,7 +1989,8 @@ def forward(; symbols: forward_prepare, forward
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +1/-0 (1 lines); hunks: -202,6 +202,7 @@ def _add_fused_moe_to_target_scheme_map(self):; symbols: _add_fused_moe_to_target_scheme_map, weight_block_size
  - `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` modified +3/-0 (3 lines); hunks: -219,6 +219,7 @@ def __init__(; -230,6 +231,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/moe/token_dispatcher/fuseep.py
@@ -16,6 +16,7 @@
+from sglang.srt.server_args import get_global_server_args
@@ -66,21 +67,45 @@ def __init__(
+        self.rank = self.group.rank()
+        self.num_ranks = self.group.size()
+        backend = self.group._get_backend(torch.device("npu"))
+        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(self.rank)
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py
@@ -132,3 +132,21 @@ def apply_weights(
+    def apply_without_routing_weights(
+        self,
+        layer,
+        hidden_states,
+        hidden_states_scale,
+        group_list_type,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -1348,9 +1348,6 @@ def forward_prepare(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` modified +37/-12; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py` modified +18/-0; `python/sglang/srt/models/deepseek_v2.py` modified +8/-5; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +1/-0; `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` modified +3/-0; `python/sglang/srt/utils/common.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21529 - Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm

- Link: https://github.com/sgl-project/sglang/pull/21529
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +308/-126, 644 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm". The diff centers on `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: Enable MXFP4 (including Quark W4A4) quantized models (e.g. GLM-5-MXFP4-Q8, Kimi-K2.5-MXFP4) to load and run on AMD MI355X with ROCm. These models use Quark's W4A4 MXFP4 scheme w...
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
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[ROCm] Fix fused MLA decode rope path for Kimi K2.5 and DeepSeek-variant models". The diff centers on `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`, `python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py`. PR body context: The SGLANG_ROCM_FUSED_DECODE_MLA=1 path had several issues preventing it from working with models like Kimi K2.5 that use DeepseekScalingRotaryEmbedding or different ForwardMeta...
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

### PR #21546 - [Fix] Catch MalformedJSON exception for DeepSeek-V3.2 function call partial parsing

- Link: https://github.com/sgl-project/sglang/pull/21546
- Status/date: open / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-1, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Catch MalformedJSON exception for DeepSeek-V3.2 function call partial parsing". The diff centers on `python/sglang/srt/function_call/deepseekv32_detector.py`. PR body context: ## Motivation During partial JSON parsing, a MalformedJSON exception may be thrown if the format is invalid. DeepSeek-V3.2 does not catch this exception, causing the parsing pro...
- Key implementation: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-1 (3 lines); hunks: -2,6 +2,7; -160,7 +161,7 @@ def _parse_parameters_from_xml(; symbols: _parse_parameters_from_xml, touching `_parse_parameters_from_xml`.
- Code diff details:
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-1 (3 lines); hunks: -2,6 +2,7; -160,7 +161,7 @@ def _parse_parameters_from_xml(; symbols: _parse_parameters_from_xml
- Key code excerpts:

```diff
diff -- python/sglang/srt/function_call/deepseekv32_detector.py
@@ -2,6 +2,7 @@
+from partial_json_parser.core.exceptions import MalformedJSON
@@ -160,7 +161,7 @@ def _parse_parameters_from_xml(
-                    except json.JSONDecodeError:
+                    except (json.JSONDecodeError, MalformedJSON):
```

- Reviewed files:
  - runtime: `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/function_call/deepseekv32_detector.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21585 - [CI] Move v32 cp test to deepep running suite

- Link: https://github.com/sgl-project/sglang/pull/21585
- Status/date: merged / 2026-03-28
- Trace source: `git log --name-only -- <model-files>` found it through `test/registered/cp/test_deepseek_v32_cp_single_node.py`; associated commits `6ef4318ec07b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[CI] Move v32 cp test to deepep running suite". The diff centers on `test/registered/cp/test_deepseek_v32_cp_single_node.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +1/-1 (2 lines); hunks: -14,7 +14,7.
- Code diff details:
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +1/-1 (2 lines); hunks: -14,7 +14,7
- Key code excerpts:

```diff
diff -- test/registered/cp/test_deepseek_v32_cp_single_node.py
@@ -14,7 +14,7 @@
-register_cuda_ci(est_time=360, suite="stage-c-test-8-gpu-h200")
+register_cuda_ci(est_time=360, suite="stage-c-test-deepep-8-gpu-h200")
```

- Reviewed files:
  - tests: `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v32_cp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21623 - [Test] Add unit tests for encoding_dsv32.py

- Link: https://github.com/sgl-project/sglang/pull/21623
- Status/date: open / 2026-03-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +871/-0, 872 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Test] Add unit tests for encoding_dsv32.py". The diff centers on `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py`. PR body context: ## Summary Part of #20865 (Improve Unit Test Coverage) Adds 72 CPU-only unit tests for `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` — the DSML encoding/decoding modu...
- Key implementation: `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` added +871/-0 (871 lines); hunks: -0,0 +1,871; symbols: _make_tool, _make_tool_call, _parse_dsml_args, TestEncodeArgumentsToDsml, touching `_make_tool, _make_tool_call, _parse_dsml_args`.
- Code diff details:
  - `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` added +871/-0 (871 lines); hunks: -0,0 +1,871; symbols: _make_tool, _make_tool_call, _parse_dsml_args, TestEncodeArgumentsToDsml
- Key code excerpts:

```diff
diff -- test/registered/unit/entrypoints/openai/test_encoding_dsv32.py
@@ -0,0 +1,871 @@
+"""Unit tests for encoding_dsv32.py — no server, no model loading.
+Tests cover encode_arguments_to_dsml, decode_dsml_to_arguments, render_tools,
+find_last_user_index, render_message, drop_thinking_messages, encode_messages,
+and _read_until_stop.
+"""
+import json
```

- Reviewed files:
  - tests: `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` added +871/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21468 - [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation

- Link: https://github.com/sgl-project/sglang/pull/21468
- Status/date: merged / 2026-03-30
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +96/-148, 305 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[NPU] Update DeepSeek-V3.2 model deployment instructions in documentation". The diff centers on `docs/platforms/ascend/ascend_npu_best_practice.md`. PR body context: ## Motivation Updates the documentation for deploying the DeepSeek-V3.2 model on Ascend NPU ## Modifications Based on the main branch of the sglang-project, we enabled the atten...
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +96/-148 (244 lines); hunks: -13,7 +13,7 @@ you encounter issues or have any questions, please [open an is...; -779,24 +779,22 @@ We tested it based on the `RANDOM` dataset..
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +96/-148 (244 lines); hunks: -13,7 +13,7 @@ you encounter issues or have any questions, please [open an is...; -779,24 +779,22 @@ We tested it based on the `RANDOM` dataset.
- Key code excerpts:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -13,7 +13,7 @@ you encounter issues or have any questions, please [open an issue](https://githu
-| DeepSeek-V3.2-Exp | Atlas 800I A3 | 32    | PD Separation | 64K+3K    | 30ms | W8A8 INT8    | [Optimal Configuration](#deepseek-v32-exp-64k-3k-30ms-on-a3-32-cards-separation-mod
+| DeepSeek-V3.2     | Atlas 800I A3 | 32    | PD Separation | 128K+1K   | 20ms | W8A8 INT8    | [Optimal Configuration](#deepseek-v32-128k-1k-20ms-on-a3-32-cards-separation-mode)
@@ -779,24 +779,22 @@ We tested it based on the `RANDOM` dataset.
-### DeepSeek-V3.2-Exp 64K-3K 30ms on A3 32 Cards Separation Mode
+### DeepSeek-V3.2 128K-1K 20ms on A3 32 Cards Separation Mode
-Model: DeepSeek-V3.2-Exp-W8A8
```

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +96/-148
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #21783 - [DSA] Support trtllm sparse mla kernel for prefill batches

- Link: https://github.com/sgl-project/sglang/pull/21783
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +12/-14, 77 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSA] Support trtllm sparse mla kernel for prefill batches". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/test/run_eval.py`. PR body context: ## Motivation Depends on flashinfer v0.6.7 #21422 ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format cod...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-0 (9 lines); hunks: -1297,6 +1297,7 @@ def forward_extend(; -1929,6 +1930,7 @@ def _forward_trtllm(; symbols: forward_extend, _forward_trtllm, touching `forward_extend, _forward_trtllm`; `python/sglang/srt/server_args.py` modified +0/-11 (11 lines); hunks: -1455,9 +1455,6 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; -1526,14 +1523,6 @@ def _handle_model_specific_adjustments(self):; symbols: _set_default_nsa_backends, _handle_model_specific_adjustments, touching `_set_default_nsa_backends, _handle_model_specific_adjustments`; `python/sglang/test/run_eval.py` modified +3/-3 (6 lines); hunks: -20,10 +20,10; -267,7 +267,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval, touching `get_thinking_kwargs, run_eval`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-0 (9 lines); hunks: -1297,6 +1297,7 @@ def forward_extend(; -1929,6 +1930,7 @@ def _forward_trtllm(; symbols: forward_extend, _forward_trtllm
  - `python/sglang/srt/server_args.py` modified +0/-11 (11 lines); hunks: -1455,9 +1455,6 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; -1526,14 +1523,6 @@ def _handle_model_specific_adjustments(self):; symbols: _set_default_nsa_backends, _handle_model_specific_adjustments
  - `python/sglang/test/run_eval.py` modified +3/-3 (6 lines); hunks: -20,10 +20,10; -267,7 +267,7 @@ def run_eval(args):; symbols: get_thinking_kwargs, run_eval
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1297,6 +1297,7 @@ def forward_extend(
+                is_prefill=True,
@@ -1929,6 +1930,7 @@ def _forward_trtllm(
+        is_prefill: bool = False,
@@ -1990,6 +1992,13 @@ def _forward_trtllm(
+        elif is_prefill:
+            page_table_1 = transform_index_page_table_prefill(
diff -- python/sglang/srt/server_args.py
@@ -1455,9 +1455,6 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:
-                logger.warning(
-                    "Flashmla is not supported on Blackwell device without DP attention. Set NSA prefill/decode backends to trtllm, which runs fast but loses a little accuracy."
-                )
@@ -1526,14 +1523,6 @@ def _handle_model_specific_adjustments(self):
-                    if self.nsa_prefill_backend == "trtllm":
-                        # We temporarily set the threshold to 128k to avoid IMA error. Should be removed after supporting flashmla prefill impl with trtllm decode impl.
diff -- python/sglang/test/run_eval.py
@@ -20,10 +20,10 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-0; `python/sglang/srt/server_args.py` modified +0/-11
  - tests: `python/sglang/test/run_eval.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `python/sglang/test/run_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21889 - [AMD] Enable FP4 (E2M1) KV cache quantization for NSA with TileLang backend

- Link: https://github.com/sgl-project/sglang/pull/21889
- Status/date: open / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +618/-7, 743 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable FP4 (E2M1) KV cache quantization for NSA with TileLang backend". The diff centers on `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`. PR body context: ## Motivation Enable FP4 KV cache for NSA on Tilelang backend, based on FP8 attention https://github.com/sgl-project/sglang/pull/21511. This PR should be rebased onto main after...
- Key implementation: `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: _e2m1_dequant, _dequant_fp4_to_fp8_paged_kernel, dequantize_fp4_to_fp8_paged, get_fp8_dtype_for_dequant, touching `_e2m1_dequant, _dequant_fp4_to_fp8_paged_kernel, dequantize_fp4_to_fp8_paged`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +60/-0 (60 lines); hunks: -9,6 +9,12; -308,6 +314,9 @@ def __init__(; symbols: __init__, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph, forward_extend, touching `__init__, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph`; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +9/-0 (9 lines); hunks: -154,10 +154,15 @@ def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:; -379,6 +384,10 @@ def _init_pools(self: ModelRunner):; symbols: calculate_mla_kv_cache_dim, _init_pools, touching `calculate_mla_kv_cache_dim, _init_pools`; `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunks: -2006,7 +2006,11 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype, touching `configure_kv_cache_dtype`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: _e2m1_dequant, _dequant_fp4_to_fp8_paged_kernel, dequantize_fp4_to_fp8_paged, get_fp8_dtype_for_dequant
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +60/-0 (60 lines); hunks: -9,6 +9,12; -308,6 +314,9 @@ def __init__(; symbols: __init__, init_cuda_graph_state, init_forward_metadata_capture_cuda_graph, forward_extend
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +9/-0 (9 lines); hunks: -154,10 +154,15 @@ def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:; -379,6 +384,10 @@ def _init_pools(self: ModelRunner):; symbols: calculate_mla_kv_cache_dim, _init_pools
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunks: -2006,7 +2006,11 @@ def configure_kv_cache_dtype(self):; symbols: configure_kv_cache_dtype
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +2/-1 (3 lines); hunks: -405,7 +405,7 @@ def forward_absorb_core(; -606,4 +606,5 @@ def _skip_rope_for_nsa_tilelang_fused(self: DeepseekV2Attent...; symbols: forward_absorb_core, _skip_rope_for_nsa_tilelang_fused
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py
@@ -0,0 +1,264 @@
+"""FP4 (E2M1) paged KV cache → FP8 dequantization for NSA.
+Vectorized 1D kernel: one program per token processes all 256 packed
+nope bytes + 64 rope elements, storing FP8 directly (no BF16 intermediate).
+Optimal config determined by bench_fp4_kernels.py on MI355:
+  warps=1, stages=0  →  2719 GB/s @ 131k tokens (4.4× over 2D grid)
+"""
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -9,6 +9,12 @@
+from sglang.srt.layers.attention.nsa.dequant_fp4_to_fp8 import (
+    FP8_TOTAL_DIM,
+    dequant_fp4_paged_decode,
+    dequant_fp4_paged_extend,
+    get_fp8_dtype_for_dequant,
+)
diff -- python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
@@ -154,10 +154,15 @@ def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py` added +264/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +60/-0; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +9/-0; `python/sglang/srt/model_executor/model_runner.py` modified +5/-1; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +2/-1; `python/sglang/srt/mem_cache/utils.py` modified +157/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/dequant_fp4_to_fp8.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/memory_pool.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21914 - [DSA] Set trtllm kernels as default for Blackwell

- Link: https://github.com/sgl-project/sglang/pull/21914
- Status/date: merged / 2026-04-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-7, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSA] Set trtllm kernels as default for Blackwell". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `python/sglang/srt/server_args.py` modified +2/-7 (9 lines); hunks: -1411,15 +1411,10 @@ def _set_default_nsa_kv_cache_dtype(self, major: int, qu...; -1450,7 +1445,7 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends, touching `_set_default_nsa_kv_cache_dtype, _set_default_nsa_backends`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +2/-7 (9 lines); hunks: -1411,15 +1411,10 @@ def _set_default_nsa_kv_cache_dtype(self, major: int, qu...; -1450,7 +1445,7 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; symbols: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -1411,15 +1411,10 @@ def _set_default_nsa_kv_cache_dtype(self, major: int, quantization: str) -> str:
-            # TODO: Temporarily set default dtype on B200 as bfloat16 to avoid performance regression.
-            # TODO: Remove this after the performance regression is fixed. (Ref: https://github.com/sgl-project/sglang/issues/21291)
-            if quantization == "modelopt_fp4" and major >= 10 and self.dp_size > 1:
+            if major >= 10:
-            # self.kv_cache_dtype = (
-            #     "fp8_e4m3" if (major >= 10 and self.dp_size > 1) else "bfloat16"
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +2/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21511 - [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend

- Link: https://github.com/sgl-project/sglang/pull/21511
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; associated commits `7431db7392fe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +517/-77, 713 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend". The diff centers on `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`. PR body context: ## Motivation Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend. ## Modifications - Upgraded TileLang to a55a823 to enable FP8 gemm suppo...
- Key implementation: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +307/-42 (349 lines); hunks: -1,3 +1,4; -44,6 +45,23 @@ def fast_round_scale(amax, fp8_max_inv):; symbols: fast_round_scale, _pick_inner_iter, act_quant_kernel, main, touching `fast_round_scale, _pick_inner_iter, act_quant_kernel`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +79/-18 (97 lines); hunks: -292,9 +292,11 @@ def forward_absorb_prepare(; -332,24 +334,69 @@ def forward_absorb_core(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, _skip_rope_for_nsa_tilelang_fused, touching `forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +307/-42 (349 lines); hunks: -1,3 +1,4; -44,6 +45,23 @@ def fast_round_scale(amax, fp8_max_inv):; symbols: fast_round_scale, _pick_inner_iter, act_quant_kernel, main
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +79/-18 (97 lines); hunks: -292,9 +292,11 @@ def forward_absorb_prepare(; -332,24 +334,69 @@ def forward_absorb_core(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, _skip_rope_for_nsa_tilelang_fused
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/tilelang_kernel.py
@@ -1,3 +1,4 @@
+from functools import lru_cache
@@ -44,6 +45,23 @@ def fast_round_scale(amax, fp8_max_inv):
+@lru_cache(maxsize=8)
+def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
+    """
+    Pick the largest valid inner_iter (power-of-two divisor of ni) that keeps
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -292,9 +292,11 @@ def forward_absorb_prepare(
+        skip_rope_for_nsa_tilelang_fused = self._skip_rope_for_nsa_tilelang_fused()
+            and (not skip_rope_for_nsa_tilelang_fused)
@@ -332,24 +334,69 @@ def forward_absorb_core(
-            extra_args = {}
-            if self._fuse_rope_for_trtllm_mla(forward_batch):
-                extra_args = {
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +307/-42; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +79/-18
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/mem_cache/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22065 - [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models.

- Link: https://github.com/sgl-project/sglang/pull/22065
- Status/date: merged / 2026-04-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models.". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -6170,6 +6170,14 @@ def check_server_args(self):; symbols: check_server_args, touching `check_server_args`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: -6170,6 +6170,14 @@ def check_server_args(self):; symbols: check_server_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -6170,6 +6170,14 @@ def check_server_args(self):
+            from sglang.srt.configs.model_config import is_deepseek_nsa
+            hf_config = self.get_model_config().hf_config
+            assert is_deepseek_nsa(hf_config), (
+                "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) models now"
+                "(e.g., DeepSeek V3.2, GLM-5). "
+            )
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21405 - Enable IndexCache for DeepSeek V3.2

- Link: https://github.com/sgl-project/sglang/pull/21405
- Status/date: merged / 2026-04-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; associated commits `5a3531641735`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +196/-20, 358 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Enable IndexCache for DeepSeek V3.2". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`. PR body context: ## Motivation fix https://github.com/sgl-project/sglang/issues/21286 ## Modifications * Port https://github.com/THUDM/IndexCache * add ut for deepseek-v3.2 ## Accuracy Tests gsm...
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -1106,6 +1106,7 @@ def __init__(; -1175,6 +1176,8 @@ def __init__(; symbols: __init__, op_prepare, op_core, touching `__init__, op_prepare, op_core`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, touching `forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla`; `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k, touching `TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunks: -1106,6 +1106,7 @@ def __init__(; -1175,6 +1176,8 @@ def __init__(; symbols: __init__, op_prepare, op_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k
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
diff -- test/registered/8-gpu-models/test_deepseek_v32_indexcache.py
@@ -0,0 +1,117 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +51/-6; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13
  - tests: `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- Link: https://github.com/sgl-project/sglang/pull/22179
- Status/date: merged / 2026-04-06
- Trace source: `git log --name-only -- <model-files>` found it through `docs/basic_usage/deepseek_v32.md`; associated commits `b311db2e4994`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-12, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation". The diff centers on `docs/basic_usage/deepseek_v32.md`. PR body context: ## Summary Remove skip-softmax section (I think it's for dense attention only, not DSA, per flashinfer constraint below) and improve docs https://github.com/flashinfer-ai/flashi...
- Key implementation: `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-....
- Code diff details:
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunks: -3,7 +3,7; -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-...
- Key code excerpts:

```diff
diff -- docs/basic_usage/deepseek_v32.md
@@ -3,7 +3,7 @@
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
+Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
@@ -56,13 +56,13 @@ python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
-To server GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
+To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
-- **MHA prefill threshold relax** To apply MHA attention to requests longer than 2048 tokens, please set flag `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` to a value larger th
```

- Reviewed files:
  - docs: `docs/basic_usage/deepseek_v32.md` modified +11/-12
- Risk and verification: This is mostly docs/examples in `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22238 - [HiSparse]: Add readme docs for HiSparse Feature

- Link: https://github.com/sgl-project/sglang/pull/22238
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +117/-0, 122 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HiSparse]: Add readme docs for HiSparse Feature". The diff centers on `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `docs/advanced_features/hisparse_guide.md` added +111/-0 (111 lines); hunks: -0,0 +1,111; `docs/basic_usage/deepseek_v32.md` modified +6/-0 (6 lines); hunks: -468,3 +468,9 @@ python -m sglang.launch_server \.
- Code diff details:
  - `docs/advanced_features/hisparse_guide.md` added +111/-0 (111 lines); hunks: -0,0 +1,111
  - `docs/basic_usage/deepseek_v32.md` modified +6/-0 (6 lines); hunks: -468,3 +468,9 @@ python -m sglang.launch_server \
- Key code excerpts:

```diff
diff -- docs/advanced_features/hisparse_guide.md
@@ -0,0 +1,111 @@
+# HiSparse: Hierarchical Sparse Attention
+HiSparse reduces per-request GPU memory consumption during the decode phase by maintaining only a small "hot" KV buffer on GPU while keeping complete KV data in CPU pinned memory.
+> **Prerequisites**: HiSparse only works with models that use **DeepSeek Sparse Attention (DSA)**  architectures (e.g., DeepSeek-V3.2, GLM-5). These models natively select a subse
+## Why HiSparse?
+In long-context LLM inference, each decoding request holds a full-length KV cache on GPU, limiting the number of concurrent requests a decode instance can serve. HiSparse addresse
+- **Reducing GPU memory per request**: Each request occupies only a fixed-size device buffer (e.g., 4KB tokens) instead of the full sequence length.
diff -- docs/basic_usage/deepseek_v32.md
@@ -468,3 +468,9 @@ python -m sglang.launch_server \
+## HiSparse: Hierarchical Sparse Attention for DSA (experimental)
+HiSparse reduces per-request GPU memory during decode by keeping only a small "hot" KV buffer on GPU while storing complete KV data in CPU pinned memory. A CUDA kernel dynamically
+HiSparse currently requires PD disaggregation mode and is enabled on the decode instance only. For detailed design, configuration, and deployment instructions, see the [HiSparse G
```

- Reviewed files:
  - docs: `docs/advanced_features/hisparse_guide.md` added +111/-0; `docs/basic_usage/deepseek_v32.md` modified +6/-0
- Risk and verification: This is mostly docs/examples in `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22268 - [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3

- Link: https://github.com/sgl-project/sglang/pull/22268
- Status/date: open / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3". The diff centers on `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation When LoRA adapters are applied to `fused_qkv_a_proj_with_mqa` in DeepSeek V2/V3 models, the fused GEMM optimization path in `prepare_qkv_latent` reads `.weight` di...
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

### PR #21932 - [HiSparse] Optimize the scheduling of decode backup.

- Link: https://github.com/sgl-project/sglang/pull/21932
- Status/date: merged / 2026-04-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +42/-9, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[HiSparse] Optimize the scheduling of decode backup.". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/managers/hisparse_coordinator.py`. PR body context: ## Motivation In overlap scheduling, the backup of decode tokens within the `prepare_for_decode` method currently requires waiting for the forward stream to complete, resulting...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -2800,6 +2800,12 @@ def _forward_raw(; symbols: _forward_raw, touching `_forward_raw`; `python/sglang/srt/managers/hisparse_coordinator.py` modified +36/-9 (45 lines); hunks: -78,8 +78,11 @@ def __init__(; -391,9 +394,6 @@ def _eager_backup_previous_token(; symbols: __init__, _eager_backup_previous_token, wait_for_pending_backup, get_front_topk_tokens, touching `__init__, _eager_backup_previous_token, wait_for_pending_backup`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -2800,6 +2800,12 @@ def _forward_raw(; symbols: _forward_raw
  - `python/sglang/srt/managers/hisparse_coordinator.py` modified +36/-9 (45 lines); hunks: -78,8 +78,11 @@ def __init__(; -391,9 +394,6 @@ def _eager_backup_previous_token(; symbols: __init__, _eager_backup_previous_token, wait_for_pending_backup, get_front_topk_tokens
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -2800,6 +2800,12 @@ def _forward_raw(
+        if (
+            self.hisparse_coordinator is not None
+            and forward_batch.forward_mode.is_decode()
+        ):
+            self.hisparse_coordinator.wait_for_pending_backup()
diff -- python/sglang/srt/managers/hisparse_coordinator.py
@@ -78,8 +78,11 @@ def __init__(
+        self.decode_backup_stream = device_module.Stream()
+        self._backup_done_event = device_module.Event()
+        self._has_pending_backup = False
@@ -391,9 +394,6 @@ def _eager_backup_previous_token(
-        if self.decode_producer_stream is not None:
-            device_module.current_stream().wait_stream(self.decode_producer_stream)
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +6/-0; `python/sglang/srt/managers/hisparse_coordinator.py` modified +36/-9
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22232 - Reduce unnecessary kernels and copies in the NSA indexer

- Link: https://github.com/sgl-project/sglang/pull/22232
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `671fe73961b7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-5, 48 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "Reduce unnecessary kernels and copies in the NSA indexer". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: cc @Jacob0226 ## Motivation The NSA indexer currently launches several unnecessary kernels: - `_project_and_scale_head_gates` and `_get_logits_head_gate` use multiple element-wi...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +13/-5 (18 lines); hunks: -257,13 +257,13 @@ def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor)...; -318,8 +318,8 @@ def _get_q_k_bf16(; symbols: _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, _get_logits_head_gate, _get_q_k_bf16, touching `_weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, _get_logits_head_gate`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +13/-5 (18 lines); hunks: -257,13 +257,13 @@ def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor)...; -318,8 +318,8 @@ def _get_q_k_bf16(; symbols: _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, _get_logits_head_gate, _get_q_k_bf16
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -257,13 +257,13 @@ def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor) -> torch.Tensor:
-    @torch.compile(dynamic=True) if not _is_hip else lambda f: f
+    @torch.compile(dynamic=True)
-    @torch.compile(dynamic=True) if not _is_hip else lambda f: f
+    @torch.compile(dynamic=True)
@@ -318,8 +318,8 @@ def _get_q_k_bf16(
-        query[..., : self.rope_head_dim] = q_rope.clone()
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +13/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22425 - [HiSparse]: Add HiSpares-DSA Model's nightly CI

- Link: https://github.com/sgl-project/sglang/pull/22425
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +84/-0, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HiSparse]: Add HiSpares-DSA Model's nightly CI". The diff centers on `test/registered/8-gpu-models/test_dsa_models_hisparse.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit te...
- Key implementation: `test/registered/8-gpu-models/test_dsa_models_hisparse.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGLM5DPHiSparse, setUpClass, tearDownClass, test_a_gsm8k, touching `TestGLM5DPHiSparse, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/8-gpu-models/test_dsa_models_hisparse.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: TestGLM5DPHiSparse, setUpClass, tearDownClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/8-gpu-models/test_dsa_models_hisparse.py
@@ -0,0 +1,84 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.run_eval import run_eval
+from sglang.test.test_utils import (
```

- Reviewed files:
  - tests: `test/registered/8-gpu-models/test_dsa_models_hisparse.py` added +84/-0
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_dsa_models_hisparse.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22424 - [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches

- Link: https://github.com/sgl-project/sglang/pull/22424
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `628df31d088f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +27/-3, 68 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: cc @Jacob0226 ## Motivation The current `LayerNorm` on HIP uses the torch implementation and triggers an extra dtype cast at both entry and exit. This results in 3 kernels per `...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-2 (14 lines); hunks: -16,12 +16,20; -212,7 +220,9 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-2 (14 lines); hunks: -16,12 +16,20; -212,7 +220,9 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -16,12 +16,20 @@
-from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu
+from sglang.srt.utils import (
+    add_prefix,
+    ceil_align,
+    get_bool_env_var,
+    is_cuda,
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22390 - [DSA] Enable all reduce fusion for DSA models

- Link: https://github.com/sgl-project/sglang/pull/22390
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSA] Enable all reduce fusion for DSA models". The diff centers on `python/sglang/srt/server_args.py`. PR body context: ## Motivation Including DeepSeek V3.2 and GLM-5 ## Modifications ## Accuracy Tests ## Speed Tests and Profiling ## Checklist - [ ] Format your code according to the Format code...
- Key implementation: `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -2130,7 +2130,9 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunks: -2130,7 +2130,9 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2130,7 +2130,9 @@ def _handle_model_specific_adjustments(self):
+                "DeepseekV32ForCausalLM",
+                "GlmMoeDsaForCausalLM",
```

- Reviewed files:
  - runtime: `python/sglang/srt/server_args.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22430 - [Fix] Fix several bugs on DSA models

- Link: https://github.com/sgl-project/sglang/pull/22430
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-5, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Fix] Fix several bugs on DSA models". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation Fix #22401 - Avoid hardcoding of nsa backends on sm100 - Fix topk transform method for draft model ## Modifications ## Accuracy Tests ## Speed Tests and Profiling...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunks: -2129,11 +2129,9 @@ def get_topk_transform_method(; symbols: get_topk_transform_method, touching `get_topk_transform_method`; `python/sglang/srt/server_args.py` modified +4/-2 (6 lines); hunks: -1490,8 +1490,10 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str,...; symbols: _set_default_nsa_backends, touching `_set_default_nsa_backends`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunks: -2129,11 +2129,9 @@ def get_topk_transform_method(; symbols: get_topk_transform_method
  - `python/sglang/srt/server_args.py` modified +4/-2 (6 lines); hunks: -1490,8 +1490,10 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str,...; symbols: _set_default_nsa_backends
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -2129,11 +2129,9 @@ def get_topk_transform_method(
+            and forward_mode == ForwardMode.EXTEND
-            if forward_mode is not None and (forward_mode.is_decode_or_idle()):
-                topk_transform_method = TopkTransformMethod.PAGED
diff -- python/sglang/srt/server_args.py
@@ -1490,8 +1490,10 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:
-                self.nsa_prefill_backend = "trtllm"
-                self.nsa_decode_backend = "trtllm"
+                if not user_set_prefill:
+                    self.nsa_prefill_backend = "trtllm"
+                if not user_set_decode:
+                    self.nsa_decode_backend = "trtllm"
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3; `python/sglang/srt/server_args.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22473 - [DSA] Add dense MLA decode fallback for short sequences

- Link: https://github.com/sgl-project/sglang/pull/22473
- Status/date: open / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +196/-3, 290 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSA] Add dense MLA decode fallback for short sequences". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/server_args.py`. PR body context: ## Summary - When all sequences in a decode batch have `seq_len 2048) sequences - [ ] Verify `SGLANG_NSA_DECODE_DENSE_ATTN_KV_LEN_THRESHOLD=0` disables the optimization - [ ] Be...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +143/-2 (145 lines); hunks: -322,6 +322,7 @@ def __init__(; -355,8 +356,13 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_decode, _forward_aiter_extend, touching `__init__, init_forward_metadata, forward_decode`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +44/-1 (45 lines); hunks: -748,6 +748,33 @@ def _forward_cuda_k_only(; -1076,7 +1103,7 @@ def forward_cuda(; symbols: _forward_cuda_k_only, _forward_decode_k_only, _get_topk_ragged_with_cp, forward_cuda, touching `_forward_cuda_k_only, _forward_decode_k_only, _get_topk_ragged_with_cp`; `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -1571,6 +1571,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`; `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ SGLang supports various environment variables that can be used....
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +143/-2 (145 lines); hunks: -322,6 +322,7 @@ def __init__(; -355,8 +356,13 @@ def __init__(; symbols: __init__, init_forward_metadata, forward_decode, _forward_aiter_extend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +44/-1 (45 lines); hunks: -748,6 +748,33 @@ def _forward_cuda_k_only(; -1076,7 +1103,7 @@ def forward_cuda(; symbols: _forward_cuda_k_only, _forward_decode_k_only, _get_topk_ragged_with_cp, forward_cuda
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -1571,6 +1571,13 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docs/references/environment_variables.md` modified +1/-0 (1 lines); hunks: -97,6 +97,7 @@ SGLang supports various environment variables that can be used...
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: -402,6 +402,7 @@ class Envs:; symbols: Envs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -322,6 +322,7 @@ def __init__(
+        self.use_dense_decode: bool = False
@@ -355,8 +356,13 @@ def __init__(
-        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
-        if self.device_sm_major >= 10 or self.nsa_decode_impl == "trtllm":
+        # Allocate global workspace buffer for TRT-LLM kernels
+        # (ragged attention on SM100/B200, trtllm decode, or dense decode fallback)
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -748,6 +748,33 @@ def _forward_cuda_k_only(
+    def _forward_decode_k_only(
+        self,
+        x: torch.Tensor,
+        positions: torch.Tensor,
+        forward_batch: ForwardBatch,
+        layer_id: int,
diff -- python/sglang/srt/server_args.py
@@ -1571,6 +1571,13 @@ def _handle_model_specific_adjustments(self):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +143/-2; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +44/-1; `python/sglang/srt/server_args.py` modified +7/-0; `python/sglang/srt/environ.py` modified +1/-0
  - docs: `docs/references/environment_variables.md` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/environ.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13546 - [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention

- Link: https://github.com/sgl-project/sglang/pull/13546
- Status/date: closed / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +254/-161, 634 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation ## Modifications ## Accuracy Tests ## Benchmarking and Profiling ## Checklist - [ ] Format your code according to the Format code with pre-commit. - [ ] Add unit t...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +191/-130 (321 lines); hunks: -4,10 +4,11; -20,7 +21,7; symbols: rotate_activation, V32LayerNorm, __init__, _forward_compiled, touching `rotate_activation, V32LayerNorm, __init__`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +57/-25 (82 lines); hunks: -10,21 +10,26; -244,6 +249,7 @@ def __init__(; symbols: __init__, forward_decode, _prepare_kv_cache, _prepare_qkv, touching `__init__, forward_decode, _prepare_kv_cache`; `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -56,7 +56,7; -1688,7 +1688,7 @@ def rebuild_cp_kv_cache(self, latent_cache, forward_batch,...; symbols: rebuild_cp_kv_cache, forward, touching `rebuild_cp_kv_cache, forward`; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -25,7 +25,7; -175,7 +175,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +191/-130 (321 lines); hunks: -4,10 +4,11; -20,7 +21,7; symbols: rotate_activation, V32LayerNorm, __init__, _forward_compiled
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +57/-25 (82 lines); hunks: -10,21 +10,26; -244,6 +249,7 @@ def __init__(; symbols: __init__, forward_decode, _prepare_kv_cache, _prepare_qkv
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunks: -56,7 +56,7; -1688,7 +1688,7 @@ def rebuild_cp_kv_cache(self, latent_cache, forward_batch,...; symbols: rebuild_cp_kv_cache, forward
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -25,7 +25,7; -175,7 +175,7 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +1/-1 (2 lines); hunks: -138,7 +138,7 @@ def cp_attn_tp_all_gather_reorganazied_into_tensor(; symbols: cp_attn_tp_all_gather_reorganazied_into_tensor, cp_all_gather_rerange_output, cp_all_gather_rearrange_output
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -4,10 +4,11 @@
+import torch.nn.functional as F
+from torch import nn
-from sglang.srt.layers.layernorm import LayerNorm
@@ -20,7 +21,7 @@
-    cp_all_gather_rerange_output,
+    cp_all_gather_rearrange_output,
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -10,21 +10,26 @@
-from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
+from sglang.srt.layers.attention.nsa.nsa_indexer import (
+    DUAL_STREAM_TOKEN_THRESHOLD,
+    BaseIndexerMetadata,
+)
+    NSA_DUAL_STREAM,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -56,7 +56,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +191/-130; `python/sglang/srt/layers/attention/nsa_backend.py` modified +57/-25; `python/sglang/srt/models/deepseek_v2.py` modified +3/-3; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2; `python/sglang/srt/layers/attention/nsa/utils.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa_backend.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22258 - [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization

- Link: https://github.com/sgl-project/sglang/pull/22258
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `dd4176448744`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +68/-25, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation When running GLM-5-FP8 on MI355X with FP8 quantization, the NSA indexer's head gate path (`weights_proj`) requires bf16 input. However, the upstream `fused_rms_fp8...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-7 (37 lines); hunks: -2,7 +2,7; -21,6 +21,7; symbols: _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, touching `_with_real_sm_count, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-7 (37 lines); hunks: -2,7 +2,7; -21,6 +21,7; symbols: _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -2,7 +2,7 @@
-from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
@@ -21,6 +21,7 @@
+    is_gfx95_supported,
@@ -31,6 +32,8 @@
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/communicator.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22372 - [DSA] Hopper FP8 FlashMLA KV padding

- Link: https://github.com/sgl-project/sglang/pull/22372
- Status/date: merged / 2026-04-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +43/-8, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[DSA] Hopper FP8 FlashMLA KV padding". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`. PR body context: ## Summary Adds q-head padding for `flashmla_kv` to support pure-TP configurations. FlashMLA requires q heads to be 64 or 128, but with tensor parallelism the per-GPU head count...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +39/-4 (43 lines); hunks: -326,6 +326,13 @@ def __init__(; -410,6 +417,16 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: __init__, init_forward_metadata, _forward_flashmla_kv, touching `__init__, init_forward_metadata, _forward_flashmla_kv`; `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -66,14 +66,14 @@ To serve GLM-5, just replace the `--model` argument with `za...; `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; symbols: _set_default_nsa_backends, touching `_set_default_nsa_backends`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +39/-4 (43 lines); hunks: -326,6 +326,13 @@ def __init__(; -410,6 +417,16 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: __init__, init_forward_metadata, _forward_flashmla_kv
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunks: -66,14 +66,14 @@ To serve GLM-5, just replace the `--model` argument with `za...
  - `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunks: -1502,9 +1502,9 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, m...; symbols: _set_default_nsa_backends
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -326,6 +326,13 @@ def __init__(
+        if self.num_q_heads <= 64:
+            self.flashmla_kv_num_q_heads = 64
+        elif self.num_q_heads <= 128:
+            self.flashmla_kv_num_q_heads = 128
+        else:
+            # Keep original head count if it exceeds current padded variants.
diff -- docs/basic_usage/deepseek_v32.md
@@ -66,14 +66,14 @@ To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
-  - `flashmla_auto`: enables automatic selection of either `flashmla_sparse` or `flashmla_kv` kernel for prefill based on KV cache dtype, hardware, and heuristics. When FP8 KV cac
+  - `flashmla_auto`: enables automatic selection of either `flashmla_sparse` or `flashmla_kv` kernel for prefill based on KV cache dtype, hardware, and heuristics. With BF16 KV ca
-    - Float8_e4m3fn KV cache: On Hopper, `flashmla_auto` prefill attention, `flashmla_kv` decode attention; On Blackwell, `trtllm` prefill attention and `trtllm` decode attention.
+    - Float8_e4m3fn KV cache: On Hopper, `flashmla_kv` prefill attention, `flashmla_kv` decode attention; On Blackwell, `trtllm` prefill attention and `trtllm` decode attention.
diff -- python/sglang/srt/server_args.py
@@ -1502,9 +1502,9 @@ def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:
-                # flashmla_auto dispatches to flashmla_sparse/flashmla_kv based on hardware and heuristics
+                # Hopper FP8 defaults to flashmla_kv for both prefill and decode.
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +39/-4; `python/sglang/srt/server_args.py` modified +2/-2
  - docs: `docs/basic_usage/deepseek_v32.md` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #21259 - [HiCache & HybridModel] mooncake backend support DSA & mamba model

- Link: https://github.com/sgl-project/sglang/pull/21259
- Status/date: merged / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +760/-232, 1439 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HiCache & HybridModel] mooncake backend support DSA & mamba model". The diff centers on `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`. PR body context: ## Motivation Added support for the Mooncake backend. Supports both Mamba and DSA models. Roadmap：https://github.com/sgl-project/sglang/issues/21846 ## Modifications ## Accuracy...
- Key implementation: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +230/-68 (298 lines); hunks: -64,6 +64,9; -187,9 +190,7 @@ def __init__(; symbols: synchronized, __init__, init_kv_buffer, touching `synchronized, __init__, init_kv_buffer`; `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` modified +180/-36 (216 lines); hunks: -5,7 +5,7; -15,6 +15,10; symbols: __init__, register_mem_pool_host, register_mem_host_pool_v2, _tag_keys, touching `__init__, register_mem_pool_host, register_mem_host_pool_v2`; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: build_nsa_hybrid_stack, layer_mapper, build_mamba_hybrid_stack, kv_layer_mapper, touching `build_nsa_hybrid_stack, layer_mapper, build_mamba_hybrid_stack`; `python/sglang/srt/mem_cache/hiradix_cache.py` modified +83/-32 (115 lines); hunks: -25,6 +25,17; -33,7 +44,6; symbols: __init__, attach_storage_backend, get_height, touching `__init__, attach_storage_backend, get_height`.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +230/-68 (298 lines); hunks: -64,6 +64,9; -187,9 +190,7 @@ def __init__(; symbols: synchronized, __init__, init_kv_buffer
  - `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` modified +180/-36 (216 lines); hunks: -5,7 +5,7; -15,6 +15,10; symbols: __init__, register_mem_pool_host, register_mem_host_pool_v2, _tag_keys
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` added +212/-0 (212 lines); hunks: -0,0 +1,212; symbols: build_nsa_hybrid_stack, layer_mapper, build_mamba_hybrid_stack, kv_layer_mapper
  - `python/sglang/srt/mem_cache/hiradix_cache.py` modified +83/-32 (115 lines); hunks: -25,6 +25,17; -33,7 +44,6; symbols: __init__, attach_storage_backend, get_height
  - `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +9/-88 (97 lines); hunks: -24,23 +24,18; -115,66 +110,6 @@ def __init__(self, params: CacheInitParams, server_args: Se...; symbols: __init__, kv_layer_mapper, mamba_layer_mapper
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/memory_pool_host.py
@@ -64,6 +64,9 @@
+# Host RAM to leave free when sizing HiCache pools (OS, other processes).
+HICACHE_HOST_MEMORY_RESERVE_BYTES: int = 10 * (1024**3)
@@ -187,9 +190,7 @@ def __init__(
-        # preserve at least 10GB for other usage
-        ten_gb = 10 * (1024**3)
-        available_bytes = host_mem.available - ten_gb
diff -- python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py
@@ -5,7 +5,7 @@
-from typing import Any, List, Optional
+from typing import Any, List, Optional, Tuple
@@ -15,6 +15,10 @@
+    PoolHitPolicy,
+    PoolName,
+    PoolTransfer,
diff -- python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py
@@ -0,0 +1,212 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/memory_pool_host.py` modified +230/-68; `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` modified +180/-36; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` added +212/-0; `python/sglang/srt/mem_cache/hiradix_cache.py` modified +83/-32; `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +9/-88; `python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py` modified +25/-2
  - tests: `test/registered/unit/mem_cache/test_nsa_pool_host_unit.py` modified +19/-5
- Risk and verification: The diff ships test coverage in `test/registered/unit/mem_cache/test_nsa_pool_host_unit.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22792 - nsa indexer: use aiter indexer_k_quant_and_cache

- Link: https://github.com/sgl-project/sglang/pull/22792
- Status/date: open / 2026-04-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 32 files, +701/-165, 1403 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "nsa indexer: use aiter indexer_k_quant_and_cache". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +53/-9 (62 lines); hunks: -152,6 +152,7; -167,8 +168,6; symbols: forward, __init__, op_prepare, touching `forward, __init__, op_prepare`; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, touching `forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-0 (29 lines); hunks: -23,6 +23,7; -999,7 +1000,35 @@ def _store_index_k_cache(; symbols: _store_index_k_cache, touching `_store_index_k_cache`; `python/sglang/srt/multimodal/processors/lfm2_vl.py` modified +8/-8 (16 lines); hunks: -12,9 +12,9; -56,7 +56,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async, touching `process_mm_data_async`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +53/-9 (62 lines); hunks: -152,6 +152,7; -167,8 +168,6; symbols: forward, __init__, op_prepare
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunks: -91,6 +91,7 @@ def forward_absorb_prepare(; -182,25 +183,31 @@ def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-0 (29 lines); hunks: -23,6 +23,7; -999,7 +1000,35 @@ def _store_index_k_cache(; symbols: _store_index_k_cache
  - `python/sglang/srt/multimodal/processors/lfm2_vl.py` modified +8/-8 (16 lines); hunks: -12,9 +12,9; -56,7 +56,7 @@ async def process_mm_data_async(; symbols: process_mm_data_async
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -1128,7 +1128,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -152,6 +152,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
@@ -167,8 +168,6 @@
-    from sglang.srt.utils.custom_op import register_custom_op
@@ -327,7 +326,7 @@ def forward(
-                if _device_sm >= 100 and self.weight.shape[0] == 256:
+                if _device_sm == 100 and self.weight.shape[0] == 256:
diff -- python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
@@ -91,6 +91,7 @@ def forward_absorb_prepare(
+        prev_topk_indices: Optional[torch.Tensor] = None,
@@ -182,25 +183,31 @@ def forward_absorb_prepare(
-                topk_indices = self.indexer(
-                    x=hidden_states,
-                    q_lora=q_lora,
-                    positions=positions,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -23,6 +23,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +53/-9; `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-0; `python/sglang/srt/multimodal/processors/lfm2_vl.py` modified +8/-8; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `python/sglang/multimodal_gen/test/run_suite.py`, `python/sglang/multimodal_gen/test/server/test_server_common.py`, `python/sglang/multimodal_gen/test/server/test_server_utils.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22851 - [RL] [DSv32] [GLM-5] Add `--nsa-topk-backend` and integrate FlashInfer and pytorch topk

- Link: https://github.com/sgl-project/sglang/pull/22851
- Status/date: open / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +584/-36, 776 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[RL] [DSv32] [GLM-5] Add `--nsa-topk-backend` and integrate FlashInfer and pytorch topk". The diff centers on `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`, `python/sglang/srt/server_args.py`. PR body context: ## Motivation @humansand Add `--nsa-topk-backend` for configurable topk backend implementation selection. `torch.topk` is used by GLM-5 for RL. FlashInfer topk has determinism a...
- Key implementation: `python/sglang/srt/layers/attention/nsa_backend.py` modified +245/-35 (280 lines); hunks: -1,8 +1,17; -75,6 +84,46; symbols: _nsa_topk_unfused, NSAFlashMLAMetadata, TopkTransformMethod, NSATopKBackend, touching `_nsa_topk_unfused, NSAFlashMLAMetadata, TopkTransformMethod`; `test/registered/kernels/test_nsa_indexer.py` modified +324/-1 (325 lines); hunks: -4,6 +4,7; -16,7 +17,13; symbols: __init__, _verify_topk_output, _run_unfused_topk_backend_validity_test, _run_fused_topk_backend_equivalence_test, touching `__init__, _verify_topk_output, _run_unfused_topk_backend_validity_test`; `python/sglang/srt/server_args.py` modified +12/-0 (12 lines); hunks: -227,6 +227,8; -490,6 +492,7 @@ class ServerArgs:; symbols: ServerArgs, add_cli_args, touching `ServerArgs, add_cli_args`; `python/sglang/srt/environ.py` modified +2/-0 (2 lines); hunks: -402,6 +402,8 @@ class Envs:; symbols: Envs, touching `Envs`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +245/-35 (280 lines); hunks: -1,8 +1,17; -75,6 +84,46; symbols: _nsa_topk_unfused, NSAFlashMLAMetadata, TopkTransformMethod, NSATopKBackend
  - `test/registered/kernels/test_nsa_indexer.py` modified +324/-1 (325 lines); hunks: -4,6 +4,7; -16,7 +17,13; symbols: __init__, _verify_topk_output, _run_unfused_topk_backend_validity_test, _run_fused_topk_backend_equivalence_test
  - `python/sglang/srt/server_args.py` modified +12/-0 (12 lines); hunks: -227,6 +227,8; -490,6 +492,7 @@ class ServerArgs:; symbols: ServerArgs, add_cli_args
  - `python/sglang/srt/environ.py` modified +2/-0 (2 lines); hunks: -402,6 +402,8 @@ class Envs:; symbols: Envs
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: -269,6 +269,7 @@ Please consult the documentation below and [server_args.py](...
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa_backend.py
@@ -1,8 +1,17 @@
-from enum import IntEnum, auto
-from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias
+from enum import Enum, IntEnum, auto
+from typing import (
+    TYPE_CHECKING,
+    Callable,
diff -- test/registered/kernels/test_nsa_indexer.py
@@ -4,6 +4,7 @@
+from sglang.srt.environ import envs
@@ -16,7 +17,13 @@
-from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend
+from sglang.srt.layers.attention.nsa_backend import (
+    NativeSparseAttnBackend,
+    NSAIndexerMetadata,
diff -- python/sglang/srt/server_args.py
@@ -227,6 +227,8 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa_backend.py` modified +245/-35; `python/sglang/srt/server_args.py` modified +12/-0; `python/sglang/srt/environ.py` modified +2/-0
  - tests: `test/registered/kernels/test_nsa_indexer.py` modified +324/-1
  - docs: `docs/advanced_features/server_arguments.md` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/kernels/test_nsa_indexer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22865 - [sparsity] extend framework to support non-NSA sparse algorithms

- Link: https://github.com/sgl-project/sglang/pull/22865
- Status/date: open / 2026-04-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +135/-30, 326 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[sparsity] extend framework to support non-NSA sparse algorithms". The diff centers on `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/forward_batch_info.py`. PR body context: - wire up sparse_coordinator, handling non-NSA sparsity ## Motivation Enable the use of `SparseCoordinator`. This allows for the use of sparse attention algorithms (e.g. Quest,...
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +41/-1 (42 lines); hunks: -475,6 +475,8 @@ def __init__(; -689,7 +691,11 @@ def initialize(self, pre_model_load_memory: float):; symbols: __init__, initialize, _forward_raw, touching `__init__, initialize, _forward_raw`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +32/-1 (33 lines); hunks: -998,6 +998,16 @@ def _fa_cp_attn(; -1033,7 +1043,20 @@ def forward_decode(; symbols: _fa_cp_attn, forward_decode, init_cuda_graph_state, touching `_fa_cp_attn, forward_decode, init_cuda_graph_state`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0 (4 lines); hunks: -71,6 +71,7; -430,6 +431,9 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: ForwardBatch, touching `ForwardBatch`; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +3/-1 (4 lines); hunks: -567,7 +567,9 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools, touching `_init_pools`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +41/-1 (42 lines); hunks: -475,6 +475,8 @@ def __init__(; -689,7 +691,11 @@ def initialize(self, pre_model_load_memory: float):; symbols: __init__, initialize, _forward_raw
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +32/-1 (33 lines); hunks: -998,6 +998,16 @@ def _fa_cp_attn(; -1033,7 +1043,20 @@ def forward_decode(; symbols: _fa_cp_attn, forward_decode, init_cuda_graph_state
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0 (4 lines); hunks: -71,6 +71,7; -430,6 +431,9 @@ class ForwardBatch(ForwardBatchDeepSeekMHAMixin):; symbols: ForwardBatch
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +3/-1 (4 lines); hunks: -567,7 +567,9 @@ def _init_pools(self: ModelRunner):; symbols: _init_pools
  - `python/sglang/srt/server_args.py` modified +27/-14 (41 lines); hunks: -6560,28 +6560,41 @@ def check_server_args(self):; symbols: check_server_args
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -475,6 +475,8 @@ def __init__(
+        # For generic sparse attention
+        self.sparse_coordinator = None
@@ -689,7 +691,11 @@ def initialize(self, pre_model_load_memory: float):
-        if self.enable_hisparse:
+        # Only NSA (DeepSeek) models use HiSparseCoordinator; non-NSA algorithms
+        # (SnapKV, Quest, …) use the generic SparseCoordinator below instead.
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -998,6 +998,16 @@ def _fa_cp_attn(
+        # Notify sparse coordinator after extend attention so it can construct
+        # per-page representations
+        _sparse_coord_ext = getattr(forward_batch, "sparse_coordinator", None)
+        if (
+            _sparse_coord_ext is not None
+            and not self.use_mla
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -71,6 +71,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +41/-1; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +32/-1; `python/sglang/srt/model_executor/forward_batch_info.py` modified +4/-0; `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +3/-1; `python/sglang/srt/server_args.py` modified +27/-14; `python/sglang/srt/managers/scheduler.py` modified +9/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/managers/scheduler.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22938 - [AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122)

- Link: https://github.com/sgl-project/sglang/pull/22938
- Status/date: open / 2026-04-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +12/-8, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR optimizes an inference path or backend selection. Title: "[AMD][MI30X] Restore DeepSeek MLA MI300X paths after MLA refactor (#19122)". The diff centers on `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_v2.py`. PR body context: ## Motivation Re-apply #18242 and #18624 behavior in forward_mla.py (logic moved out of deepseek_v2.py by #19122 but not fully carried over): ## Modifications - Import fused_qk_...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Allow piecewise CUDA graph with speculative decoding". The diff centers on `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/model_runner.py`, `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`. PR body context: ## Summary - Allow `--enable-piecewise-cuda-graph` to coexist with all speculative decoding algorithms (EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM) - Previously all speculative algorit...
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

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- Link: https://github.com/sgl-project/sglang/pull/22850
- Status/date: merged / 2026-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; associated commits `03828f420547`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +24/-5, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Motivation Redundant kernels in the NSA indexer on HIP: weights_proj: - The ReplicatedLinear layer uses fp32 params_dtype, preventing tgemm from dispatching to the tuned bf16...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache, touching `__init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunks: -14,7 +14,7; -32,14 +32,16; symbols: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -14,7 +14,7 @@
-from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
+from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
@@ -32,14 +32,16 @@
-_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
+if _use_aiter:
+    from aiter.ops.cache import indexer_k_quant_and_cache
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23195 - [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors

- Link: https://github.com/sgl-project/sglang/pull/23195
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +138/-14, 186 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors". The diff centers on `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`. PR body context: Hey all, this is my first PR, thought I'd try my hand a a recent bug. ---- ## Motivation Fixes #22855. When a DeepseekV2-architecture checkpoint is loaded with a `compressed_ten...
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

### PR #21249 - Support allreduce fusion with cp

- Link: https://github.com/sgl-project/sglang/pull/21249
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +201/-27, 382 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Support allreduce fusion with cp". The diff centers on `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py`. PR body context: ## Motivation Enable the allreduce_fusion with context parallel. This requires the allreduce_fusion can work with any sub comm group. ## Modifications This PR can only work afte...
- Key implementation: `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +178/-22 (200 lines); hunks: -8,10 +8,14; -20,7 +24,7; symbols: _always_disable_fabric, _FixedTorchDistBackend, __init__, bcast, touching `_always_disable_fabric, _FixedTorchDistBackend, __init__`; `python/sglang/srt/model_executor/model_runner.py` modified +22/-0 (22 lines); hunks: -719,6 +719,7 @@ def initialize(self, pre_model_load_memory: float):; -2181,6 +2182,27 @@ def kernel_warmup(self):; symbols: initialize, kernel_warmup, _pre_initialize_flashinfer_allreduce_workspace, _should_run_flashinfer_autotune, touching `initialize, kernel_warmup, _pre_initialize_flashinfer_allreduce_workspace`; `python/sglang/srt/layers/communicator.py` modified +1/-4 (5 lines); hunks: -160,9 +160,6 @@ def apply_flashinfer_allreduce_fusion(batch_size: int):; -518,7 +515,7 @@ def prepare_attn(; symbols: apply_flashinfer_allreduce_fusion, prepare_attn, touching `apply_flashinfer_allreduce_fusion, prepare_attn`; `python/sglang/srt/server_args.py` modified +0/-1 (1 lines); hunks: -2214,7 +2214,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, touching `_handle_model_specific_adjustments`.
- Code diff details:
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +178/-22 (200 lines); hunks: -8,10 +8,14; -20,7 +24,7; symbols: _always_disable_fabric, _FixedTorchDistBackend, __init__, bcast
  - `python/sglang/srt/model_executor/model_runner.py` modified +22/-0 (22 lines); hunks: -719,6 +719,7 @@ def initialize(self, pre_model_load_memory: float):; -2181,6 +2182,27 @@ def kernel_warmup(self):; symbols: initialize, kernel_warmup, _pre_initialize_flashinfer_allreduce_workspace, _should_run_flashinfer_autotune
  - `python/sglang/srt/layers/communicator.py` modified +1/-4 (5 lines); hunks: -160,9 +160,6 @@ def apply_flashinfer_allreduce_fusion(batch_size: int):; -518,7 +515,7 @@ def prepare_attn(; symbols: apply_flashinfer_allreduce_fusion, prepare_attn
  - `python/sglang/srt/server_args.py` modified +0/-1 (1 lines); hunks: -2214,7 +2214,6 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/flashinfer_comm_fusion.py
@@ -8,10 +8,14 @@
+    get_attn_tp_group,
+    get_moe_ep_group,
+    get_moe_tp_group,
+    get_tp_group,
@@ -20,7 +24,7 @@
-_workspace_manager = None
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -719,6 +719,7 @@ def initialize(self, pre_model_load_memory: float):
+            self._pre_initialize_flashinfer_allreduce_workspace()
@@ -2181,6 +2182,27 @@ def kernel_warmup(self):
+    def _pre_initialize_flashinfer_allreduce_workspace(self):
+        """Pre-initialize flashinfer allreduce fusion workspaces.
+        Must run before CUDA graph capture to avoid collective operations
+        (broadcasts, barriers) inside the graph capture context, which can
diff -- python/sglang/srt/layers/communicator.py
@@ -160,9 +160,6 @@ def apply_flashinfer_allreduce_fusion(batch_size: int):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +178/-22; `python/sglang/srt/model_executor/model_runner.py` modified +22/-0; `python/sglang/srt/layers/communicator.py` modified +1/-4; `python/sglang/srt/server_args.py` modified +0/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22914 - [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel

- Link: https://github.com/sgl-project/sglang/pull/22914
- Status/date: merged / 2026-04-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/models/deepseek_v2.py`; associated commits `c304d0d64d30`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +148/-402, 783 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR changes model-related implementation. Title: "[Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel". The diff centers on `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`. PR body context: ## Summary - Removed ~270 lines of duplicated context-parallel utility functions from `layers/attention/nsa/utils.py`, consolidating them into `layers/utils/cp_utils.py` - Unifi...
- Key implementation: `python/sglang/srt/layers/attention/nsa/utils.py` modified +2/-353 (355 lines); hunks: -1,24 +1,14; -135,27 +125,7 @@ def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", ns...; symbols: pad_nsa_cache_seqlens, NSAContextParallelMetadata, can_cp_split, can_nsa_cp_split, touching `pad_nsa_cache_seqlens, NSAContextParallelMetadata, can_cp_split`; `python/sglang/srt/layers/utils/cp_utils.py` modified +103/-12 (115 lines); hunks: -5,7 +5,15; -60,6 +68,18 @@ def can_cp_split(seq_len: int, cp_size: int, forward_batch):; symbols: can_cp_split, cp_split_and_rebuild_data, cp_split_and_rebuild_position, cp_all_gather_reorganized_into_tensor, touching `can_cp_split, cp_split_and_rebuild_data, cp_split_and_rebuild_position`; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-17 (37 lines); hunks: -53,14 +53,14; -358,7 +358,7 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16, forward_cuda, forward_npu, touching `_get_q_k_bf16, forward_cuda, forward_npu`; `python/sglang/srt/models/deepseek_v2.py` modified +11/-7 (18 lines); hunks: -55,13 +55,9; -112,6 +108,12; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +2/-353 (355 lines); hunks: -1,24 +1,14; -135,27 +125,7 @@ def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", ns...; symbols: pad_nsa_cache_seqlens, NSAContextParallelMetadata, can_cp_split, can_nsa_cp_split
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +103/-12 (115 lines); hunks: -5,7 +5,15; -60,6 +68,18 @@ def can_cp_split(seq_len: int, cp_size: int, forward_batch):; symbols: can_cp_split, cp_split_and_rebuild_data, cp_split_and_rebuild_position, cp_all_gather_reorganized_into_tensor
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-17 (37 lines); hunks: -53,14 +53,14; -358,7 +358,7 @@ def _get_q_k_bf16(; symbols: _get_q_k_bf16, forward_cuda, forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-7 (18 lines); hunks: -55,13 +55,9; -112,6 +108,12; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/utils.py
@@ -1,24 +1,14 @@
-# temp NSA debugging environ
-from dataclasses import dataclass
-from itertools import accumulate
-import torch.nn.functional as F
-from sglang.srt.distributed.device_communicators.pynccl_allocator import (
-    use_symmetric_memory,
diff -- python/sglang/srt/layers/utils/cp_utils.py
@@ -5,7 +5,15 @@
-from sglang.srt.layers.dp_attention import get_attention_cp_group
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
+)
+from sglang.srt.layers.dp_attention import (
+    attn_cp_all_gather_into_tensor,
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -53,14 +53,14 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/utils.py` modified +2/-353; `python/sglang/srt/layers/utils/cp_utils.py` modified +103/-12; `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-17; `python/sglang/srt/models/deepseek_v2.py` modified +11/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23257 - Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention

- Link: https://github.com/sgl-project/sglang/pull/23257
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-0, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention". The diff centers on `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`. PR body context: ## Motivation Running `nvidia/DeepSeek-V3-0324-NVFP4` with `--moe-runner-backend flashinfer_cutedsl --enable-dp-attention --ep-size N --dp-size N` (N>1) hit two separate bugs. *...
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

### PR #23268 - 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache

- Link: https://github.com/sgl-project/sglang/pull/23268
- Status/date: open / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +21/-5, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`. PR body context: ## Motivation when enable both nsa cp and prefixcache, inference accuracy is abnormal. ## Modifications python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_...
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

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- Link: https://github.com/sgl-project/sglang/pull/22003
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +276/-25, 485 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Support moe_dp_size = 1 for various attention_cp_size". The diff centers on `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py`. PR body context: ## Motivation Previously, we can only support attention_cp_size == moe_dp_size which is too restricted. In the real world case, we should let the MoE part unchanged and only app...
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode, touching `ScatterMode, model_input_output, _compute_layer_input_mode`; `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size, touching `attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank`; `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings, touching `__init__, get_input_embeddings`; `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split, touching `is_prefill_cp_in_seq_split, can_cp_split`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-1 (6 lines); hunks: -33,6 +33,9; -709,6 +712,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -50,8 +50,12 @@
+    get_moe_cp_rank,
+    get_moe_cp_size,
+    is_enable_moe_cp_allgather,
+    moe_cp_all_gather_into_tensor,
@@ -188,11 +192,13 @@ class ScatterMode(Enum):
+    MOE_FULL: full within the MoE group (cp_per_moe CP chunks), used when moe_dp_size < attn_cp_size
diff -- python/sglang/srt/layers/dp_attention.py
@@ -18,6 +18,9 @@
+)
+from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
+from sglang.srt.distributed import (
@@ -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
+def get_moe_cp_group() -> GroupCoordinator:
+    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -968,9 +968,10 @@ def __init__(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +164/-10; `python/sglang/srt/layers/dp_attention.py` modified +28/-0; `python/sglang/srt/models/qwen3_moe.py` modified +4/-3; `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4; `python/sglang/srt/models/qwen2_moe.py` modified +5/-1; `python/sglang/srt/distributed/parallel_state.py` modified +13/-7
  - tests: `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_qwen3_30b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1296/-33, 1579 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1". The diff centers on `python/sglang/srt/model_executor/cuda_graph_runner.py`, `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`. PR body context: ## Motivation One of the core parameters of speculative decoding is `speculative_num_steps`, which controls how many autoregressive draft-model steps are executed in each round....
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

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- Link: https://github.com/sgl-project/sglang/pull/23219
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +41/-15, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[AMD] Enable MTP for GLM-5-mxfp4 model". The diff centers on `python/sglang/srt/models/deepseek_nextn.py`. PR body context: ## Motivation Fix https://github.com/sgl-project/sglang/issues/23142. Quark-quantized GLM-5-MXFP4 checkpoints store MTP (NextN) weights — including `eh_proj` — in FP4-packed for...
- Key implementation: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunks: -42,6 +42,7; -99,7 +100,18 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -42,6 +42,7 @@
+from sglang.srt.layers.linear import ReplicatedLinear
@@ -99,7 +100,18 @@ def __init__(
-        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
+        if quant_config is not None and quant_config.get_name() == "quark":
+            self.eh_proj = ReplicatedLinear(
+                2 * config.hidden_size,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_nextn.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- Link: https://github.com/sgl-project/sglang/pull/23315
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +72/-4, 131 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Opt-in strip of thinking tokens from radix cache". The diff centers on `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`. PR body context: Fixes #22373. Closes #22617, #22950. Opt-in: enable with `--strip-thinking-cache`. Off by default. ## Why Reasoning-model requests (`--reasoning-parser `) insert all output toke...
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
- Motivation: For DeepSeek V3.2, this PR fixes a launch, loading, parsing, or numerical issue. Title: "[fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)". The diff centers on `python/sglang/srt/parser/reasoning_parser.py`, `python/sglang/srt/configs/model_config.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking.py`. PR body context: # Fix dead reasoning branches in radix cache for multi-turn separated-thinking models Fixes #22373. Related to #22617. --- ## Motivation ### What is the problem Reasoning models...
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
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[SPEC V2][2/N] feat: adaptive spec support spec v2". The diff centers on `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`. PR body context: ## Motivation adaptive spec support spec v2: 1. launch sgl+adaptive spec+spec v2 2. benchmark 3. result Models: - Target model: `/models/ZhipuAI/GLM-4.7-FP8` - Draft model: `/mo...
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

### PR #23351 - Support piecewise CUDA graph with NSA

- Link: https://github.com/sgl-project/sglang/pull/23351
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +302/-56, 646 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "Support piecewise CUDA graph with NSA". The diff centers on `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/radix_attention.py`. PR body context: ## Motivation GLM-5/DSV3.2 currently doesn't allow piecewise CUDA graph due to incompatibilities in NSA attention backend and NSA indexer. This commit fixes the incompatibilitie...
- Key implementation: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +163/-34 (197 lines); hunks: -11,6 +11,10; -73,6 +77,82; symbols: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata, touching `k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg`; `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunks: -51,7 +51,26; symbols: _layernorm_fake_impl, layernorm, touching `_layernorm_fake_impl, layernorm`; `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunks: -148,6 +148,12 @@ def unified_attention_with_output(; -166,6 +172,14 @@ def unified_attention_with_output(; symbols: unified_attention_with_output, touching `unified_attention_with_output`; `python/sglang/srt/layers/attention/nsa_backend.py` modified +5/-2 (7 lines); hunks: -1,12 +1,15; -2116,8 +2119,8 @@ def _forward_trtllm(; symbols: _forward_trtllm, _pad_topk_indices, touching `_forward_trtllm, _pad_topk_indices`.
- Code diff details:
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +163/-34 (197 lines); hunks: -11,6 +11,10; -73,6 +77,82; symbols: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata
  - `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunks: -51,7 +51,26; symbols: _layernorm_fake_impl, layernorm
  - `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunks: -148,6 +148,12 @@ def unified_attention_with_output(; -166,6 +172,14 @@ def unified_attention_with_output(; symbols: unified_attention_with_output
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +5/-2 (7 lines); hunks: -1,12 +1,15; -2116,8 +2119,8 @@ def _forward_trtllm(; symbols: _forward_trtllm, _pad_topk_indices
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunks: -2656,6 +2656,7 @@ def init_piecewise_cuda_graphs(self):; -2708,6 +2709,11 @@ def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/nsa/nsa_indexer.py
@@ -11,6 +11,10 @@
+from sglang.srt.compilation.piecewise_context_manager import (
+    get_forward_context,
+    is_in_piecewise_cuda_graph,
+)
@@ -73,6 +77,82 @@
+if _is_cuda:
diff -- python/sglang/srt/layers/layernorm.py
@@ -51,7 +51,26 @@
-            from flashinfer.norm import layernorm
+            import flashinfer.norm
+            from sglang.srt.utils.custom_op import register_custom_op
+            def _layernorm_fake_impl(
+                input: torch.Tensor,
+                gamma: torch.Tensor,
diff -- python/sglang/srt/layers/radix_attention.py
@@ -148,6 +148,12 @@ def unified_attention_with_output(
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +163/-34; `python/sglang/srt/layers/layernorm.py` modified +20/-1; `python/sglang/srt/layers/radix_attention.py` modified +14/-0; `python/sglang/srt/layers/attention/nsa_backend.py` modified +5/-2; `python/sglang/srt/model_executor/model_runner.py` modified +6/-0; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +4/-0
  - tests: `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py` added +70/-0
- Risk and verification: The diff ships test coverage in `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22774 - [MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)

- Link: https://github.com/sgl-project/sglang/pull/22774
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +184/-44, 795 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[MUSA][16/N] Add MUSA backend support for layers and DeepSeek models (V2/V3/R1)". The diff centers on `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`. PR body context: ## Motivation Enable SGLang to run DeepSeek models on Moore Threads MUSA GPUs. This PR adds MUSA backend support across the inference stack, including layers, quantization, MoE,...
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

### PR #23241 - [HiCache & HybridModel] 3FS backend support DSA & mamba model

- Link: https://github.com/sgl-project/sglang/pull/23241
- Status/date: merged / 2026-04-24
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +399/-55, 733 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: For DeepSeek V3.2, this PR adds or enables a model support/runtime surface. Title: "[HiCache & HybridModel] 3FS backend support DSA & mamba model". The diff centers on `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`, `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`, `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`. PR body context: ## Motivation Added support for the 3FS backend. Supports both Mamba and DSA models. Roadmap：#21846 ## Modifications ## Accuracy Tests 3FS Config **mamba** **DSA （DeepSeek-V3.2-...
- Key implementation: `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py` modified +287/-5 (292 lines); hunks: -7,6 +7,7; -16,6 +17,10; symbols: Hf3fsMetadataInterface, initialize, reserve_and_allocate_page_indices, confirm_write, touching `Hf3fsMetadataInterface, initialize, reserve_and_allocate_page_indices`; `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py` modified +111/-50 (161 lines); hunks: -14,6 +14,7; -115,7 +116,7 @@ class GlobalMetadataState:; symbols: GlobalMetadataState, __init__, load_from_disk, save_to_disk, touching `GlobalMetadataState, __init__, load_from_disk`; `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +1/-0 (1 lines); hunks: -142,6 +142,7 @@ def __init__(self, params: CacheInitParams, server_args: Ser...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py` modified +287/-5 (292 lines); hunks: -7,6 +7,7; -16,6 +17,10; symbols: Hf3fsMetadataInterface, initialize, reserve_and_allocate_page_indices, confirm_write
  - `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py` modified +111/-50 (161 lines); hunks: -14,6 +14,7; -115,7 +116,7 @@ class GlobalMetadataState:; symbols: GlobalMetadataState, __init__, load_from_disk, save_to_disk
  - `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +1/-0 (1 lines); hunks: -142,6 +142,7 @@ def __init__(self, params: CacheInitParams, server_args: Ser...; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py
@@ -7,6 +7,7 @@
+from dataclasses import dataclass
@@ -16,6 +17,10 @@
+    PoolHitPolicy,
+    PoolName,
+    PoolTransfer,
+    PoolTransferResult,
diff -- python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py
@@ -14,6 +14,7 @@
+from sglang.srt.mem_cache.hicache_storage import PoolName
@@ -115,7 +116,7 @@ class GlobalMetadataState:
-        self.ranks: Dict[int, RankMetadata] = {}
+        self.ranks: Dict[str, RankMetadata] = {}
@@ -132,13 +133,14 @@ def load_from_disk(self):
-                for rank_id_str, data in persisted_data.items():
diff -- python/sglang/srt/mem_cache/hi_mamba_radix_cache.py
@@ -142,6 +142,7 @@ def __init__(self, params: CacheInitParams, server_args: ServerArgs):
```

- Reviewed files:
  - runtime: `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py` modified +287/-5; `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py` modified +111/-50; `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`, `python/sglang/srt/mem_cache/storage/hf3fs/mini_3fs_metadata_server.py`, `python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- This version rejects title-only PR lists; every PR must include trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
