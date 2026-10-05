# TensorRT-LLM Kimi K2/K2.5/K3/Linear/VL Model PR Optimization History

## 2026-08-23 Source Head Refresh

Rechecked TensorRT-LLM upstream main at `NVIDIA/TensorRT-LLM@da38c1d2e0dffd073b7dfb6d69e15ee7b45d84a9`.
The seven-commit range from the previous recorded head
`1b4ffc0291d75a21ad20118e8f44de6e3831f786` was inspected with
`git log --name-only` and complete local source diffs.

Result: PR #16805 is promoted because it fixes disaggregated speculative
draft-token and sequence-length accounting used by Kimi-style PD flows.
PR #16763 is retained as a cross-model startup-memory card. The remaining four
commits only adjust CI, fakes, Slurm cleanup, or Docker copies and are not
presented as model optimization evidence. The seventh commit, VisualGen/Wan
Attention2D plus TP PR #16677, is also outside the Kimi LLM scope. PR #14848
remains the prior high-signal Kimi runtime merge.

## 2026-06-27 PR Backfill Audit

The per-PR diff audit cards on this page were generated from TensorRT-LLM
upstream `HEAD@4164b932c6c8a14d1be85d0fd62e44b7d0171980`. The root
TensorRT-LLM history index now tracks the 2026-06-27 runtime refresh at
`aaffa2f9fef3025e0f698d978385a73460344e0b`. This page provides model
implementation coverage, a timeline, and per-PR diff audit cards for Kimi K2
Thinking / Kimi K2.5.

Filter used in this pass: merged PRs whose titles or files matched `Kimi`, `kimi_k25`, `KimiK25`, `K2.5`, `K2 Thinking`, `NVFP4`, `multimodal`, `tool_parser`, `reasoning_parser`, `guided decoding`, `spec dec`, `rejection sampling`, or `NIXL`. Formatting-only and unrelated infrastructure PRs were excluded.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `cpp/tensorrt_llm/kernels/kimiK3AttnRes/CMakeLists.txt` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225), [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225), [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225), [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu` | [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.h` | [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711), [#11645](https://github.com/NVIDIA/TensorRT-LLM/pull/11645) |
| `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327), [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333), [#17800](https://github.com/NVIDIA/TensorRT-LLM/pull/17800), [#17845](https://github.com/NVIDIA/TensorRT-LLM/pull/17845), [#18164](https://github.com/NVIDIA/TensorRT-LLM/pull/18164), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179), [#19191](https://github.com/NVIDIA/TensorRT-LLM/pull/19191) |
| `examples/configs/curated/kimi-k2-thinking.yaml` | [#11645](https://github.com/NVIDIA/TensorRT-LLM/pull/11645) |
| `examples/kimi_k3/README.md` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327), [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333), [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334), [#17455](https://github.com/NVIDIA/TensorRT-LLM/pull/17455), [#17456](https://github.com/NVIDIA/TensorRT-LLM/pull/17456), [#17800](https://github.com/NVIDIA/TensorRT-LLM/pull/17800), [#17939](https://github.com/NVIDIA/TensorRT-LLM/pull/17939), [#18164](https://github.com/NVIDIA/TensorRT-LLM/pull/18164), [#19784](https://github.com/NVIDIA/TensorRT-LLM/pull/19784) |
| `examples/kimi_k3/disagg/README.md` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334), [#17479](https://github.com/NVIDIA/TensorRT-LLM/pull/17479), [#17483](https://github.com/NVIDIA/TensorRT-LLM/pull/17483), [#17939](https://github.com/NVIDIA/TensorRT-LLM/pull/17939) |
| `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334), [#17479](https://github.com/NVIDIA/TensorRT-LLM/pull/17479) |
| `examples/kimi_k3/disagg/ctx_config.yaml` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334), [#17483](https://github.com/NVIDIA/TensorRT-LLM/pull/17483) |
| `examples/kimi_k3/disagg/disagg_proxy_config.yaml` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334) |
| `examples/kimi_k3/disagg/gen_config.yaml` | [#17939](https://github.com/NVIDIA/TensorRT-LLM/pull/17939) |
| `examples/kimi_k3/disagg/gen_config_no_sa.yaml` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334) |
| `examples/kimi_k3/eval_extra_llm_options.yaml` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333), [#17624](https://github.com/NVIDIA/TensorRT-LLM/pull/17624) |
| `examples/kimi_k3/eval_extra_llm_options_dflash.yaml` | [#17846](https://github.com/NVIDIA/TensorRT-LLM/pull/17846) |
| `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml` | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) |
| `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_gpqa.yaml` | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) |
| `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe.yaml` | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) |
| `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml` | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) |
| `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep8.yaml` | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) |
| `examples/kimi_k3/eval_extra_llm_options_reuse.yaml` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/kimi_k3/eval_extra_llm_options_sa.yaml` | [#17455](https://github.com/NVIDIA/TensorRT-LLM/pull/17455) |
| `examples/kimi_k3/make_synthetic_dflash_drafter.py` | [#17846](https://github.com/NVIDIA/TensorRT-LLM/pull/17846) |
| `examples/kimi_k3/measure_dspark_acceptance.py` | [#17846](https://github.com/NVIDIA/TensorRT-LLM/pull/17846) |
| `examples/kimi_k3/perf_sweep/acc_sweep.sbatch` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) |
| `examples/kimi_k3/perf_sweep/perf_sweep.sbatch` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) |
| `examples/kimi_k3/perf_sweep/submit_acc_sweep.sh` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/kimi_k3/perf_sweep/submit_perf_sweep.sh` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/kimi_k3/quick_start_kimi_k3.py` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/kimi_k3/quick_start_kimi_k3.sbatch` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/kimi_k3/run_dspark_acceptance.sbatch` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17846](https://github.com/NVIDIA/TensorRT-LLM/pull/17846) |
| `examples/kimi_k3/run_eval_kimi_k3.sbatch` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) |
| `examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch` | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) |
| `examples/models/core/kimi_k2/README.md` | no direct PR-number commit |
| `examples/models/core/kimi_k2/kimi_k2_tool_calling_example.py` | no direct PR-number commit |
| `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml` | [#11645](https://github.com/NVIDIA/TensorRT-LLM/pull/11645) |
| `tensorrt_llm/_torch/configs/kimi_k3.py` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050) |
| `tensorrt_llm/_torch/configs/kimi_linear.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) |
| `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225), [#18294](https://github.com/NVIDIA/TensorRT-LLM/pull/18294), [#19040](https://github.com/NVIDIA/TensorRT-LLM/pull/19040), [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/__init__.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/akk_inverse.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/kda_mtp_decode.py` | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) |
| `tensorrt_llm/_torch/kimi_k3_cache_policy.py` | no direct PR-number commit |
| `tensorrt_llm/_torch/models/modeling_kimi_k25.py` | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788), [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379), [#14392](https://github.com/NVIDIA/TensorRT-LLM/pull/14392), [#14741](https://github.com/NVIDIA/TensorRT-LLM/pull/14741), [#15179](https://github.com/NVIDIA/TensorRT-LLM/pull/15179), [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180), [#16482](https://github.com/NVIDIA/TensorRT-LLM/pull/16482), [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17125](https://github.com/NVIDIA/TensorRT-LLM/pull/17125), [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865), [#19277](https://github.com/NVIDIA/TensorRT-LLM/pull/19277) |
| `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17845](https://github.com/NVIDIA/TensorRT-LLM/pull/17845), [#18064](https://github.com/NVIDIA/TensorRT-LLM/pull/18064) |
| `tensorrt_llm/_torch/models/modeling_kimi_linear.py` | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050), [#17053](https://github.com/NVIDIA/TensorRT-LLM/pull/17053), [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#17311](https://github.com/NVIDIA/TensorRT-LLM/pull/17311), [#17312](https://github.com/NVIDIA/TensorRT-LLM/pull/17312), [#17413](https://github.com/NVIDIA/TensorRT-LLM/pull/17413), [#17421](https://github.com/NVIDIA/TensorRT-LLM/pull/17421), [#17446](https://github.com/NVIDIA/TensorRT-LLM/pull/17446), [#17480](https://github.com/NVIDIA/TensorRT-LLM/pull/17480), [#17624](https://github.com/NVIDIA/TensorRT-LLM/pull/17624), [#17684](https://github.com/NVIDIA/TensorRT-LLM/pull/17684), [#17741](https://github.com/NVIDIA/TensorRT-LLM/pull/17741), ... (26 total) |
| `tensorrt_llm/_torch/modules/kimi_k3_attn_res/__init__.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) |
| `tensorrt_llm/_torch/modules/kimi_k3_attn_res/_attn_res_kernels.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) |
| `tensorrt_llm/_torch/modules/kimi_k3_attn_res/kimi_k3_attn_res.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) |
| `tensorrt_llm/_torch/modules/kimi_k3_mla/__init__.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) |
| `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#17445](https://github.com/NVIDIA/TensorRT-LLM/pull/17445), [#17684](https://github.com/NVIDIA/TensorRT-LLM/pull/17684), [#17796](https://github.com/NVIDIA/TensorRT-LLM/pull/17796), [#17800](https://github.com/NVIDIA/TensorRT-LLM/pull/17800), [#17816](https://github.com/NVIDIA/TensorRT-LLM/pull/17816), [#18164](https://github.com/NVIDIA/TensorRT-LLM/pull/18164), [#18226](https://github.com/NVIDIA/TensorRT-LLM/pull/18226), [#18728](https://github.com/NVIDIA/TensorRT-LLM/pull/18728), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) |
| `tensorrt_llm/_torch/modules/kimi_kda/__init__.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#17822](https://github.com/NVIDIA/TensorRT-LLM/pull/17822) |
| `tensorrt_llm/_torch/modules/kimi_kda/_kda_decode.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#18251](https://github.com/NVIDIA/TensorRT-LLM/pull/18251) |
| `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#17822](https://github.com/NVIDIA/TensorRT-LLM/pull/17822), [#17870](https://github.com/NVIDIA/TensorRT-LLM/pull/17870) |
| `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` | [#18064](https://github.com/NVIDIA/TensorRT-LLM/pull/18064), [#18294](https://github.com/NVIDIA/TensorRT-LLM/pull/18294) |
| `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269), [#17741](https://github.com/NVIDIA/TensorRT-LLM/pull/17741), [#17816](https://github.com/NVIDIA/TensorRT-LLM/pull/17816), [#17822](https://github.com/NVIDIA/TensorRT-LLM/pull/17822), [#17870](https://github.com/NVIDIA/TensorRT-LLM/pull/17870), [#18251](https://github.com/NVIDIA/TensorRT-LLM/pull/18251), [#18294](https://github.com/NVIDIA/TensorRT-LLM/pull/18294), [#18643](https://github.com/NVIDIA/TensorRT-LLM/pull/18643), [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179), [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `tensorrt_llm/serve/extensions/kimi_k3.py` | no direct PR-number commit |
| `tensorrt_llm/serve/tool_parser/kimi_k2_tool_parser.py` | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) |
| `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` | [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327), [#17845](https://github.com/NVIDIA/TensorRT-LLM/pull/17845), [#17980](https://github.com/NVIDIA/TensorRT-LLM/pull/17980) |
| `tests/integration/defs/accuracy/test_kimi3.py` | [#18461](https://github.com/NVIDIA/TensorRT-LLM/pull/18461), [#18699](https://github.com/NVIDIA/TensorRT-LLM/pull/18699), [#19084](https://github.com/NVIDIA/TensorRT-LLM/pull/19084) |
| `tests/integration/defs/kimi_k3_disagg_parity.py` | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334), [#17921](https://github.com/NVIDIA/TensorRT-LLM/pull/17921) |
| `tests/integration/defs/kimi_k3_sa_harness.py` | [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327), [#18682](https://github.com/NVIDIA/TensorRT-LLM/pull/18682) |
| `tests/integration/defs/test_kimi_k3_specdec.py` | [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327), [#17921](https://github.com/NVIDIA/TensorRT-LLM/pull/17921), [#18682](https://github.com/NVIDIA/TensorRT-LLM/pull/18682) |
| `tests/microbenchmarks/kimi_k3_attn_res_add_rmsnorm.py` | [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443), [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960), [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) |
| ... | 49 more files omitted from table; all were used for git tracing. |

## PR Coverage Summary

- Git-traced PRs: 72
- Extra PRs preserved from existing docs: 7
- Total PRs in this document: 79
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-12-05 | [#9711](https://github.com/NVIDIA/TensorRT-LLM/pull/9711) | merged | Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell | deployment guide |
| 2025-12-12 | [#9830](https://github.com/NVIDIA/TensorRT-LLM/pull/9830) | merged | Support tool parser for Kimi K2 | OpenAI server/tool parser |
| 2026-02-24 | [#11645](https://github.com/NVIDIA/TensorRT-LLM/pull/11645) | merged | [None][chore] Moving kimi-k2-thinking deployment guide configs to config files. | `examples/configs/curated/kimi-k2-thinking.yaml`, `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md`, `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml` |
| 2026-03-04 | [#11777](https://github.com/NVIDIA/TensorRT-LLM/pull/11777) | merged | Add Kimi-K2.5 text model support (NVFP4) | `modeling_deepseekv3.py`, accuracy tests |
| 2026-03-05 | [#11780](https://github.com/NVIDIA/TensorRT-LLM/pull/11780) | merged | AutoDeploy onboarding agent + Kimi K2.5 AD modeling code | AutoDeploy Kimi model/config/tests |
| 2026-05-11 | [#13801](https://github.com/NVIDIA/TensorRT-LLM/pull/13801) | merged | Add reasoning parser for kimi-k2.5 and enable auto flow | command/reasoning parser |
| 2026-05-14 | [#12788](https://github.com/NVIDIA/TensorRT-LLM/pull/12788) | merged | Add Kimi K2.5 multimodal vision support | `modeling_kimi_k25.py`, multimodal eval/tests |
| 2026-05-22 | [#14379](https://github.com/NVIDIA/TensorRT-LLM/pull/14379) | merged | Fix Kimi_k25 with spec dec | `modeling_kimi_k25.py` |
| 2026-05-25 | [#14392](https://github.com/NVIDIA/TensorRT-LLM/pull/14392) | merged | [https://nvbugs/6182617][fix] Restore K2.5 multimodal dep8 accuracy test on transformers 5.5.x | `tensorrt_llm/_torch/models/modeling_kimi_k25.py` |
| 2026-06-09 | [#14741](https://github.com/NVIDIA/TensorRT-LLM/pull/14741) | merged | [https://nvbugs/6227203][fix] Remove redundant TikTokenTokenizer shim from KimiK25InputProcessor | `tensorrt_llm/_torch/models/modeling_kimi_k25.py` |
| 2026-06-11 | [#14960](https://github.com/NVIDIA/TensorRT-LLM/pull/14960) | merged | [None][test] Update K2.5 andGLM-5 into CI Perf Test | `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` |
| 2026-06-17 | [#15233](https://github.com/NVIDIA/TensorRT-LLM/pull/15233) | merged | Fix embedding vocab mask for rejection sampling in Kimi-K2.5 | `embedding.py` |
| 2026-06-23 | [#15443](https://github.com/NVIDIA/TensorRT-LLM/pull/15443) | merged | Un-waive K2.5 Thinking FP4 disagg-NIXL tests | waives and perf-sanity YAML |
| 2026-06-25 | [#15180](https://github.com/NVIDIA/TensorRT-LLM/pull/15180) | merged | Add necessary methods for guided decoding in Kimi K2.5 | `modeling_kimi_k25.py` |
| 2026-07-15 | [#14848](https://github.com/NVIDIA/TensorRT-LLM/pull/14848) | merged | [TRTLLM-12373][feat] RMSNorm nvfp4 quant fusion for DS V3.2 / Kimi-K2.5 | `tensorrt_llm/_torch/models/modeling_deepseekv3.py`, `tensorrt_llm/_torch/models/modeling_nemotron_h.py`, `tests/unittest/_torch/modules/test_fused_rmsnorm_fp4_quantize.py` |
| 2026-07-16 | [#15966](https://github.com/NVIDIA/TensorRT-LLM/pull/15966) | merged | [TRTLLM-13639][perf] Migrate Kimi perf-sanity tests to Transceiver v2 | `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` |
| 2026-07-21 | [#16482](https://github.com/NVIDIA/TensorRT-LLM/pull/16482) | merged | [TRTLLM-13639][test] Migrate Kimi dis-agg tests to Transceiver v2 and trim tests | `tensorrt_llm/_torch/models/modeling_kimi_k25.py`, `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` |
| 2026-07-27 | [#16763](https://github.com/NVIDIA/TensorRT-LLM/pull/16763) | merged | [https://nvbugs/6198785][fix] Unify phase-1 CUDA graph cleanup | `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`, `tests/integration/test_lists/waives.txt` |
| 2026-07-27 | [#16805](https://github.com/NVIDIA/TensorRT-LLM/pull/16805) | merged | [None][fix] Fix disaggregated draft token accounting | `tests/unittest/bindings/test_bindings_ut.py`, `tests/unittest/_torch/executor/test_request_utils.py`, `cpp/include/tensorrt_llm/batch_manager/llmRequest.h` |
| 2026-08-05 | [#17225](https://github.com/NVIDIA/TensorRT-LLM/pull/17225) | merged | [None][feat] Add Kimi K3 KDA prefill/MTP decode CuTe DSL kernels and fused attention-residual kernel | `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py`, `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` |
| 2026-08-06 | [#17269](https://github.com/NVIDIA/TensorRT-LLM/pull/17269) | merged | [TRTLLM-14813][feat] Add Kimi K3 (KimiLinear) model | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/configs/kimi_linear.py`, `tensorrt_llm/_torch/configs/__init__.py` |
| 2026-08-07 | [#17125](https://github.com/NVIDIA/TensorRT-LLM/pull/17125) | merged | [None][feat] Default Kimi K2.5 to KV cache manager V2 | `tensorrt_llm/_torch/models/modeling_kimi_k25.py` |
| 2026-08-07 | [#17332](https://github.com/NVIDIA/TensorRT-LLM/pull/17332) | merged | [TRTLLM-14813][test] Port Kimi K3 unit tests and wire GB300 L0 stages | `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` |
| 2026-08-08 | [#17333](https://github.com/NVIDIA/TensorRT-LLM/pull/17333) | merged | [TRTLLM-14813][doc] Add Kimi K3 examples and deployment guide | `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/perf_sweep/perf_sweep.sbatch`, `examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch` |
| 2026-08-08 | [#17327](https://github.com/NVIDIA/TensorRT-LLM/pull/17327) | merged | [TRTLLM-14814][feat] Kimi K3 serving parsers, chat template, and speculative decoding (suffix automaton + DFlash scaffold) | `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py`, `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` |
| 2026-08-08 | [#17334](https://github.com/NVIDIA/TensorRT-LLM/pull/17334) | merged | [TRTLLM-14815][feat] Enable disaggregated serving for Kimi K3 | `tests/integration/defs/kimi_k3_disagg_parity.py`, `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` |
| 2026-08-10 | [#17445](https://github.com/NVIDIA/TensorRT-LLM/pull/17445) | merged | [None][fix] Kimi K3 MLA: pass attn_output to MLA.forward_impl | `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` |
| 2026-08-10 | [#17421](https://github.com/NVIDIA/TensorRT-LLM/pull/17421) | merged | [None][fix] Kimi K3: eager CUDA-graph buffer allocation and prebuilt fused-verify constants | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-10 | [#17456](https://github.com/NVIDIA/TensorRT-LLM/pull/17456) | merged | [None][doc] Note runtime-dependency requirement for the kimi_k3 Slurm container image | `examples/kimi_k3/README.md` |
| 2026-08-10 | [#17446](https://github.com/NVIDIA/TensorRT-LLM/pull/17446) | merged | [TRTLLM-15215][fix] Kimi K3: make the FP8 weight-read master switch opt-in | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-10 | [#17413](https://github.com/NVIDIA/TensorRT-LLM/pull/17413) | merged | [TRTLLM-15177][chore] Kimi K3 post-merge cleanup: config/import/test hygiene + L0 wiring | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-11 | [#17479](https://github.com/NVIDIA/TensorRT-LLM/pull/17479) | merged | [TRTLLM-15264][doc] Kimi K3 disagg: stop recommending a UCX_TLS pin by default | `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` |
| 2026-08-11 | [#17455](https://github.com/NVIDIA/TensorRT-LLM/pull/17455) | merged | [TRTLLM-14814][chore] Add SA speculative-decoding eval config for Kimi K3 | `examples/kimi_k3/eval_extra_llm_options_sa.yaml`, `examples/kimi_k3/README.md` |
| 2026-08-13 | [#17050](https://github.com/NVIDIA/TensorRT-LLM/pull/17050) | merged | [TRTLLM-14704][feat] Support multi-modal part of K3 | `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`, `tensorrt_llm/_torch/configs/kimi_k3.py`, `tensorrt_llm/_torch/models/modeling_kimi_k25.py` |
| 2026-08-14 | [#17311](https://github.com/NVIDIA/TensorRT-LLM/pull/17311) | merged | [None][perf] Fuse Kimi K3 KDA projections | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-15 | [#17624](https://github.com/NVIDIA/TensorRT-LLM/pull/17624) | merged | [TRTLLM-15284][feat] add Kimi K3 SiTU MegaMoE support | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `examples/kimi_k3/eval_extra_llm_options.yaml` |
| 2026-08-17 | [#17480](https://github.com/NVIDIA/TensorRT-LLM/pull/17480) | merged | [TRTLLM-15264][fix] Reject non-Python transceiver routes for Kimi K3 disaggregated serving | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-18 | [#17312](https://github.com/NVIDIA/TensorRT-LLM/pull/17312) | merged | [None][refactor] Refactor Kimi K3 MLP | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-18 | [#17802](https://github.com/NVIDIA/TensorRT-LLM/pull/17802) | merged | [None][test] Add kimi k3 cases for multi-node disagg | `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` |
| 2026-08-18 | [#17861](https://github.com/NVIDIA/TensorRT-LLM/pull/17861) | merged | [None][test] Add back kimi k25 and deepseek v32 cases from qa side | `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` |
| 2026-08-18 | [#17053](https://github.com/NVIDIA/TensorRT-LLM/pull/17053) | merged | [None][perf] Preallocate Kimi attention residual snapshots | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-19 | [#17939](https://github.com/NVIDIA/TensorRT-LLM/pull/17939) | merged | [TRTLLM-15465][feat] Support SA speculative decoding under disaggregated serving for Kimi K3 | `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/gen_config.yaml`, `examples/kimi_k3/README.md` |
| 2026-08-19 | [#17684](https://github.com/NVIDIA/TensorRT-LLM/pull/17684) | merged | [None][feat] Remove padding in Kimi K3 MLA module | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` |
| 2026-08-20 | [#17804](https://github.com/NVIDIA/TensorRT-LLM/pull/17804) | merged | [None][feat] Add Kimi K3 to layer-wise benchmarks | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-21 | [#17865](https://github.com/NVIDIA/TensorRT-LLM/pull/17865) | merged | [None][feat] bring up Kimi K3 NVFP4 with CUTLASS and cuteDSL MegaMoE SiTU | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/models/modeling_kimi_k25.py`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml` |
| 2026-08-21 | [#17846](https://github.com/NVIDIA/TensorRT-LLM/pull/17846) | merged | [TRTLLM-14818][test] Port Kimi K3 DFlash/DSpark eval helpers and KDA FP8 prefill test to main | `examples/kimi_k3/make_synthetic_dflash_drafter.py`, `examples/kimi_k3/measure_dspark_acceptance.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` |
| 2026-08-22 | [#17999](https://github.com/NVIDIA/TensorRT-LLM/pull/17999) | merged | [None][fix] release Kimi K3 checkpoint mappings after loading | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-22 | [#17483](https://github.com/NVIDIA/TensorRT-LLM/pull/17483) | merged | [TRTLLM-15264][test] Kimi K3 disagg review fixups: KDA test geometry, gate docs, example cleanup | `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/ctx_config.yaml`, `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` |
| 2026-08-22 | [#17800](https://github.com/NVIDIA/TensorRT-LLM/pull/17800) | merged | [TRTLLM-15033][feat] Upstream Kimi K3 MLA decode backend selection to main | `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` |
| 2026-08-23 | [#17822](https://github.com/NVIDIA/TensorRT-LLM/pull/17822) | merged | [TRTLLM-15498][refactor] consolidate Kimi KDA production frontend | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modules/kimi_kda/kimi_kda_test_utils.py` |
| 2026-08-25 | [#17980](https://github.com/NVIDIA/TensorRT-LLM/pull/17980) | merged | [TRTLLM-15176][fix] Harden Kimi K3 tool-call parsing | `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` |
| 2026-08-25 | [#18059](https://github.com/NVIDIA/TensorRT-LLM/pull/18059) | merged | [None][fix] Kimi K3: bound MegaMoE expert-weight memory at EP8; drop the MoE TP/EP env overrides | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` |
| 2026-08-25 | [#17796](https://github.com/NVIDIA/TensorRT-LLM/pull/17796) | merged | [None][feat] Kimi K3: KDA-TP + MLA-DCP (helix) wiring | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` |
| 2026-08-27 | [#17741](https://github.com/NVIDIA/TensorRT-LLM/pull/17741) | merged | [None][feat] k3 weight pipeline opt | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` |
| 2026-08-28 | [#18226](https://github.com/NVIDIA/TensorRT-LLM/pull/18226) | merged | [None][fix] Keep CuTe-DSL MLA decode for Kimi K3 H=96 speculative-verify batches | `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` |
| 2026-08-28 | [#17870](https://github.com/NVIDIA/TensorRT-LLM/pull/17870) | merged | [TRTLLM-15498][perf] optimize Kimi KDA prefill convolution data flow | `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py` |
| 2026-08-28 | [#17845](https://github.com/NVIDIA/TensorRT-LLM/pull/17845) | merged | [TRTLLM-14764][feat] trtllm-serve: Kimi K3 API compliance for the Kimi Vendor Verifier (KVV) | `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py`, `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`, `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py` |
| 2026-08-31 | [#18164](https://github.com/NVIDIA/TensorRT-LLM/pull/18164) | merged | [TRTLLM-14705][fix] Kimi K3 B200 enablement: MLA decode dispatch fix, L0 wiring, docs | `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`, `examples/kimi_k3/README.md` |
| 2026-09-01 | [#17921](https://github.com/NVIDIA/TensorRT-LLM/pull/17921) | merged | [TRTLLM-15035][test] Wire Kimi K3 spec-dec and suffix-automaton tests into L0 CI | `tests/integration/defs/test_kimi_k3_specdec.py`, `tests/integration/defs/kimi_k3_disagg_parity.py` |
| 2026-09-02 | [#18251](https://github.com/NVIDIA/TensorRT-LLM/pull/18251) | merged | [TRTLLM-15498][perf] optimize Kimi KDA decode data flow | `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/thop/parallel/test_kda_decode.py` |
| 2026-09-02 | [#18491](https://github.com/NVIDIA/TensorRT-LLM/pull/18491) | merged | [None][test] Add coverage for KimiLinearForCausalLM._setup_helix_mappings and related paths | `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py` |
| 2026-09-02 | [#18064](https://github.com/NVIDIA/TensorRT-LLM/pull/18064) | merged | [TRTLLM-15498][perf] prepare Kimi K3 prefill metadata | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` |
| 2026-09-04 | [#18643](https://github.com/NVIDIA/TensorRT-LLM/pull/18643) | merged | [None][perf] Give the K3 KDA prefill conv input its layout without a repack | `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` |
| 2026-09-04 | [#17816](https://github.com/NVIDIA/TensorRT-LLM/pull/17816) | merged | [None][feat] Kimi k3 Support bcg | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` |
| 2026-09-05 | [#18682](https://github.com/NVIDIA/TensorRT-LLM/pull/18682) | merged | [https://nvbugs/6707518][fix] Fix Kimi K3 spec dec test | `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/integration/defs/test_kimi_k3_specdec.py` |
| 2026-09-07 | [#18699](https://github.com/NVIDIA/TensorRT-LLM/pull/18699) | merged | [None][test] add Kimi K3 feature matrix coverage | `tests/integration/defs/accuracy/test_kimi3.py` |
| 2026-09-07 | [#18294](https://github.com/NVIDIA/TensorRT-LLM/pull/18294) | merged | [None][feat] support Kimi K3 KDA replay with KV cache manager V2 | `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` |
| 2026-09-09 | [#18709](https://github.com/NVIDIA/TensorRT-LLM/pull/18709) | merged | [None][fix] Kimi K3: admit every trtllm-gen SiTu quant format, not just MXFP4 | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` |
| 2026-09-09 | [#18728](https://github.com/NVIDIA/TensorRT-LLM/pull/18728) | merged | [None][fix] share Kimi auxiliary streams | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py`, `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` |
| 2026-09-13 | [#19088](https://github.com/NVIDIA/TensorRT-LLM/pull/19088) | merged | [None][test] Supply auxiliary streams in Kimi K3 NVFP4 regression | `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` |
| 2026-09-15 | [#18461](https://github.com/NVIDIA/TensorRT-LLM/pull/18461) | merged | [TRTLLM-16020][test] Add Kimi K3 GSM8K accuracy tests to GB300 multi-node post-merge CI | `tests/integration/defs/accuracy/test_kimi3.py` |
| 2026-09-16 | [#19191](https://github.com/NVIDIA/TensorRT-LLM/pull/19191) | merged | [TRTLLM-16188][test] Add Kimi K3 short-context perf cases and fix stale disagg note | `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` |
| 2026-09-16 | [#19084](https://github.com/NVIDIA/TensorRT-LLM/pull/19084) | merged | [https://nvbugs/6656598][tests] Deprecate K2 E2E test for K3 | `tests/integration/defs/accuracy/test_kimi3.py` |
| 2026-09-18 | [#19277](https://github.com/NVIDIA/TensorRT-LLM/pull/19277) | merged | [None][fix] Make MoonViT replication helix-aware | `tensorrt_llm/_torch/models/modeling_kimi_k25.py`, `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py` |
| 2026-09-19 | [#19182](https://github.com/NVIDIA/TensorRT-LLM/pull/19182) | merged | [None][feat] Kimi K3 attention-residual RMSNorm fusion + KDA beta-cache alignment | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu`, `tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py` |
| 2026-09-19 | [#19179](https://github.com/NVIDIA/TensorRT-LLM/pull/19179) | merged | [None][refactor] Clean up Kimi checkpoint FP8 attention loading | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`, `tests/unittest/_torch/modeling/kimi_k3_mla_reference.py` |
| 2026-09-21 | [#19040](https://github.com/NVIDIA/TensorRT-LLM/pull/19040) | merged | [TRTLLM-16564][feat] MLA-backboned standalone DSpark drafter (Inferact/Kimi-K3-DSpark) | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py`, `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` |
| 2026-09-21 | [#19003](https://github.com/NVIDIA/TensorRT-LLM/pull/19003) | merged | [None][feat] Kimi K3: unlock the CUTEDSL MoE backend for NVFP4 SiTU | `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` |
| 2026-10-02 | [#19784](https://github.com/NVIDIA/TensorRT-LLM/pull/19784) | merged | [https://nvbugs/6783973][doc] Stop listing supported SA speculation as a Kimi K3 limitation | `examples/kimi_k3/README.md` |

## Per-PR Diff Audit Cards

### PR #9711 - Deployment Guide for Kimi K2 Thinking on TensorRT LLM - Blackwell

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9711
- Status/date: merged / 2025-12-05
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +309/-0, 534 cached patch lines.
- Motivation: provide an official Blackwell/GB200 deployment guide for Kimi K2 Thinking NVFP4.
- Key implementation: documents Docker, `trtllm-serve`, 8-way EP/attention DP, SLURM wide EP, and disaggregated serving.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+trtllm-serve nvidia/Kimi-K2-Thinking-NVFP4 \
+--extra_llm_api_options
```

- Reviewed files: deployment guide markdown
- Risk and verification: record Blackwell/GB200 and disaggregation assumptions when using this as competitor evidence.

### PR #9830 - Support tool parser for Kimi K2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/9830
- Status/date: merged / 2025-12-12
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 5 files, +374/-1, 528 cached patch lines.
- Motivation: Kimi K2 OpenAI-compatible serving needs correct tool-call parsing for agentic workloads.
- Key implementation: adds a Kimi K2 tool parser and wires it into the OpenAI server postprocess and parser factory.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+from .kimi_k2_tool_parser import KimiK2ToolParser
+class KimiK2ToolParser(BaseToolParser):
+        "kimi_k2": KimiK2ToolParser,
```

- Reviewed files: OpenAI server, postprocess handlers, `kimi_k2_tool_parser.py`, factory, tests
- Risk and verification: agentic correctness includes parser behavior, not just speed.

### PR #11645 - [None][chore] Moving kimi-k2-thinking deployment guide configs to config files.

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11645
- Status/date: merged / 2026-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md`, `examples/configs/curated/kimi-k2-thinking.yaml`, `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml`; associated commits `730797461ee5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +168/-129, 336 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/configs/curated/kimi-k2-thinking.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15; `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` modified +50/-129 (179 lines); hunks: -30,37 +30,42 @@ make -C docker release_build IMAGE_TAG=kimi-k2-thinking-local; -84,120 +89,36 @@ When the `Status: 200` code is returned, the server is read...; `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98.
- Code diff details:
  - `examples/configs/curated/kimi-k2-thinking.yaml` added +15/-0 (15 lines); hunks: -0,0 +1,15
  - `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` modified +50/-129 (179 lines); hunks: -30,37 +30,42 @@ make -C docker release_build IMAGE_TAG=kimi-k2-thinking-local; -84,120 +89,36 @@ When the `Status: 200` code is returned, the server is read...
  - `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml` added +98/-0 (98 lines); hunks: -0,0 +1,98
- Key code excerpts:

```diff
diff -- examples/configs/curated/kimi-k2-thinking.yaml
@@ -0,0 +1,15 @@
+max_batch_size: 128
+max_num_tokens: 8448
+max_seq_len: 8212
+tensor_parallel_size: 8
+moe_expert_parallel_size: 8
+enable_attention_dp: true
diff -- docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md
@@ -30,37 +30,42 @@ make -C docker release_build IMAGE_TAG=kimi-k2-thinking-local
-### Launch the TensorRT LLM Server
+### Recommended Performance Settings
+We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory
+'''shell
+TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
+EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/kimi-k2-thinking.yaml
diff -- examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml
@@ -0,0 +1,98 @@
```

- Extracted files (not manually reviewed):
  - docs: `examples/configs/curated/kimi-k2-thinking.yaml` added +15/-0; `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md` modified +50/-129; `examples/wide_ep/slurm_scripts/kimi-k2-thinking.yaml` added +98/-0
- Risk and verification: This is mostly docs/examples in `docs/source/deployment-guide/deployment-guide-for-kimi-k2-thinking-on-trtllm.md`, `docs/source/deployment-guide/index.rst`, `examples/configs/curated/kimi-k2-thinking.yaml`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #11777 - Add Kimi-K2.5 text model support (NVFP4)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11777
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +96/-0, 532 cached patch lines.
- Motivation: support Kimi-K2.5 text NVFP4 in the PyTorch backend.
- Key implementation: adapts the DeepSeekV3-style runtime and adds accuracy refs/tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+MODEL_NAME = "moonshotai/Kimi-K2.5"
+quant_algo: NVFP4
```

- Reviewed files: `modeling_deepseekv3.py`, accuracy refs/tests
- Risk and verification: separate text-only Kimi K2.5 from multimodal Kimi paths.

### PR #11780 - AutoDeploy onboarding agent + Kimi K2.5 AD modeling code

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/11780
- Status/date: merged / 2026-03-05
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 9 files, +2190/-9, 2,807 cached patch lines.
- Motivation: add AutoDeploy modeling code for Kimi K2.5.
- Key implementation: adds `modeling_kimi_k2.py`, registry config, MLA custom ops, and AutoDeploy tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+model_factory: KimiK2ForCausalLM
+flashinfer_mla
```

- Reviewed files: agent scaffold, `kimi_k2.yaml`, MLA ops, `modeling_kimi_k2.py`, AD tests
- Risk and verification: competitor path may be AutoDeploy rather than the plain PyTorch wrapper.

### PR #13801 - Add reasoning parser for Kimi-K2.5 and enable auto flow

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/13801
- Status/date: merged / 2026-05-11
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +5/-1, 47 cached patch lines.
- Motivation: auto-select the right reasoning parser for Kimi-K2.5 thinking outputs.
- Key implementation: adds `kimi_k2/kimi_k25` auto-detect hints and registers `kimi_k25` with `reasoning_at_start=True`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+"kimi_k25": "kimi_k25",
+@register_reasoning_parser("kimi_k25", reasoning_at_start=True)
```

- Reviewed files: `commands/serve.py`, `reasoning_parser.py`
- Risk and verification: parser selection affects eval scores.

### PR #12788 - Add Kimi K2.5 multimodal vision support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/12788
- Status/date: merged / 2026-05-14
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 12 files, +2912/-64, 3,536 cached patch lines.
- Motivation: enable text/image/video Kimi K2.5 multimodal serving.
- Key implementation: adds `KimiK25ForConditionalGeneration`, vision model, input processor, placeholders, multimodal eval, and tests.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+@register_auto_model("KimiK25ForConditionalGeneration")
+class KimiK25ForConditionalGeneration(PreTrainedModel):
+    "video_placeholder": "<|kimi_k25_video_placeholder|>",
```

- Reviewed files: `modeling_kimi_k25.py`, `modeling_deepseekv3.py`, eval wrappers, multimodal tests
- Risk and verification: profile vision encoder, placeholder expansion, hashing fallback, and text decode as separate stages.

### PR #14379 - Fix Kimi_k25 with spec dec

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14379
- Status/date: merged / 2026-05-22
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +53/-35, 187 cached patch lines.
- Motivation: speculative decoding missed Kimi K2.5 multimodal params and `lm_head` delegation.
- Key implementation: threads `multimodal_params` through `forward` and adds an `lm_head` proxy.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+multimodal_params: Optional[List[MultimodalParams]] = None
+def lm_head(self): return self.llm.lm_head
```

- Reviewed files: `modeling_kimi_k25.py`
- Risk and verification: spec-dec comparisons need to verify context-only multimodal handling.

### PR #14392 - [https://nvbugs/6182617][fix] Restore K2.5 multimodal dep8 accuracy test on transformers 5.5.x

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14392
- Status/date: merged / 2026-05-25
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k25.py`; associated commits `546a5b091256`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +42/-3, 82 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +36/-0 (36 lines); hunks: -1052,6 +1052,13 @@ def __init__(; -1168,6 +1175,35 @@ def get_num_tokens_per_video(self, *, video: List, **kwar...; symbols: __init__, config, get_num_tokens_per_video, _ensure_k25_slow_tokenizer, touching `__init__, config, get_num_tokens_per_video`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +36/-0 (36 lines); hunks: -1052,6 +1052,13 @@ def __init__(; -1168,6 +1175,35 @@ def get_num_tokens_per_video(self, *, video: List, **kwar...; symbols: __init__, config, get_num_tokens_per_video, _ensure_k25_slow_tokenizer
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -1052,6 +1052,13 @@ def __init__(
+        # transformers 5.5.x ``AutoTokenizer`` may route K2.5 to the Rust
+        # fast backend, which BPE-splits ``<|media_pad|>`` / ``<|im_user|>``
+        # / etc. instead of mapping them to their canonical IDs. Force the
+        # K2.5 slow ``TikTokenTokenizer`` for deterministic tokenization.
+        # See NVBug 6182617.
+        self._ensure_k25_slow_tokenizer()
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +36/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_llm_api_pytorch_multimodal.py`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14741 - [https://nvbugs/6227203][fix] Remove redundant TikTokenTokenizer shim from KimiK25InputProcessor

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14741
- Status/date: merged / 2026-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k25.py`; associated commits `28845ddf99a3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +0/-85, 130 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +0/-84 (84 lines); hunks: -291,23 +291,6 @@ def _frames_to_chunks(; -1069,16 +1052,6 @@ def __init__(; symbols: _frames_to_chunks, __init__, config, get_num_tokens_per_video, touching `_frames_to_chunks, __init__, config`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +0/-84 (84 lines); hunks: -291,23 +291,6 @@ def _frames_to_chunks(; -1069,16 +1052,6 @@ def __init__(; symbols: _frames_to_chunks, __init__, config, get_num_tokens_per_video
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -291,23 +291,6 @@ def _frames_to_chunks(
-# K2.5 special token markers that the transformers 5.5.x Rust fast tokenizer
-# BPE-splits instead of mapping to canonical IDs. When any of these appear in
-# a prompt, we must route tokenization through the slow ``TikTokenTokenizer``.
-# Pure text (no markers and no multimodal data) keeps the fast tokenizer.
-# See NVBug 6182617 (correctness) and NVBug 6248987 (perf).
-_K25_SPECIAL_TOKEN_MARKERS = (
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +0/-84
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14960 - [None][test] Update K2.5 andGLM-5 into CI Perf Test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14960
- Status/date: merged / 2026-06-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`; associated commits `835fd6115bf7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 48 files, +2666/-41, 2937 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -60,6 +60,9 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -60,6 +60,9 @@ worker_config:; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -71,6 +71,9 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -71,6 +71,9 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -60,6 +60,9 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -60,6 +60,9 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -71,6 +71,9 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0 (3 lines); hunks: -71,6 +71,9 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml
@@ -60,6 +60,9 @@ worker_config:
+      load_balancer:
+        num_slots: 416
+        layer_updates_per_iter: 1
diff -- tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml
@@ -60,6 +60,9 @@ worker_config:
+      load_balancer:
+        num_slots: 416
+        layer_updates_per_iter: 1
diff -- tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml
@@ -71,6 +71,9 @@ worker_config:
+      load_balancer:
+        num_slots: 416
+        layer_updates_per_iter: 1
diff -- tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml
@@ -71,6 +71,9 @@ worker_config:
+      load_balancer:
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0; `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` renamed +3/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/integration/test_lists/test-db/l0_b200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_gpus_perf_sanity.yml`, `tests/integration/test_lists/test-db/l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node8_gpu32.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15233 - Fix embedding vocab mask for rejection sampling in Kimi-K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15233
- Status/date: merged / 2026-06-17
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +15/-8, 129 cached patch lines.
- Motivation: FlashInfer rejection sampling can pad rejected tokens with non-vocab values.
- Key implementation: masks/clamps input before `F.embedding` in `pre_comm_embedding_ops`.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+# flashinfer's rejection kernel pads non-accepted tokens
+        input_, input_mask = get_masked_input_and_mask(
+            input_,
+            0,
+            weight.shape[0],
+        )
```

- Reviewed files: `embedding.py`
- Risk and verification: correctness risk sits in embedding preprocessing, not a visible hot kernel.

### PR #15443 - Un-waive K2.5 Thinking FP4 disagg-NIXL tests

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15443
- Status/date: merged / 2026-06-23
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 2 files, +2/-3, 86 cached patch lines.
- Motivation: Kimi K2.5 Thinking FP4 disaggregated NIXL lanes became stable enough to un-waive.
- Key implementation: removes Kimi NIXL skips and raises KV transfer timeout in perf-sanity YAML.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+kv_transfer_timeout_ms: 600000
```

- Reviewed files: `waives.txt`, Kimi NIXL perf-sanity YAML
- Risk and verification: disaggregated NIXL is a separate benchmark bucket.

### PR #15180 - Add necessary methods for guided decoding in Kimi K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15180
- Status/date: merged / 2026-06-25
- Trace source: `git log --name-only -- <model-files>` plus GitHub Pull Request files API.
- Diff scope read: 1 file, +3/-0, 28 cached patch lines.
- Motivation: Kimi K2.5 wrapper missed guided decoding delegation methods.
- Key implementation: proxies `set_guided_decoder` to the inner LLM.
- Code diff details: See the diff scope line above and the excerpt below for the audited file-level changes.
- Key code excerpts:

```diff
+def set_guided_decoder(self, *args, **kwargs):
+    return self.llm.set_guided_decoder(*args, **kwargs)
```

- Reviewed files: `modeling_kimi_k25.py`
- Risk and verification: guided decoding changes decode control flow; record whether it is enabled.


### PR #14848 - RMSNorm NVFP4 quant fusion for DeepSeek-V3.2 / Kimi-K2.5

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/14848
- Status/date: merged / 2026-07-15
- Trace source: `git log --name-only -- <model-files>` plus the final upstream commit and PR body.
- Diff scope read: full 2,487-line diff, 16 files, +1993/-101.
- Motivation: static-NVFP4 Kimi-K2.5 executed RMSNorm and activation quantization as separate kernels before each compatible linear, materializing the normalized tensor and paying an extra launch.
- Key implementation: adds Blackwell-only C++/CUDA operators for fused optional residual-add + RMSNorm + NVFP4 quantization, supports packed and row-strided input, returns the unquantized norm when another consumer needs it, and routes eligible RMSNorm-to-linear edges through the fused result.
- Code diff details: `rmsNormFp4QuantKernel` performs the reduction and emits packed E2M1 values plus E4M3 block scales; Python dispatch keeps unsupported architectures and shapes on the existing path.
- Key code excerpts:

```diff
+// Fused (optional residual-add +) RMSNorm + NVFP4 input-quantize.
+__global__ void rmsNormFp4QuantKernel(RmsNormFp4QuantParams params)
+{
+    float const denom = rsqrtf(acc / params.hidden_size + params.eps);
+    uint32_t const quant_val = cvt_warp_fp16_to_fp4<T, kSfVecSize, false>(
+        pv, sf_scale, sf_out_ptr);
+}
```

- Reviewed files: runtime: `cpp/tensorrt_llm/kernels/rmsNormFp4QuantKernels.{cu,h}`, `cpp/tensorrt_llm/thop/rmsNormFp4Quant.cpp`, `tensorrt_llm/_torch/modules/{rms_norm,linear,mla}.py`, `modeling_deepseekv3.py`; tests: `test_fused_rmsnorm_fp4_quantize.py`, `test_fp4_num_tokens_slice.py`, B200 test database.
- Risk and verification: the fused FP4 epilogue is restricted to SM10.x; validation compares packed FP4 values and unswizzled scale factors, checks strided inputs and no input mutation, and separates acceptable RMSNorm rounding drift from bit-exact re-quantization of the returned norm.

### PR #15966 - [TRTLLM-13639][perf] Migrate Kimi perf-sanity tests to Transceiver v2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/15966
- Status/date: merged / 2026-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` and 12 files; associated commits `4188dbe33076`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +32/-0, 193 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -63,6 +63,8 @@ worker_config:; -88,5 +90,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -86,5 +88,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -86,5 +88,7 @@ worker_config:; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -92,6 +94,8 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -63,6 +63,8 @@ worker_config:; -88,5 +90,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -86,5 +88,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -86,5 +88,7 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -61,6 +61,8 @@ worker_config:; -92,6 +94,8 @@ worker_config:
  - `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -63,6 +63,7 @@ worker_config:; -88,5 +89,6 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml
@@ -63,6 +63,8 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 384
@@ -88,5 +90,7 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 384
diff -- tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml
@@ -61,6 +61,8 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 384
@@ -86,5 +88,7 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 384
diff -- tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml
@@ -61,6 +61,8 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 384
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf-sanity/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con2048_ctx1_dep4_gen1_dep32_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4096_ctx1_dep4_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf-sanity/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16482 - [TRTLLM-13639][test] Migrate Kimi dis-agg tests to Transceiver v2 and trim tests

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16482
- Status/date: merged / 2026-07-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k25.py`, `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py`; associated commits `93909240665c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +32/-59, 148 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +16/-1 (17 lines); hunks: -34,7 +34,7; -1517,6 +1517,21 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):; symbols: KimiK25ForConditionalGeneration, get_preferred_transceiver_runtime, __init__, touching `KimiK25ForConditionalGeneration, get_preferred_transceiver_runtime, __init__`; `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` modified +6/-0 (6 lines); hunks: -605,6 +605,12 @@ def test_auto_model_registered(self):; symbols: test_auto_model_registered, test_prefers_python_transceiver, touching `test_auto_model_registered, test_prefers_python_transceiver`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +16/-1 (17 lines); hunks: -34,7 +34,7; -1517,6 +1517,21 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):; symbols: KimiK25ForConditionalGeneration, get_preferred_transceiver_runtime, __init__
  - `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` modified +6/-0 (6 lines); hunks: -605,6 +605,12 @@ def test_auto_model_registered(self):; symbols: test_auto_model_registered, test_prefers_python_transceiver
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -34,7 +34,7 @@
-from typing import Any, Dict, List, Optional, Tuple, Union
+from typing import Any, Dict, List, Literal, Optional, Tuple, Union
@@ -1517,6 +1517,21 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):
+    @classmethod
+    def get_preferred_transceiver_runtime(
+        cls,
diff -- tests/unittest/_torch/modeling/test_modeling_kimi_k25.py
@@ -605,6 +605,12 @@ def test_auto_model_registered(self):
+    def test_prefers_python_transceiver(self):
+        """Kimi-K2.5 defaults to the Python KV-cache transceiver in disagg."""
+        self.assertEqual(
+            KimiK25ForConditionalGeneration.get_preferred_transceiver_runtime(), "PYTHON"
+        )
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +16/-1
  - tests: `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py` modified +6/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_disaggregated_serving.py`, `tests/integration/test_lists/qa/llm_function_core.txt`, `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16763 - Unify phase-1 CUDA graph cleanup

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16763
- Status/date: merged / 2026-07-27
- Trace source: final upstream commit
  `6046f34e1ac8fc20c2c5553d00a98eb15a172555`; complete two-file diff read
  locally.
- Diff scope read: 2 files, +5/-8.
- Motivation: KV-capacity estimation already shuts down the phase-1 executor
  and releases its CUDA graphs; a second explicit graph release duplicated
  ownership and complicated final KV-cache allocation.
- Key implementation: rely on `configure_kv_cache_capacity` for graph/resource
  teardown and clear only profiling attention metadata before rebuilding the
  final KV managers.
- Code diff details: removes the second `_release_cuda_graphs()` call from
  `py_executor_creator.py`, keeps `eng.attn_metadata = None`, and unwaives the
  two configurations that exercise the corrected ownership path.
- Key code excerpts:

```diff
-            if eng.attn_metadata is not None:
-                if llm_args.cuda_graph_config is not None:
-                    eng._release_cuda_graphs()
+            if eng is not None:
         eng.attn_metadata = None
```

- Reviewed files: `py_executor_creator.py` and the two newly unwaived
  DeepSeek-V3Lite integration cases.
- Risk and verification: compare startup memory before/after capacity
  estimation and ensure graph resources are released exactly once.

### PR #16805 - Fix disaggregated draft-token accounting

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/16805
- Status/date: merged / 2026-07-27
- Trace source: merge commit
  `93924532ff65e6dce000c1dcee604e585386781b`; full six-commit increment and
  five-file PR diff read locally.
- Diff scope read: 5 files, +164/-4.
- Motivation: generation-only requests received first-generation and draft
  tokens through `ContextPhaseParams`, but the decode request did not adopt the
  draft tokens and sequence-length setup counted only the first token.
- Key implementation: `GenericLlmRequest` adopts handoff draft tokens at
  construction or late context assignment; a shared helper counts first-gen
  plus draft tokens, and decoder sequence-length setup uses that total.
- Code diff details: `llmRequest.h` adds draft-token adoption and the shared
  count helper; `createNewDecoderRequests.cpp` replaces first-token-only
  arithmetic; C++ and Python tests cover constructor and late-assignment paths.
- Key code excerpts:

```diff
+        adoptContextPhaseDraftTokens();
+        auto numTokens = static_cast<SizeType32>(
+            contextPhaseParams.getFirstGenTokens().size());
+        numTokens += static_cast<SizeType32>(draftTokens->size());
```

- Reviewed files: `llmRequest.h`, `createNewDecoderRequests.cpp`, C++ request
  tests, Python executor-request tests, and binding tests.
- Risk and verification: validate context/decode handoff with and without draft
  tokens, late context assignment, and downstream sequence lengths; this is a
  correctness fix, not throughput evidence.

### PR #17225 - [None][feat] Add Kimi K3 KDA prefill/MTP decode CuTe DSL kernels and fused attention-residual kernel

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17225
- Status/date: merged / 2026-08-05
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/kimiK3AttnRes/CMakeLists.txt`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h`, `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py`, `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` and 12 files; associated commits `3e16ef24091b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +12059/-0, 9819 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py` added +2305/-0 (2305 lines); `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py` added +2278/-0 (2278 lines); hunks: -0,0 +1,2278; symbols: k1_internal_barrier, mma_tf32_m16n8k8, read_clock, fast_rcp, touching `k1_internal_barrier, mma_tf32_m16n8k8, read_clock`; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` added +1557/-0 (1557 lines); hunks: -0,0 +1,1557; `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py` added +1496/-0 (1496 lines); hunks: -0,0 +1,1496; symbols: _DummyExperimentalDSL, mma_tf32_m16n8k8, inv_internal_barrier, _invert_diag, touching `_DummyExperimentalDSL, mma_tf32_m16n8k8, inv_internal_barrier`.
- Code diff details:
  - `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py` added +2305/-0 (2305 lines)
  - `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py` added +2278/-0 (2278 lines); hunks: -0,0 +1,2278; symbols: k1_internal_barrier, mma_tf32_m16n8k8, read_clock, fast_rcp
  - `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` added +1557/-0 (1557 lines); hunks: -0,0 +1,1557
  - `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py` added +1496/-0 (1496 lines); hunks: -0,0 +1,1496; symbols: _DummyExperimentalDSL, mma_tf32_m16n8k8, inv_internal_barrier, _invert_diag
  - `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py` added +1352/-0 (1352 lines); hunks: -0,0 +1,1352; symbols: _ct, _current_cu_stream, _get_eqlen_dummies, _cute_int_type
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py
@@ -0,0 +1,2278 @@
+# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu
@@ -0,0 +1,1557 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py
@@ -0,0 +1,1496 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py` added +2305/-0; `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py` added +2278/-0; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` added +1557/-0; `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py` added +1496/-0; `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_custom_ops.py` added +1352/-0; `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/akk_inverse.py` added +1075/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/unittest/_torch/cute_dsl/test_kimi_k3_kda_ptx_patch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17269 - [TRTLLM-14813][feat] Add Kimi K3 (KimiLinear) model

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17269
- Status/date: merged / 2026-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/configs/kimi_linear.py`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_attn_res/__init__.py`, `tensorrt_llm/_torch/modules/kimi_k3_attn_res/_attn_res_kernels.py`, `tensorrt_llm/_torch/modules/kimi_k3_attn_res/kimi_k3_attn_res.py` and 15 files; associated commits `36922c7283ed`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 49 files, +10705/-60, 8877 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` added +2640/-0 (2640 lines); `tensorrt_llm/_torch/configs/kimi_linear.py` added +175/-0 (175 lines); hunks: -0,0 +1,175; symbols: KimiLinearConfig, __init__, attribute, is_mla, touching `KimiLinearConfig, __init__, attribute`; `tensorrt_llm/_torch/configs/__init__.py` modified +8/-0 (8 lines); hunks: -23,6 +23,7; -54,6 +55,12 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers, touching `_register_custom_configs_with_transformers`; `tensorrt_llm/_torch/models/__init__.py` modified +2/-0 (2 lines); hunks: -30,6 +30,7; -89,6 +90,7.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` added +2640/-0 (2640 lines)
  - `tensorrt_llm/_torch/configs/kimi_linear.py` added +175/-0 (175 lines); hunks: -0,0 +1,175; symbols: KimiLinearConfig, __init__, attribute, is_mla
  - `tensorrt_llm/_torch/configs/__init__.py` modified +8/-0 (8 lines); hunks: -23,6 +23,7; -54,6 +55,12 @@ def _register_custom_configs_with_transformers() -> None:; symbols: _register_custom_configs_with_transformers
  - `tensorrt_llm/_torch/models/__init__.py` modified +2/-0 (2 lines); hunks: -30,6 +30,7; -89,6 +90,7
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` added +723/-0 (723 lines); hunks: -0,0 +1,723; symbols: _meta_safe_cast_dtype, _cast, _MetaSafeFusedRMSNormGated, reset_parameters
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/configs/kimi_linear.py
@@ -0,0 +1,175 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""In-tree config for Kimi K3 ("kimi_linear") text checkpoints.
+Mirrors the checkpoint-shipped ``configuration_kimi_k3.KimiLinearConfig`` so
+TRT-LLM can parse Kimi K3 checkpoints without ``trust_remote_code`` for the
+config. The top-level Kimi K3 checkpoints use a composite VLM config
diff -- tensorrt_llm/_torch/configs/__init__.py
@@ -23,6 +23,7 @@
+from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
@@ -54,6 +55,12 @@ def _register_custom_configs_with_transformers() -> None:
+        # Kimi K3 text config ("kimi_linear"). The composite "kimi_k3"
+        # model_type is flattened to the text config by
+        # pyexecutor.config_utils.load_pretrained_config; registering the
+        # text config here lets AutoConfig / AutoTokenizer resolve
diff -- tensorrt_llm/_torch/models/__init__.py
@@ -30,6 +30,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` added +2640/-0; `tensorrt_llm/_torch/configs/kimi_linear.py` added +175/-0; `tensorrt_llm/_torch/configs/__init__.py` modified +8/-0; `tensorrt_llm/_torch/models/__init__.py` modified +2/-0; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` added +723/-0; `tensorrt_llm/_torch/modules/kimi_k3_attn_res/kimi_k3_attn_res.py` added +403/-0
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_state_parity.py` added +397/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kda_mtp_decode_cute_parity.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fused_verify_parity.py`, `tests/unittest/_torch/modeling/test_kimi_kda_verify_parity.py`, `tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_op.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17125 - [None][feat] Default Kimi K2.5 to KV cache manager V2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17125
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k25.py`; associated commits `a9be9ecf23f3`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +5/-0, 12 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +5/-0 (5 lines); hunks: -1517,6 +1517,11 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):; symbols: KimiK25ForConditionalGeneration, get_model_defaults, get_preferred_transceiver_runtime, touching `KimiK25ForConditionalGeneration, get_model_defaults, get_preferred_transceiver_runtime`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +5/-0 (5 lines); hunks: -1517,6 +1517,11 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):; symbols: KimiK25ForConditionalGeneration, get_model_defaults, get_preferred_transceiver_runtime
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -1517,6 +1517,11 @@ class KimiK25ForConditionalGeneration(PreTrainedModel):
+    @classmethod
+    def get_model_defaults(cls, llm_args: Any) -> dict:
+        """Use the C++ KV cache manager V2 by default."""
+        return {"kv_cache_config": {"use_kv_cache_manager_v2": True}}
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17332 - [TRTLLM-14813][test] Port Kimi K3 unit tests and wire GB300 L0 stages

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17332
- Status/date: merged / 2026-08-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`; associated commits `2c96f9424a6c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +1130/-3, 1186 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` added +401/-0 (401 lines); hunks: -0,0 +1,401; symbols: _has_supported_gpu, _make_attention_pair, _make_cache, _assert_close, touching `_has_supported_gpu, _make_attention_pair, _make_cache`.
- Code diff details:
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` added +401/-0 (401 lines); hunks: -0,0 +1,401; symbols: _has_supported_gpu, _make_attention_pair, _make_cache, _assert_close
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py
@@ -0,0 +1,401 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Parity tests for the optimized Kimi K3 KDA decode op."""
+import copy
+import pytest
+import torch
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` added +401/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_gb300_multi_gpus.yml`, `tests/unittest/_torch/executor/test_mamba_cache_manager.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`, `tests/unittest/_torch/modules/moe/test_kimi_k3_mlp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17333 - [TRTLLM-14813][doc] Add Kimi K3 examples and deployment guide

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17333
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/README.md`, `examples/kimi_k3/eval_extra_llm_options.yaml`, `examples/kimi_k3/eval_extra_llm_options_reuse.yaml`, `examples/kimi_k3/perf_sweep/acc_sweep.sbatch` and 11 files; associated commits `d7053e55a7f7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1781/-0, 1806 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` added +361/-0 (361 lines); hunks: -0,0 +1,361; `examples/kimi_k3/perf_sweep/perf_sweep.sbatch` added +264/-0 (264 lines); hunks: -0,0 +1,264; `examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch` added +233/-0 (233 lines); hunks: -0,0 +1,233; `examples/kimi_k3/README.md` added +196/-0 (196 lines); hunks: -0,0 +1,196.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` added +361/-0 (361 lines); hunks: -0,0 +1,361
  - `examples/kimi_k3/perf_sweep/perf_sweep.sbatch` added +264/-0 (264 lines); hunks: -0,0 +1,264
  - `examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch` added +233/-0 (233 lines); hunks: -0,0 +1,233
  - `examples/kimi_k3/README.md` added +196/-0 (196 lines); hunks: -0,0 +1,196
  - `examples/kimi_k3/perf_sweep/acc_sweep.sbatch` added +177/-0 (177 lines); hunks: -0,0 +1,177
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md
@@ -0,0 +1,361 @@
+# Deployment Guide for Kimi K3 on TensorRT LLM - Blackwell
+## Introduction
+This deployment guide provides step-by-step instructions for running the Kimi K3 model using TensorRT LLM on NVIDIA Blackwell GPUs. The deployment configurations and results in th
+Kimi K3 is a large hybrid Mixture-of-Experts (MoE) model. Its 93 decoder layers interleave two attention families: most layers use Kimi Delta Attention (KDA), a linear-attention m
+This guide uses Slurm and the `trtllm-llmapi-launch` multi-node launcher. The configuration walkthrough focuses on the **DEP16** high-throughput deployment: attention runs data-pa
+## Prerequisites
diff -- examples/kimi_k3/perf_sweep/perf_sweep.sbatch
@@ -0,0 +1,264 @@
+#!/bin/bash
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Kimi K3 serving perf sweep (8K-in/1K-out): one trtllm-serve launch
+# per job, all requested concurrencies measured against it.
diff -- examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch
@@ -0,0 +1,233 @@
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` added +361/-0; `examples/kimi_k3/perf_sweep/perf_sweep.sbatch` added +264/-0; `examples/kimi_k3/run_serving_benchmark_kimi_k3.sbatch` added +233/-0; `examples/kimi_k3/README.md` added +196/-0; `examples/kimi_k3/perf_sweep/acc_sweep.sbatch` added +177/-0; `examples/kimi_k3/quick_start_kimi_k3.sbatch` added +116/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/attention/sparse/dsa/test_req_idx_per_token.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17327 - [TRTLLM-14814][feat] Kimi K3 serving parsers, chat template, and speculative decoding (suffix automaton + DFlash scaffold)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17327
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/README.md`, `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py`, `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/integration/defs/test_kimi_k3_specdec.py` and 7 files; associated commits `937bacc2ab87`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 24 files, +4066/-29, 4431 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _unescape_attr, _parse_attrs, KimiK3ToolParser, __init__, touching `_unescape_attr, _parse_attrs, KimiK3ToolParser`; `tests/integration/defs/kimi_k3_sa_harness.py` added +558/-0 (558 lines); hunks: -0,0 +1,558; symbols: _truncated_checkpoint, _graphs_enabled, _build_llm, _parity_prompts, touching `_truncated_checkpoint, _graphs_enabled, _build_llm`; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` added +494/-0 (494 lines); hunks: -0,0 +1,494; symbols: _RefVanillaMarkov, __init__, compute_step_bias, sample_block_tokens, touching `_RefVanillaMarkov, __init__, compute_step_bias`; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: _load_generator, test_even_spacing_matches_k27_reference, test_tensor_plan_matches_k27_schema, test_even_spacing_matches_real_k3_drafter, touching `_load_generator, test_even_spacing_matches_k27_reference, test_tensor_plan_matches_k27_schema`.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` added +201/-0 (201 lines); hunks: -0,0 +1,201; symbols: _unescape_attr, _parse_attrs, KimiK3ToolParser, __init__
  - `tests/integration/defs/kimi_k3_sa_harness.py` added +558/-0 (558 lines); hunks: -0,0 +1,558; symbols: _truncated_checkpoint, _graphs_enabled, _build_llm, _parity_prompts
  - `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` added +494/-0 (494 lines); hunks: -0,0 +1,494; symbols: _RefVanillaMarkov, __init__, compute_step_bias, sample_block_tokens
  - `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py` added +344/-0 (344 lines); hunks: -0,0 +1,344; symbols: _load_generator, test_even_spacing_matches_k27_reference, test_tensor_plan_matches_k27_schema, test_even_spacing_matches_real_k3_drafter
  - `tests/integration/defs/test_kimi_k3_specdec.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: _find_checkpoint, test_kimi_k3_sa_specdec_logits_parity, test_kimi_k3_disagg_parity_selftest
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py
@@ -0,0 +1,201 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Tool-call parser for the Kimi K3 XTML output format.
+K3 emits tool calls as an XTML tag stream built from the special tokens
+``<|open|>`` / ``<|close|>`` / ``<|sep|>`` with plain-text tag headers
+(authoritative rendering: the checkpoint's ``encoding_k3.py``)::
diff -- tests/integration/defs/kimi_k3_sa_harness.py
@@ -0,0 +1,558 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Kimi K3 SA spec-dec / sanity test harness via the LLM API.
+Runs greedy prompts through the standard TRT-LLM PyTorch backend and
+asserts the outputs contain expected content, optionally with SA
+(suffix-automaton) speculative decoding and parity checking against a
diff -- tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py
@@ -0,0 +1,494 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` added +201/-0
  - tests: `tests/integration/defs/kimi_k3_sa_harness.py` added +558/-0; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` added +494/-0; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py` added +344/-0; `tests/integration/defs/test_kimi_k3_specdec.py` added +95/-0
  - docs: `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +0/-2; `examples/kimi_k3/README.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/integration/defs/test_kimi_k3_specdec.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_dflash_accept_stats.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17334 - [TRTLLM-14815][feat] Enable disaggregated serving for Kimi K3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17334
- Status/date: merged / 2026-08-08
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/README.md`, `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml`, `examples/kimi_k3/disagg/ctx_config.yaml`, `examples/kimi_k3/disagg/disagg_proxy_config.yaml` and 7 files; associated commits `4d02d80eaa90`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 25 files, +2666/-85, 3164 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/kimi_k3_disagg_parity.py` added +792/-0 (792 lines); hunks: -0,0 +1,792; symbols: _http_json, _served_model, Endpoint, __init__, touching `_http_json, _served_model, Endpoint`; `examples/kimi_k3/disagg/README.md` added +165/-0 (165 lines); hunks: -0,0 +1,165; `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` added +136/-0 (136 lines); hunks: -0,0 +1,136; `examples/kimi_k3/disagg/ctx_config.yaml` added +67/-0 (67 lines); hunks: -0,0 +1,67.
- Code diff details:
  - `tests/integration/defs/kimi_k3_disagg_parity.py` added +792/-0 (792 lines); hunks: -0,0 +1,792; symbols: _http_json, _served_model, Endpoint, __init__
  - `examples/kimi_k3/disagg/README.md` added +165/-0 (165 lines); hunks: -0,0 +1,165
  - `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` added +136/-0 (136 lines); hunks: -0,0 +1,136
  - `examples/kimi_k3/disagg/ctx_config.yaml` added +67/-0 (67 lines); hunks: -0,0 +1,67
  - `examples/kimi_k3/disagg/gen_config_no_sa.yaml` added +50/-0 (50 lines); hunks: -0,0 +1,50
- Key code excerpts:

```diff
diff -- tests/integration/defs/kimi_k3_disagg_parity.py
@@ -0,0 +1,792 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Kimi K3 two-endpoint parity harness (aggregated vs disaggregated).
+Compares a REFERENCE deployment against a CANDIDATE deployment through
+their OpenAI-compatible ``/v1/completions`` endpoints and produces a
+parity report:
diff -- examples/kimi_k3/disagg/README.md
@@ -0,0 +1,165 @@
+# Kimi K3 disaggregated serving (ctx/gen split)
+Configuration pair + deployment wiring for running Kimi K3 with separate
+context (prefill) and generation (decode) servers. Status: **validated
+end-to-end on hardware** (GB300 NVL72, 1 ctx + 1 gen, DEP16 both sides,
+GSM8K accuracy parity with aggregated serving) — see the caveats section
+for constraints.
diff -- examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml
@@ -0,0 +1,136 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/kimi_k3_disagg_parity.py` added +792/-0
  - docs: `examples/kimi_k3/disagg/README.md` added +165/-0; `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` added +136/-0; `examples/kimi_k3/disagg/ctx_config.yaml` added +67/-0; `examples/kimi_k3/disagg/gen_config_no_sa.yaml` added +50/-0; `examples/kimi_k3/disagg/disagg_proxy_config.yaml` added +23/-0; `examples/kimi_k3/README.md` modified +0/-1
  - runtime: `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +129/-0
- Risk and verification: The diff ships test coverage in `examples/disaggregated/slurm/cache_transceiver_test/configs/kda_payload_kimi_k3.yaml`, `examples/disaggregated/slurm/cache_transceiver_test/run_cache_transceiver_test.py`, `tests/integration/defs/disaggregated/test_configs/disagg_config_overlap_transceiver_runtime_python_bounce.yaml`, `tests/integration/defs/kimi_k3_disagg_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17445 - [None][fix] Kimi K3 MLA: pass attn_output to MLA.forward_impl

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17445
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; associated commits `4ec478deded5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +7/-6, 22 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-6 (13 lines); hunks: -228,14 +228,15 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-6 (13 lines); hunks: -228,14 +228,15 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -228,14 +228,15 @@ def forward(
-        attn_out = self.create_output(
-            hidden_states,
-            attn_metadata.num_contexts,
-        )
+        # _create_outputs() rather than create_output(): the base implementation
+        # takes a list so a sparse-attention backend can append its own buffers,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-6
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17421 - [None][fix] Kimi K3: eager CUDA-graph buffer allocation and prebuilt fused-verify constants

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17421
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `ae4520506f4d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +53/-24, 131 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +49/-24 (73 lines); hunks: -986,8 +986,13 @@ def __init__(; -1056,7 +1061,7 @@ def _build_decode_kernel_constants(self) -> None:; symbols: __init__, finalize_decode_weights, _build_decode_kernel_constants, finalize_decode_weights_fp8, touching `__init__, finalize_decode_weights, _build_decode_kernel_constants`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +49/-24 (73 lines); hunks: -986,8 +986,13 @@ def __init__(; -1056,7 +1061,7 @@ def _build_decode_kernel_constants(self) -> None:; symbols: __init__, finalize_decode_weights, _build_decode_kernel_constants, finalize_decode_weights_fp8
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -986,8 +986,13 @@ def __init__(
-        # per-section conv windows (lazily sized to the pool slot count).
+        # per-section conv windows. Sized once, on the first decode call,
+        # to the conv pool's slot count and never reallocated (see
+        # ``_forward_decode``).
+        # fp32 [dim, W] conv weights for the fused verify kernel, prebuilt
+        # by ``_build_mtp_conv_weights()`` at weight-load finalize time.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +49/-24
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_kda_fused_verify_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17456 - [None][doc] Note runtime-dependency requirement for the kimi_k3 Slurm container image

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17456
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/README.md`; associated commits `6fb2f4fc996f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +17/-1, 25 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/README.md` modified +17/-1 (18 lines); hunks: -32,7 +32,23 @@ other GPU architectures may be added in a future release..
- Code diff details:
  - `examples/kimi_k3/README.md` modified +17/-1 (18 lines); hunks: -32,7 +32,23 @@ other GPU architectures may be added in a future release.
- Key code excerpts:

```diff
diff -- examples/kimi_k3/README.md
@@ -32,7 +32,23 @@ other GPU architectures may be added in a future release.
-  image.
+  image. The image passed as `--image` below must already provide
+  TensorRT-LLM's runtime dependencies, that is, a release-style TensorRT-LLM
+  container; a build or devel image without them does not work. The Slurm
+  scripts start a fresh container from that image. They mount the
+  repository, including `.venv-3.12`, and the checkpoint, so packages installed in
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/README.md` modified +17/-1
- Risk and verification: This is mostly docs/examples in `examples/kimi_k3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17446 - [TRTLLM-15215][fix] Kimi K3: make the FP8 weight-read master switch opt-in

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17446
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `c67879c43287`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +103/-4, 129 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +21/-4 (25 lines); hunks: -135,7 +135,9; -178,6 +180,23; symbols: _resolve_fp8_weight_read_gates, get_tensor, touching `_resolve_fp8_weight_read_gates, get_tensor`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +21/-4 (25 lines); hunks: -135,7 +135,9; -178,6 +180,23; symbols: _resolve_fp8_weight_read_gates, get_tensor
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -135,7 +135,9 @@
-# lossy relative to BF16; set this to "0" to keep BF16.
+# lossy relative to BF16, so it is opt-in: set this to "1" to trade accuracy
+# for decode bandwidth. Default "0" keeps BF16, which is what the published
+# accuracy numbers are measured against.
@@ -178,6 +180,23 @@
+def _resolve_fp8_weight_read_gates() -> tuple[bool, bool, bool]:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +21/-4
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_k3_fp8_weight_read_gates.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17413 - [TRTLLM-15177][chore] Kimi K3 post-merge cleanup: config/import/test hygiene + L0 wiring

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17413
- Status/date: merged / 2026-08-10
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `f949d3b8924e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +205/-184, 655 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +100/-79 (179 lines); hunks: -74,10 +74,14; -95,7 +99,7; symbols: forward, _swap_linear_to_fp8_weight_read, _convert_moe_mlps_to_fp8_weight_read, _swap, touching `forward, _swap_linear_to_fp8_weight_read, _convert_moe_mlps_to_fp8_weight_read`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +100/-79 (179 lines); hunks: -74,10 +74,14; -95,7 +99,7; symbols: forward, _swap_linear_to_fp8_weight_read, _convert_moe_mlps_to_fp8_weight_read, _swap
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -74,10 +74,14 @@
+import gc
+import json
-from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
+from contextlib import ExitStack
+from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
+from safetensors import safe_open
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +100/-79
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_kimi_kda_fused_verify_parity.py`, `tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/_moe_kernels.py`, `tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/_mxfp4.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17479 - [TRTLLM-15264][doc] Kimi K3 disagg: stop recommending a UCX_TLS pin by default

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17479
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml`; associated commits `fdbe8d5d0c9b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +30/-24, 89 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/disagg/README.md` modified +19/-16 (35 lines); hunks: -53,14 +53,12 @@ for constraints.; -103,9 +101,11 @@ python3 examples/disaggregated/slurm/benchmark/submit.py \; `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` modified +11/-8 (19 lines); hunks: -57,13 +57,16 @@ environment:; -73,7 +76,7 @@ environment:.
- Code diff details:
  - `examples/kimi_k3/disagg/README.md` modified +19/-16 (35 lines); hunks: -53,14 +53,12 @@ for constraints.; -103,9 +101,11 @@ python3 examples/disaggregated/slurm/benchmark/submit.py \
  - `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` modified +11/-8 (19 lines); hunks: -57,13 +57,16 @@ environment:; -73,7 +76,7 @@ environment:
- Key code excerpts:

```diff
diff -- examples/kimi_k3/disagg/README.md
@@ -53,14 +53,12 @@ for constraints.
-Each K3 worker spans 16 GPUs (4 NVL72 nodes at 4 GPUs/node). Environment
-prerequisites for every worker shell (see caveats below for why):
-'''bash
-export UCX_TLS=tcp,self,sm,cuda_copy,cuda_ipc   # on clusters where verbs cannot
-                                                # initialize; a container-default
-                                                # UCX_TLS=tcp breaks V2 NIXL
diff -- examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml
@@ -57,13 +57,16 @@ environment:
-  # - On clusters where verbs UCX transports cannot initialize on the
-  #   compute nodes, UCX_TLS must be pinned — UCX_TLS=all can wedge
-  #   native NIXL init asymmetrically, leaving the surviving ranks hung
-  #   in the V2 setup MPI collectives. start_worker.sh clears
-  #   UCX_TLS (a container-provided UCX_TLS=tcp also breaks V2 NIXL VRAM
-  #   registration), so the pin is carried via TRTLLM_WORKER_UCX_TLS and
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/disagg/README.md` modified +19/-16; `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml` modified +11/-8
- Risk and verification: This is mostly docs/examples in `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/benchmark_kimi_k3_dep16.yaml`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17455 - [TRTLLM-14814][chore] Add SA speculative-decoding eval config for Kimi K3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17455
- Status/date: merged / 2026-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/README.md`, `examples/kimi_k3/eval_extra_llm_options_sa.yaml`; associated commits `e61a6e93c9f0`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +79/-4, 136 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/eval_extra_llm_options_sa.yaml` added +41/-0 (41 lines); hunks: -0,0 +1,41; `examples/kimi_k3/README.md` modified +16/-1 (17 lines); hunks: -113,6 +113,21 @@ approximately:; -208,4 +223,4 @@ default cache manager..
- Code diff details:
  - `examples/kimi_k3/eval_extra_llm_options_sa.yaml` added +41/-0 (41 lines); hunks: -0,0 +1,41
  - `examples/kimi_k3/README.md` modified +16/-1 (17 lines); hunks: -113,6 +113,21 @@ approximately:; -208,4 +223,4 @@ default cache manager.
- Key code excerpts:

```diff
diff -- examples/kimi_k3/eval_extra_llm_options_sa.yaml
@@ -0,0 +1,41 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Suffix-automaton (SA) speculative-decoding evaluation config. Identical
+# to eval_extra_llm_options.yaml except for the SA-specific keys, each
+# marked below: max_batch_size, cuda_graph_config.max_batch_size,
diff -- examples/kimi_k3/README.md
@@ -113,6 +113,21 @@ approximately:
+To evaluate with suffix-automaton (SA) speculative decoding, add `--sa`:
+'''bash
+sbatch examples/kimi_k3/run_gsm8k_kimi_k3.sbatch \
+    --model /path/to/kimi-k3-checkpoint \
+    --image /path/to/tensorrt-llm-container.sqsh \
+    --sa
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/eval_extra_llm_options_sa.yaml` added +41/-0; `examples/kimi_k3/README.md` modified +16/-1
- Risk and verification: This is mostly docs/examples in `examples/kimi_k3/README.md`, `examples/kimi_k3/eval_extra_llm_options_sa.yaml`, `examples/kimi_k3/run_gsm8k_kimi_k3.sbatch`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #17050 - [TRTLLM-14704][feat] Support multi-modal part of K3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17050
- Status/date: merged / 2026-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/README.md`, `examples/kimi_k3/perf_sweep/acc_sweep.sbatch`, `examples/kimi_k3/run_dspark_acceptance.sbatch`, `examples/kimi_k3/run_eval_kimi_k3.sbatch` and 10 files; associated commits `1f17e7cc3f48`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +1746/-236, 2245 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` added +540/-0 (540 lines); hunks: -0,0 +1,540; symbols: K3PatchEmbed3d, __init__, forward, K3VisionMLP, touching `K3PatchEmbed3d, __init__, forward`; `tensorrt_llm/_torch/configs/kimi_k3.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiK3VisionConfig, __init__, KimiK3Config, touching `KimiK3VisionConfig, __init__, KimiK3Config`; `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +64/-17 (81 lines); hunks: -355,8 +355,28 @@ def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:; -694,6 +714,7 @@ def __init__(; symbols: _gelu_tanh, _get_vision_tp_mapping, _vision_requires_replication, __init__, touching `_gelu_tanh, _get_vision_tp_mapping, _vision_requires_replication`; `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +6/-2 (8 lines); hunks: -2089,10 +2089,14 @@ def _materialize(value) -> torch.Tensor:; symbols: _materialize, KimiLinearForCausalLM, __init__, touching `_materialize, KimiLinearForCausalLM, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` added +540/-0 (540 lines); hunks: -0,0 +1,540; symbols: K3PatchEmbed3d, __init__, forward, K3VisionMLP
  - `tensorrt_llm/_torch/configs/kimi_k3.py` added +144/-0 (144 lines); hunks: -0,0 +1,144; symbols: KimiK3VisionConfig, __init__, KimiK3Config
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +64/-17 (81 lines); hunks: -355,8 +355,28 @@ def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:; -694,6 +714,7 @@ def __init__(; symbols: _gelu_tanh, _get_vision_tp_mapping, _vision_requires_replication, __init__
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +6/-2 (8 lines); hunks: -2089,10 +2089,14 @@ def _materialize(value) -> torch.Tensor:; symbols: _materialize, KimiLinearForCausalLM, __init__
  - `examples/kimi_k3/run_eval_kimi_k3.sbatch` added +348/-0 (348 lines); hunks: -0,0 +1,348
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py
@@ -0,0 +1,540 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- tensorrt_llm/_torch/configs/kimi_k3.py
@@ -0,0 +1,144 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""In-tree composite config for Kimi K3 ("kimi_k3") multimodal checkpoints.
+Mirrors the checkpoint-shipped ``configuration_kimi_k3.KimiK3Config`` /
+``KimiK3VisionConfig`` so TRT-LLM can parse the released Kimi K3 VLM checkpoint
+without ``trust_remote_code`` for the config. The composite ``kimi_k3`` config
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -355,8 +355,28 @@ def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` added +540/-0; `tensorrt_llm/_torch/configs/kimi_k3.py` added +144/-0; `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +64/-17; `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +6/-2
  - docs: `examples/kimi_k3/run_eval_kimi_k3.sbatch` added +348/-0; `examples/kimi_k3/run_dspark_acceptance.sbatch` added +189/-0; `examples/kimi_k3/README.md` modified +19/-21
  - tests: `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py` added +151/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py`, `tests/unittest/others/test_lm_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17311 - [None][perf] Fuse Kimi K3 KDA projections

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17311
- Status/date: merged / 2026-08-14
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `a702ae903517`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +437/-137, 873 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +260/-121 (381 lines); hunks: -108,6 +108,12; -165,14 +171,12; symbols: _convert_kda_projections_to_fp8_weight_read, KimiKDARuntime, __init__, touching `_convert_kda_projections_to_fp8_weight_read, KimiKDARuntime, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +260/-121 (381 lines); hunks: -108,6 +108,12; -165,14 +171,12; symbols: _convert_kda_projections_to_fp8_weight_read, KimiKDARuntime, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -108,6 +108,12 @@
+# Heuristic ported from SGLang's Blackwell cutoff:
+# https://github.com/sgl-project/sglang/blob/e84bbf68efb683c9e2eef4168c5198042544599d/python/sglang/srt/models/kimi_k3.py#L946-L954
+# It has not been tuned for TensorRT-LLM; benchmark and retune it for TRT-LLM's
+# projection kernels. Verify intentionally counts B * num_steps because those
+# flattened token rows form the projection GEMMs' M dimension.
+_KDA_BFA_MULTISTREAM_MAX_ROWS = 128
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +260/-121
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_kda_fused_verify_parity.py`, `tests/unittest/_torch/modeling/test_kimi_kda_verify_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17624 - [TRTLLM-15284][feat] add Kimi K3 SiTU MegaMoE support

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17624
- Status/date: merged / 2026-08-15
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/eval_extra_llm_options.yaml`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `c63946966483`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +688/-68, 1154 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +57/-10 (67 lines); hunks: -662,7 +662,7 @@ def _convert_mla_projections_to_fp8_weight_read(model: nn.Mo...; -706,7 +706,7 @@ def __init__(; symbols: _convert_mla_projections_to_fp8_weight_read, KimiK3MoERuntime, __init__, touching `_convert_mla_projections_to_fp8_weight_read, KimiK3MoERuntime, __init__`; `examples/kimi_k3/eval_extra_llm_options.yaml` modified +2/-0 (2 lines); hunks: -11,6 +11,8 @@ cuda_graph_config:.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +57/-10 (67 lines); hunks: -662,7 +662,7 @@ def _convert_mla_projections_to_fp8_weight_read(model: nn.Mo...; -706,7 +706,7 @@ def __init__(; symbols: _convert_mla_projections_to_fp8_weight_read, KimiK3MoERuntime, __init__
  - `examples/kimi_k3/eval_extra_llm_options.yaml` modified +2/-0 (2 lines); hunks: -11,6 +11,8 @@ cuda_graph_config:
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -662,7 +662,7 @@ def _convert_mla_projections_to_fp8_weight_read(model: nn.Module) -> int:
-    """Kimi K3 latent MoE block backed by ConfigurableMoE/TRTLLM-Gen."""
+    """Kimi K3 latent MoE block backed by ConfigurableMoE."""
@@ -706,7 +706,7 @@ def __init__(
-        self.routed_experts = create_moe(
+        routed_moe_kwargs = dict(
@@ -716,20 +716,38 @@ def __init__(
diff -- examples/kimi_k3/eval_extra_llm_options.yaml
@@ -11,6 +11,8 @@ cuda_graph_config:
+  # TRTLLM chunking bound. Kimi's MegaMoE path privately raises this to
+  # max_num_tokens * dp_size for per-rank SymmBuffer capacity.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +57/-10
  - docs: `examples/kimi_k3/eval_extra_llm_options.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modules/moe/test_kimi_k3_situ_moe.py`, `tests/unittest/_torch/modules/moe/test_moe_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17480 - [TRTLLM-15264][fix] Reject non-Python transceiver routes for Kimi K3 disaggregated serving

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17480
- Status/date: merged / 2026-08-17
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `55be7e53d345`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +94/-1, 130 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +18/-1 (19 lines); hunks: -78,7 +78,7; -2144,6 +2144,23 @@ def get_model_defaults(cls, llm_args) -> dict:; symbols: get_model_defaults, get_preferred_transceiver_runtime, touching `get_model_defaults, get_preferred_transceiver_runtime`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +18/-1 (19 lines); hunks: -78,7 +78,7; -2144,6 +2144,23 @@ def get_model_defaults(cls, llm_args) -> dict:; symbols: get_model_defaults, get_preferred_transceiver_runtime
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -78,7 +78,7 @@
-from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
+from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple
@@ -2144,6 +2144,23 @@ def get_model_defaults(cls, llm_args) -> dict:
+    @classmethod
+    def get_preferred_transceiver_runtime(
+        cls,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +18/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/waives.txt`, `tests/unittest/_torch/executor/test_mamba_cache_manager.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17312 - [None][refactor] Refactor Kimi K3 MLP

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17312
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `bef843ef979f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +730/-855, 1944 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +241/-98 (339 lines); hunks: -50,8 +50,15; -76,6 +83,7; symbols: KimiK3MoEGate, __init__, compute_logits, routing_method, touching `KimiK3MoEGate, __init__, compute_logits`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +241/-98 (339 lines); hunks: -50,8 +50,15; -76,6 +83,7; symbols: KimiK3MoEGate, __init__, compute_logits, routing_method
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -50,8 +50,15 @@
-nonlinear/linear layers applied to the full sum). ``lm_head`` uses the stock
-``LMHead`` (vocab-sharded + gather), so logits are identical on all ranks.
+nonlinear/linear layers applied to the full sum). When attention DP is off,
+the shared experts use standard MLP TP over the model TP group: gate/up are
+column-sharded and down is row-sharded. Direct MoE-TP combines the shared
+hidden-width partial and routed latent partial into one all-reduce after the
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +241/-98
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/_moe_kernels.py`, `tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/_mxfp4.py`, `tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/kimi_k3_mlp_test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17802 - [None][test] Add kimi k3 cases for multi-node disagg

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17802
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`; associated commits `aede3825b7c4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +135/-0, 150 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` added +130/-0 (130 lines); hunks: -0,0 +1,130.
- Code diff details:
  - `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` added +130/-0 (130 lines); hunks: -0,0 +1,130
- Key code excerpts:

```diff
diff -- tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml
@@ -0,0 +1,130 @@
+metadata:
+  model_name: kimi_k3
+  precision: fp4
+  model_dir_name: Kimi-K3
+  supported_gpus:
+  - GB300
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` added +130/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/perf/_model_paths.py`, `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17861 - [None][test] Add back kimi k25 and deepseek v32 cases from qa side

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17861
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` and 6 files; associated commits `5a8462aef72e`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 12 files, +50/-7, 211 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +7/-1 (8 lines); hunks: -35,7 +35,7 @@ environment:; -74,6 +74,9 @@ worker_config:; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -74,6 +74,8 @@ worker_config:; -99,5 +101,7 @@ worker_config:; `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -72,6 +72,8 @@ worker_config:; -103,6 +105,8 @@ worker_config:; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -77,6 +77,7 @@ worker_config:; -107,6 +108,7 @@ worker_config:.
- Code diff details:
  - `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +7/-1 (8 lines); hunks: -35,7 +35,7 @@ environment:; -74,6 +74,9 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -74,6 +74,8 @@ worker_config:; -99,5 +101,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0 (4 lines); hunks: -72,6 +72,8 @@ worker_config:; -103,6 +105,8 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -77,6 +77,7 @@ worker_config:; -107,6 +108,7 @@ worker_config:
  - `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0 (2 lines); hunks: -72,6 +72,7 @@ worker_config:; -103,6 +104,7 @@ worker_config:
- Key code excerpts:

```diff
diff -- tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml
@@ -35,7 +35,7 @@ environment:
-  worker_env_var: TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1 ENROOT_ALLOW_DEV=yes
+  worker_env_var: TLLM_LOG_LEVEL=INFO TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1 TRTLLM_ENABLE_PDL=1 ENROOT_ALLOW_DEV=yes TRTLLM_KV_TRANSFER_NUM_THREADS=4
@@ -74,6 +74,9 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_cache_bounce_size_mb: 2048
+      kv_transfer_timeout_ms: 600000
diff -- tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml
@@ -74,6 +74,8 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_transfer_timeout_ms: 600000
@@ -99,5 +101,7 @@ worker_config:
+      transceiver_runtime: PYTHON
+      kv_transfer_timeout_ms: 600000
diff -- tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml
@@ -72,6 +72,8 @@ worker_config:
+      transceiver_runtime: PYTHON
```

- Extracted files (not manually reviewed):
  - tests: `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +7/-1; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4096_ctx1_dep4_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +4/-0; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf/disaggregated/gb200_kimi-k25-thinking-fp4_8k1k_con4_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml` modified +2/-0; `tests/scripts/perf/disaggregated/gb300_kimi-k25-thinking-fp4_8k1k_con1024_ctx1_dep4_gen1_dep32_eplb416_mtp3_ccb-NIXL.yaml` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_multinode.txt`, `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con256_ctx1_dep4_gen1_dep32_eplb0_mtp3_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_deepseek-v32-fp4_32k4k_con256_ctx1_dep8_gen1_dep8_eplb0_mtp0_ccb-NIXL.yaml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17053 - [None][perf] Preallocate Kimi attention residual snapshots

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17053
- Status/date: merged / 2026-08-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `e59fbf2bfd6b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +29/-12, 97 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +29/-12 (41 lines); hunks: -2281,26 +2281,30 @@ def forward(; -2312,7 +2316,7 @@ def forward(; symbols: forward, __init__, touching `forward, __init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +29/-12 (41 lines); hunks: -2281,26 +2281,30 @@ def forward(; -2312,7 +2316,7 @@ def forward(; symbols: forward, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -2281,26 +2281,30 @@ def forward(
+        num_snapshots: int,
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
+    ) -> Tuple[torch.Tensor, int]:
-        Returns ``(prefix_sum, block_residual)`` with the snapshot stack in
-        kernel-native ``[K, M, H]`` layout; the running prefix sum is the
-        hidden state handed to the next layer.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +29/-12
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17939 - [TRTLLM-15465][feat] Support SA speculative decoding under disaggregated serving for Kimi K3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17939
- Status/date: merged / 2026-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/README.md`, `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/gen_config.yaml`; associated commits `8325542983ad`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +177/-36, 311 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/disagg/README.md` modified +40/-35 (75 lines); hunks: -11,7 +11,8 @@ for constraints.; -34,8 +35,8 @@ for constraints.; `examples/kimi_k3/disagg/gen_config.yaml` added +48/-0 (48 lines); hunks: -0,0 +1,48; `examples/kimi_k3/README.md` modified +1/-1 (2 lines); hunks: -221,4 +221,4 @@ decoding requires the default cache manager, which cannot re....
- Code diff details:
  - `examples/kimi_k3/disagg/README.md` modified +40/-35 (75 lines); hunks: -11,7 +11,8 @@ for constraints.; -34,8 +35,8 @@ for constraints.
  - `examples/kimi_k3/disagg/gen_config.yaml` added +48/-0 (48 lines); hunks: -0,0 +1,48
  - `examples/kimi_k3/README.md` modified +1/-1 (2 lines); hunks: -221,4 +221,4 @@ decoding requires the default cache manager, which cannot re...
- Key code excerpts:

```diff
diff -- examples/kimi_k3/disagg/README.md
@@ -11,7 +11,8 @@ for constraints.
-| `gen_config_no_sa.yaml` | Generation-server options, no speculative decoding (CUDA graphs ON by default: GSM8K 96.89, 765/2138 tok/s @c64/c256 vs aggregated 643/1972; null `cuda
+| `gen_config.yaml` | Generation-server options WITH suffix-automaton (SA) speculative decoding (DEP16, eager) |
+| `gen_config_no_sa.yaml` | Generation-server options WITHOUT spec decode — use this first (CUDA graphs ON by default: GSM8K 96.89, 765/2138 tok/s @c64/c256 vs aggregated 643/1972
@@ -34,8 +35,8 @@ for constraints.
-  requirement) and on the gen server (keeps the smoke runs maximally
-  comparable across configurations).
diff -- examples/kimi_k3/disagg/gen_config.yaml
@@ -0,0 +1,48 @@
+# Kimi K3 disaggregated serving - GENERATION (decode) server extra LLM-API
+# options WITH suffix-automaton (SA) speculative decoding
+# (`trtllm-serve <model> --config gen_config.yaml`).
+#
+# DEP16 deployment (attention data-parallel + MoE EP dispatch/combine),
+# mirroring examples/kimi_k3/eval_extra_llm_options_sa.yaml: SA runs
diff -- examples/kimi_k3/README.md
@@ -221,4 +221,4 @@ decoding requires the default cache manager, which cannot reuse blocks.
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/disagg/README.md` modified +40/-35; `examples/kimi_k3/disagg/gen_config.yaml` added +48/-0; `examples/kimi_k3/README.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/disaggregated/test_configs/disagg_config_sa.yaml`, `tests/integration/defs/disaggregated/test_configs/disagg_config_sa_python.yaml`, `tests/integration/defs/disaggregated/test_disaggregated.py`, `tests/integration/test_lists/qa/llm_function_core.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17684 - [None][feat] Remove padding in Kimi K3 MLA module

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17684
- Status/date: merged / 2026-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; associated commits `4815338556cb`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +82/-199, 440 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +55/-187 (242 lines); hunks: -673,78 +673,6 @@ def _convert_kda_projections_to_fp8_weight_read(model: nn.M...; -2076,15 +2004,13 @@ def _output_gate_and_proj(; symbols: _convert_kda_projections_to_fp8_weight_read, _load_kimi_k3_mla_kv_b_proj, _convert_mla_projections_to_fp8_weight_read, _output_gate_and_proj, touching `_convert_kda_projections_to_fp8_weight_read, _load_kimi_k3_mla_kv_b_proj, _convert_mla_projections_to_fp8_weight_read`; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +14/-7 (21 lines); hunks: -13,13 +13,12; -158,15 +157,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +55/-187 (242 lines); hunks: -673,78 +673,6 @@ def _convert_kda_projections_to_fp8_weight_read(model: nn.M...; -2076,15 +2004,13 @@ def _output_gate_and_proj(; symbols: _convert_kda_projections_to_fp8_weight_read, _load_kimi_k3_mla_kv_b_proj, _convert_mla_projections_to_fp8_weight_read, _output_gate_and_proj
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +14/-7 (21 lines); hunks: -13,13 +13,12; -158,15 +157,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -673,78 +673,6 @@ def _convert_kda_projections_to_fp8_weight_read(model: nn.Module) -> int:
-@torch.no_grad()
-def _load_kimi_k3_mla_kv_b_proj(
-    mixer: nn.Module,
-    source: torch.Tensor,
-    *,
-    head_start: int,
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -13,13 +13,12 @@
-from torch import nn
-from ....models.modeling_utils import QuantConfig
+from ..linear import Linear, TensorParallelMode
@@ -158,15 +157,12 @@ def __init__(
-        quant_config: Optional[QuantConfig] = None,
+        model_config: ModelConfig,
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +55/-187; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +14/-7
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/attention_backend/fmha/cute_dsl_mla.py`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17804 - [None][feat] Add Kimi K3 to layer-wise benchmarks

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17804
- Status/date: merged / 2026-08-20
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `32dbd5b41cef`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +383/-17, 666 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +32/-6 (38 lines); hunks: -549,6 +549,16 @@ def _swap_linear_to_fp8_weight_read(; -563,6 +573,8 @@ def _convert_moe_mlps_to_fp8_weight_read(; symbols: _swap_linear_to_fp8_weight_read, _has_weights, _convert_moe_mlps_to_fp8_weight_read, _convert_kda_projections_to_fp8_weight_read, touching `_swap_linear_to_fp8_weight_read, _has_weights, _convert_moe_mlps_to_fp8_weight_read`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +32/-6 (38 lines); hunks: -549,6 +549,16 @@ def _swap_linear_to_fp8_weight_read(; -563,6 +573,8 @@ def _convert_moe_mlps_to_fp8_weight_read(; symbols: _swap_linear_to_fp8_weight_read, _has_weights, _convert_moe_mlps_to_fp8_weight_read, _convert_kda_projections_to_fp8_weight_read
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -549,6 +549,16 @@ def _swap_linear_to_fp8_weight_read(
+def _has_weights(module: nn.Module) -> bool:
+    """False once ``modeling_utils.remove_weights()`` has stripped a module.
+    Post-load finalization walks every decoder layer, so it must skip layers
+    whose parameters were dropped — the layer-wise benchmarks keep only the
+    profiled slice resident.
+    """
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +32/-6
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/tools/test_layer_wise_benchmarks.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17865 - [None][feat] bring up Kimi K3 NVFP4 with CUTLASS and cuteDSL MegaMoE SiTU

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17865
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_gpqa.yaml`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe.yaml`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep8.yaml` and 9 files; associated commits `19573c81f30f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 31 files, +2553/-125, 3498 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +635/-38 (673 lines); hunks: -85,8 +85,20; -100,13 +112,14; symbols: _resolve_fp8_weight_read_gates, _resolve_kimi_situ_betas, quantize_weight, prepare_checkpoint_scale, touching `_resolve_fp8_weight_read_gates, _resolve_kimi_situ_betas, quantize_weight`; `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +4/-0 (4 lines); hunks: -1733,6 +1733,10 @@ def load_weights(self, weights) -> None:; symbols: load_weights, touching `load_weights`; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml` added +52/-0 (52 lines); hunks: -0,0 +1,52; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml` added +48/-0 (48 lines); hunks: -0,0 +1,48.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +635/-38 (673 lines); hunks: -85,8 +85,20; -100,13 +112,14; symbols: _resolve_fp8_weight_read_gates, _resolve_kimi_situ_betas, quantize_weight, prepare_checkpoint_scale
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +4/-0 (4 lines); hunks: -1733,6 +1733,10 @@ def load_weights(self, weights) -> None:; symbols: load_weights
  - `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml` added +52/-0 (52 lines); hunks: -0,0 +1,52
  - `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml` added +48/-0 (48 lines); hunks: -0,0 +1,48
  - `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_gpqa.yaml` added +37/-0 (37 lines); hunks: -0,0 +1,37
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -85,8 +85,20 @@
+import threading
-from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple
+from typing import (
+    TYPE_CHECKING,
+    Any,
+    Callable,
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -1733,6 +1733,10 @@ def load_weights(self, weights) -> None:
+            checkpoint_dir = getattr(weights, "checkpoint_dir", None)
+            if checkpoint_dir is not None:
+                lm_weights.checkpoint_dir = checkpoint_dir
+            lm_weights.checkpoint_prefix = self._LANG_PREFIX
diff -- examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml
@@ -0,0 +1,52 @@
+# Kimi K3 NVFP4 on DEP16 for GPQA-Diamond, MegaMoE CuteDSL (4 nodes x 4 GPU).
+#
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +635/-38; `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +4/-0
  - docs: `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe_gpqa.yaml` added +52/-0; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml` added +48/-0; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_gpqa.yaml` added +37/-0; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16_megamoe.yaml` added +37/-0; `examples/kimi_k3/run_eval_kimi_k3.sbatch` modified +23/-8; `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep8.yaml` added +30/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b300.yml`, `tests/unittest/_torch/modeling/test_modeling_kimi_k25.py`, `tests/unittest/_torch/models/checkpoints/hf/test_weight_loader.py`, `tests/unittest/_torch/modules/moe/test_kimi_k3_situ_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17846 - [TRTLLM-14818][test] Port Kimi K3 DFlash/DSpark eval helpers and KDA FP8 prefill test to main

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17846
- Status/date: merged / 2026-08-21
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/eval_extra_llm_options_dflash.yaml`, `examples/kimi_k3/make_synthetic_dflash_drafter.py`, `examples/kimi_k3/measure_dspark_acceptance.py`, `examples/kimi_k3/run_dspark_acceptance.sbatch`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`; associated commits `d4b7b61e92a7`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +1029/-1, 1048 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/make_synthetic_dflash_drafter.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: even_target_layer_ids, validate_target_layer_ids, drafter_tensor_plan, drafter_config, touching `even_target_layer_ids, validate_target_layer_ids, drafter_tensor_plan`; `examples/kimi_k3/measure_dspark_acceptance.py` added +383/-0 (383 lines); hunks: -0,0 +1,383; symbols: parse_arguments, load_prompts, build_llm, _remove_warmup, touching `parse_arguments, load_prompts, build_llm`; `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _Cfg, _Layer, __init__, _Model, touching `_Cfg, _Layer, __init__`; `examples/kimi_k3/eval_extra_llm_options_dflash.yaml` added +39/-0 (39 lines); hunks: -0,0 +1,39.
- Code diff details:
  - `examples/kimi_k3/make_synthetic_dflash_drafter.py` added +402/-0 (402 lines); hunks: -0,0 +1,402; symbols: even_target_layer_ids, validate_target_layer_ids, drafter_tensor_plan, drafter_config
  - `examples/kimi_k3/measure_dspark_acceptance.py` added +383/-0 (383 lines); hunks: -0,0 +1,383; symbols: parse_arguments, load_prompts, build_llm, _remove_warmup
  - `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` added +203/-0 (203 lines); hunks: -0,0 +1,203; symbols: _Cfg, _Layer, __init__, _Model
  - `examples/kimi_k3/eval_extra_llm_options_dflash.yaml` added +39/-0 (39 lines); hunks: -0,0 +1,39
  - `examples/kimi_k3/run_dspark_acceptance.sbatch` modified +1/-1 (2 lines); hunks: -167,7 +167,7 @@ if [[ "$SKIP_BASELINE" -eq 0 ]]; then
- Key code excerpts:

```diff
diff -- examples/kimi_k3/make_synthetic_dflash_drafter.py
@@ -0,0 +1,402 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Emit a synthetic (random-weight) Kimi K3 DFlash/DSpark drafter checkpoint.
+The real K3 drafter (training in progress) is a DSpark drafter — DeepSeek's
+DFlash follow-up (arXiv 2607.05147): a dense Qwen3-style parallel block
+backbone (q/k-norm attention, SiLU MLP) plus the DFlash pooling projection,
diff -- examples/kimi_k3/measure_dspark_acceptance.py
@@ -0,0 +1,383 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+r"""Kimi K3 DSpark acceptance / speedup readiness harness.
+Drives generation over a GSM8K prompt subset with the DSpark (DFlash)
+drafter and reports the weights-drop-day figures of merit:
+- AL: mean accepted tokens per target verify step (bonus token included),
diff -- tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py
@@ -0,0 +1,203 @@
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/make_synthetic_dflash_drafter.py` added +402/-0; `examples/kimi_k3/measure_dspark_acceptance.py` added +383/-0; `examples/kimi_k3/eval_extra_llm_options_dflash.yaml` added +39/-0; `examples/kimi_k3/run_dspark_acceptance.sbatch` modified +1/-1
  - tests: `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` added +203/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17999 - [None][fix] release Kimi K3 checkpoint mappings after loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17999
- Status/date: merged / 2026-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `75b023cd1250`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +22/-9, 61 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +5/-0 (5 lines); hunks: -2991,6 +2991,11 @@ def load_weights(self, weights: Dict[str, torch.Tensor])...; symbols: load_weights, _validate_checkpoint_keys, touching `load_weights, _validate_checkpoint_keys`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +5/-0 (5 lines); hunks: -2991,6 +2991,11 @@ def load_weights(self, weights: Dict[str, torch.Tensor])...; symbols: load_weights, _validate_checkpoint_keys
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -2991,6 +2991,11 @@ def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
+        device = next(self.parameters()).device
+        if device.type == "cuda":
+            # Lazy source mappings are load-scoped; finish nonblocking H2D
+            # work before the caller can release the weights container.
+            torch.cuda.synchronize(device)
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17483 - [TRTLLM-15264][test] Kimi K3 disagg review fixups: KDA test geometry, gate docs, example cleanup

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17483
- Status/date: merged / 2026-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/disagg/README.md`, `examples/kimi_k3/disagg/ctx_config.yaml`; associated commits `afe626dd5cd8`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +262/-175, 685 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/disagg/README.md` modified +17/-7 (24 lines); hunks: -20,11 +20,12 @@ for constraints.; -52,6 +53,15 @@ for constraints.; `examples/kimi_k3/disagg/ctx_config.yaml` modified +4/-2 (6 lines); hunks: -8,8 +8,10; `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +12/-9 (21 lines); hunks: -363,12 +363,15 @@ def validate_peer_compatible(; -411,9 +414,9 @@ def _check_global(field: str, self_bytes: int, peer_bytes: i...; symbols: validate_peer_compatible, _check_global, touching `validate_peer_compatible, _check_global`; `tensorrt_llm/_torch/disaggregation/transceiver.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def _create_kv_slice(self, req: LlmRequest) -> KVSlice:; symbols: _create_kv_slice, _slice_num_bytes, touching `_create_kv_slice, _slice_num_bytes`.
- Code diff details:
  - `examples/kimi_k3/disagg/README.md` modified +17/-7 (24 lines); hunks: -20,11 +20,12 @@ for constraints.; -52,6 +53,15 @@ for constraints.
  - `examples/kimi_k3/disagg/ctx_config.yaml` modified +4/-2 (6 lines); hunks: -8,8 +8,10
  - `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +12/-9 (21 lines); hunks: -363,12 +363,15 @@ def validate_peer_compatible(; -411,9 +414,9 @@ def _check_global(field: str, self_bytes: int, peer_bytes: i...; symbols: validate_peer_compatible, _check_global
  - `tensorrt_llm/_torch/disaggregation/transceiver.py` modified +7/-1 (8 lines); hunks: -372,7 +372,13 @@ def _create_kv_slice(self, req: LlmRequest) -> KVSlice:; symbols: _create_kv_slice, _slice_num_bytes
  - `tensorrt_llm/_torch/disaggregation/native/bounce/config.py` modified +7/-0 (7 lines); hunks: -25,6 +25,13
- Key code excerpts:

```diff
diff -- examples/kimi_k3/disagg/README.md
@@ -20,11 +20,12 @@ for constraints.
-- **Matched ctx/gen parallelism (DEP16 = DEP16)** for now. Heterogeneous
-  ctx/gen TP with attention-DP *off* would silently corrupt memory for
-  K3's replicated KDA state and is rejected at peer registration — do
-  not deviate. Hetero DEP with attention-DP on both sides is believed
-  correct but unvalidated.
+- **Matched ctx/gen parallelism (DEP16 = DEP16)** for now: only this
diff -- examples/kimi_k3/disagg/ctx_config.yaml
@@ -8,8 +8,10 @@
-# Ctx/gen parallelism must match: heterogeneous ctx/gen TP is rejected
-# for K3's replicated KDA recurrent state.
+# Keep ctx/gen parallelism matched: heterogeneous ctx/gen TP passes
+# peer validation (the KDA recurrent state is head-sharded with
+# attention-DP off) but only matched DEP16=DEP16 is validated
+# end-to-end.
diff -- tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py
@@ -363,12 +363,15 @@ def validate_peer_compatible(
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/disagg/README.md` modified +17/-7; `examples/kimi_k3/disagg/ctx_config.yaml` modified +4/-2
  - runtime: `tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py` modified +12/-9; `tensorrt_llm/_torch/disaggregation/transceiver.py` modified +7/-1; `tensorrt_llm/_torch/disaggregation/native/bounce/config.py` modified +7/-0
- Risk and verification: The diff ships test coverage in `examples/disaggregated/slurm/cache_transceiver_test/configs/kda_payload_kimi_k3.yaml`, `tests/unittest/disaggregated/test_kda_mamba_transfer.py`, `tests/unittest/disaggregated/test_mamba_transfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17800 - [TRTLLM-15033][feat] Upstream Kimi K3 MLA decode backend selection to main

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17800
- Status/date: merged / 2026-08-22
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/README.md`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`, `tests/scripts/perf/disaggregated/gb300_kimi-k3-fp4_8k1k_con512_ctx1_dep16_gen1_dep16_eplb0_mtp0_ccb-NIXL.yaml`; associated commits `d32190761933`, `f51e32335aef`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 13 files, +717/-25, 1042 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +82/-1 (83 lines); hunks: -10,17 +10,90; -183,6 +256,7 @@ def __init__(; symbols: _select_mla_generation_backend, _kimi_k3_mla_decode_backend_policy, _meta_safe_cast_dtype, __init__, touching `_select_mla_generation_backend, _kimi_k3_mla_decode_backend_policy, _meta_safe_cast_dtype`.
- Code diff details:
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +82/-1 (83 lines); hunks: -10,17 +10,90; -183,6 +256,7 @@ def __init__(; symbols: _select_mla_generation_backend, _kimi_k3_mla_decode_backend_policy, _meta_safe_cast_dtype, __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -10,17 +10,90 @@
+import os
+from functools import partial
-from ...attention_backend import AttentionMetadata, TrtllmAttention
+from ....logger import logger
+from ....models.modeling_utils import QuantConfig
+from ...attention_backend import AttentionMetadata, TrtllmAttention, TrtllmAttentionMetadata
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +82/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/sysinfo/get_sysinfo.py`, `tests/unittest/_torch/attention/sparse/dsa/test_req_idx_per_token.py`, `tests/unittest/_torch/attention/test_fmha_page_index.py`, `tests/unittest/_torch/modules/test_kimi_k3_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17822 - [TRTLLM-15498][refactor] consolidate Kimi KDA production frontend

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17822
- Status/date: merged / 2026-08-23
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_kda/__init__.py`, `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` and 12 files; associated commits `88592cda9dfa`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 16 files, +1802/-1713, 4235 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +28/-983 (1011 lines); hunks: -109,12 +109,13; -129,14 +130,6; symbols: _convert_kda_projections_to_fp8_weight_read, _routed_output, _kda_split_conv_sections, KimiKDARuntime, touching `_convert_kda_projections_to_fp8_weight_read, _routed_output, _kda_split_conv_sections`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +944/-577 (1521 lines); hunks: -1,83 +1,69; -86,162 +72,82 @@ def reset_parameters(self) -> None:; symbols: _meta_safe_cast_dtype, _cast, _MetaSafeFusedRMSNormGated, touching `_meta_safe_cast_dtype, _cast, _MetaSafeFusedRMSNormGated`; `tests/unittest/_torch/modules/kimi_kda/kimi_kda_test_utils.py` added +351/-0 (351 lines); hunks: -0,0 +1,351; symbols: get_production_prefill_kernel_path, get_production_decode_kernel_path, _meta_safe_cast_dtype, _cast, touching `get_production_prefill_kernel_path, get_production_decode_kernel_path, _meta_safe_cast_dtype`; `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +202/-42 (244 lines); hunks: -2,12 +2,21; -25,7 +34,9 @@ def _has_supported_gpu() -> bool:; symbols: _has_supported_gpu, _make_attention_pair, _run_production_prefill, touching `_has_supported_gpu, _make_attention_pair, _run_production_prefill`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +28/-983 (1011 lines); hunks: -109,12 +109,13; -129,14 +130,6; symbols: _convert_kda_projections_to_fp8_weight_read, _routed_output, _kda_split_conv_sections, KimiKDARuntime
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +944/-577 (1521 lines); hunks: -1,83 +1,69; -86,162 +72,82 @@ def reset_parameters(self) -> None:; symbols: _meta_safe_cast_dtype, _cast, _MetaSafeFusedRMSNormGated
  - `tests/unittest/_torch/modules/kimi_kda/kimi_kda_test_utils.py` added +351/-0 (351 lines); hunks: -0,0 +1,351; symbols: get_production_prefill_kernel_path, get_production_decode_kernel_path, _meta_safe_cast_dtype, _cast
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +202/-42 (244 lines); hunks: -2,12 +2,21; -25,7 +34,9 @@ def _has_supported_gpu() -> bool:; symbols: _has_supported_gpu, _make_attention_pair, _run_production_prefill
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +160/-25 (185 lines); hunks: -3,17 +3,19; -36,7 +38,9 @@ def _has_supported_gpu() -> bool:; symbols: _has_supported_gpu, _make_attention_pair, _make_cache
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -109,12 +109,13 @@
-from ..distributed import AllReduce, AllReduceParams, AllReduceStrategy
+from ..distributed import AllReduce, AllReduceParams
+from ..modules.kimi_kda import KimiKDALinearAttention
@@ -129,14 +130,6 @@
-_KDA_INDEXED_STATE_POOL_ENABLED = os.environ.get("TLLM_KDA_ENABLE_INDEXED_STATE_POOL", "1") == "1"
-# Heuristic ported from SGLang's Blackwell cutoff:
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -1,83 +1,69 @@
-"""KimiKDALinearAttention — Kimi K3 linear-attention module for the PyTorch backend.
+"""Kimi K3 KDA production frontend for the TensorRT-LLM PyTorch executor.
-Structural mirror of the HF reference ``KimiDeltaAttention`` in
-``modeling_kimi.py``. Same parameter names, same layer shapes, same short
-convolution + FLA gating + FusedRMSNormGated output-gate stack. The
-delta-rule inner loop is routed through :mod:`_kda_kernels`, which selects
diff -- tests/unittest/_torch/modules/kimi_kda/kimi_kda_test_utils.py
@@ -0,0 +1,351 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +28/-983; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +944/-577
  - tests: `tests/unittest/_torch/modules/kimi_kda/kimi_kda_test_utils.py` added +351/-0; `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +202/-42; `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +160/-25; `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` modified +30/-29; `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` added +39/-0; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` renamed +15/-12
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_gb300_multi_gpus.yml`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`, `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17980 - [TRTLLM-15176][fix] Harden Kimi K3 tool-call parsing

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17980
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py`; associated commits `0a4861dcabd9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +600/-22, 782 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +64/-10 (74 lines); hunks: -50,29 +50,43 @@ class KimiK3ToolParser(BaseToolParser):; -99,9 +113,8 @@ def _coerce_value(value: str, value_type: str) -> Any:; symbols: KimiK3ToolParser, __init__, _coerce_value, _parse_call_arguments, touching `KimiK3ToolParser, __init__, _coerce_value`.
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +64/-10 (74 lines); hunks: -50,29 +50,43 @@ class KimiK3ToolParser(BaseToolParser):; -99,9 +113,8 @@ def _coerce_value(value: str, value_type: str) -> Any:; symbols: KimiK3ToolParser, __init__, _coerce_value, _parse_call_arguments
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py
@@ -50,29 +50,43 @@ class KimiK3ToolParser(BaseToolParser):
+    # Forced/named tool_choice has no grammar for XTML (no structural-tag
+    # support), so the model output still carries preamble + markup and the
+    # serving layer must extract instead of passing raw text through.
+    extracts_forced_tool_calls = True
+        # Set once a complete tools section has been emitted. A K3 tools
+        # section terminates the message, so anything streamed afterwards is
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +64/-10
- Risk and verification: The diff ships test coverage in `tests/unittest/llmapi/apps/test_tool_parsers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18059 - [None][fix] Kimi K3: bound MegaMoE expert-weight memory at EP8; drop the MoE TP/EP env overrides

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18059
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`; associated commits `3d4e91920bde`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +390/-102, 628 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-24 (27 lines); hunks: -45,8 +45,7; -130,14 +129,6; symbols: _select_moe_tp_ep, touching `_select_moe_tp_ep`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-24 (27 lines); hunks: -45,8 +45,7; -130,14 +129,6; symbols: _select_moe_tp_ep
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -45,8 +45,7 @@
-``moe_expert_parallel_size`` explicitly (or the ``TLLM_K3_MOE_TP_SIZE`` /
-``TLLM_K3_MOE_EP_SIZE`` env overrides). Routing is computed replicated; the
+``moe_expert_parallel_size`` explicitly. Routing is computed replicated; the
@@ -130,14 +129,6 @@
-# Routed-expert MoE TP/EP split overrides (read per model init, not import).
-# Highest precedence; either one may be set alone, the other is derived from
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-24
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b300.yml`, `tests/unittest/_torch/modules/moe/test_kimi_k3_situ_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17796 - [None][feat] Kimi K3: KDA-TP + MLA-DCP (helix) wiring

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17796
- Status/date: merged / 2026-08-25
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; associated commits `f1f9f00b0883`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +135/-9, 270 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +112/-3 (115 lines); hunks: -148,6 +148,8; -2406,10 +2408,11 @@ class KimiMLARuntime(nn.Module):; symbols: KimiMLARuntime, __init__, forward, touching `KimiMLARuntime, __init__, forward`; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-3 (10 lines); hunks: -18,6 +18,7; -231,6 +232,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +112/-3 (115 lines); hunks: -148,6 +148,8; -2406,10 +2408,11 @@ class KimiMLARuntime(nn.Module):; symbols: KimiMLARuntime, __init__, forward
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-3 (10 lines); hunks: -18,6 +18,7; -231,6 +232,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -148,6 +148,8 @@
+    from ...llmapi.llm_args import DecodingBaseConfig
@@ -2406,10 +2408,11 @@ class KimiMLARuntime(nn.Module):
-        cfg,
+        cfg: "PretrainedConfig",
-    ):
+        mapping_with_cp: Optional[Mapping] = None,
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -18,6 +18,7 @@
+from ....mapping import Mapping
@@ -231,6 +232,7 @@ def __init__(
+        mapping_with_cp: Optional[Mapping] = None,
@@ -253,6 +255,7 @@ def __init__(
+            mapping_with_cp=mapping_with_cp,
@@ -266,14 +269,15 @@ def __init__(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +112/-3; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +7/-3
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`, `tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17741 - [None][feat] k3 weight pipeline opt

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17741
- Status/date: merged / 2026-08-27
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`; associated commits `88f79692f938`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +332/-109, 644 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +116/-77 (193 lines); hunks: -116,6 +116,7; -440,6 +441,74 @@ def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:; symbols: _gate_up_ckpt_keys, _shard_head_major_param, _helix_cp_v_b_shard, _load_trunk_params, touching `_gate_up_ckpt_keys, _shard_head_major_param, _helix_cp_v_b_shard`; `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +147/-2 (149 lines); hunks: -1,6 +1,6; -9,7 +9,11; symbols: test_checkpoint_plan_preserves_external_attention_names, _trunk_parameters, _distinct, test_shard_kda_column_projection, touching `test_checkpoint_plan_preserves_external_attention_names, _trunk_parameters, _distinct`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +20/-26 (46 lines); hunks: -316,7 +316,7 @@ def forward(; -336,9 +336,9 @@ def forward(; symbols: forward, _has_kda_replay_caches, touching `forward, _has_kda_replay_caches`; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +15/-4 (19 lines); hunks: -64,6 +64,9 @@ def __init__(self, slots, dim3, w, h, v, k, t_max, device):; -140,10 +143,14 @@ def test_kda_fused_prefill_matches_separate_projections():; symbols: __init__, _decode_metadata, test_kda_fused_prefill_matches_separate_projections, test_kda_qkvg_multistream_decode_matches_separate_projections, touching `__init__, _decode_metadata, test_kda_fused_prefill_matches_separate_projections`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +116/-77 (193 lines); hunks: -116,6 +116,7; -440,6 +441,74 @@ def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:; symbols: _gate_up_ckpt_keys, _shard_head_major_param, _helix_cp_v_b_shard, _load_trunk_params
  - `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +147/-2 (149 lines); hunks: -1,6 +1,6; -9,7 +9,11; symbols: test_checkpoint_plan_preserves_external_attention_names, _trunk_parameters, _distinct, test_shard_kda_column_projection
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +20/-26 (46 lines); hunks: -316,7 +316,7 @@ def forward(; -336,9 +336,9 @@ def forward(; symbols: forward, _has_kda_replay_caches
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +15/-4 (19 lines); hunks: -64,6 +64,9 @@ def __init__(self, slots, dim3, w, h, v, k, t_max, device):; -140,10 +143,14 @@ def test_kda_fused_prefill_matches_separate_projections():; symbols: __init__, _decode_metadata, test_kda_fused_prefill_matches_separate_projections, test_kda_qkvg_multistream_decode_matches_separate_projections
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +2/-0 (2 lines); hunks: -147,6 +147,7 @@ def _conv_cache(section):; -156,6 +157,7 @@ def _make_seq_layer_cache(B):; symbols: _conv_cache, _make_seq_layer_cache
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -116,6 +116,7 @@
+from ..modules.linear import TensorParallelMode, load_weight_shard
@@ -440,6 +441,74 @@ def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
+def _shard_head_major_param(
+    name: str,
+    src: torch.Tensor,
+    param: torch.nn.Parameter,
diff -- tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py
@@ -1,6 +1,6 @@
-"""Checkpoint-name tests for the Kimi Linear model."""
+"""Checkpoint-name and weight-shard tests for the Kimi Linear model."""
@@ -9,7 +9,11 @@
-from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM  # noqa: E402
+from tensorrt_llm._torch.models.modeling_kimi_linear import (  # noqa: E402
+    KimiLinearForCausalLM,
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -316,7 +316,7 @@ def forward(
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +116/-77; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +20/-26
  - tests: `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +147/-2; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +15/-4; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18226 - [None][fix] Keep CuTe-DSL MLA decode for Kimi K3 H=96 speculative-verify batches

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18226
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; associated commits `bd03d5ff32bc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +24/-14, 70 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +11/-9 (20 lines); hunks: -77,19 +77,21 @@ def _kimi_k3_mla_decode_backend_policy(; symbols: _kimi_k3_mla_decode_backend_policy, touching `_kimi_k3_mla_decode_backend_policy`.
- Code diff details:
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +11/-9 (20 lines); hunks: -77,19 +77,21 @@ def _kimi_k3_mla_decode_backend_policy(; symbols: _kimi_k3_mla_decode_backend_policy
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -77,19 +77,21 @@ def _kimi_k3_mla_decode_backend_policy(
-    they fall back to TRTLLM-Gen. The H=96 path is the correctness exception:
-    TRTLLM-Gen may select a 64-head Q tile, which does not divide 96 and
-    produces an invalid configuration after K3's head padding was removed.
-    The CuTe-DSL kernel itself accepts multi-token queries, but K3's decode
-    tuning covers only the one-token-per-request regime, so generation-only
-    speculative verification also falls back.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +11/-9
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modules/test_kimi_k3_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17870 - [TRTLLM-15498][perf] optimize Kimi KDA prefill convolution data flow

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17870
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`, `tests/unittest/_torch/modules/kimi_kda/kda_prefill_test_utils.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` and 8 files; associated commits `2dddac64810f`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 15 files, +610/-229, 1468 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +217/-32 (249 lines); hunks: -2,17 +2,25; -34,39 +42,45 @@ def _has_supported_gpu() -> bool:; symbols: _has_supported_gpu, _make_attention_pair, _make_kda, _make_reference, touching `_has_supported_gpu, _make_attention_pair, _make_kda`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +101/-117 (218 lines); hunks: -10,7 +10,7; -27,10 +27,11; symbols: _cast, _kda_split_conv_sections, _kda_expand_fla_conv_cache, KimiKDALinearAttention, touching `_cast, _kda_split_conv_sections, _kda_expand_fla_conv_cache`; `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py` modified +201/-7 (208 lines); hunks: -27,9 +27,10; -68,6 +69,202 @@ def is_kda_optimized_supported() -> bool:; symbols: is_kda_optimized_supported, _fused_kda_post_conv_kernel, fused_kda_post_conv, _copy_kda_replay_conv_window_kernel, touching `is_kda_optimized_supported, _fused_kda_post_conv_kernel, fused_kda_post_conv`; `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +21/-10 (31 lines); hunks: -139,9 +139,9 @@ def _run_production_decode(; -165,9 +165,20 @@ def _run_production_decode(; symbols: _run_production_decode, test_optimized_decode_updates_indexed_recurrent_state_pool_in_place, touching `_run_production_decode, test_optimized_decode_updates_indexed_recurrent_state_pool_in_place`.
- Code diff details:
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +217/-32 (249 lines); hunks: -2,17 +2,25; -34,39 +42,45 @@ def _has_supported_gpu() -> bool:; symbols: _has_supported_gpu, _make_attention_pair, _make_kda, _make_reference
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +101/-117 (218 lines); hunks: -10,7 +10,7; -27,10 +27,11; symbols: _cast, _kda_split_conv_sections, _kda_expand_fla_conv_cache, KimiKDALinearAttention
  - `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py` modified +201/-7 (208 lines); hunks: -27,9 +27,10; -68,6 +69,202 @@ def is_kda_optimized_supported() -> bool:; symbols: is_kda_optimized_supported, _fused_kda_post_conv_kernel, fused_kda_post_conv, _copy_kda_replay_conv_window_kernel
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +21/-10 (31 lines); hunks: -139,9 +139,9 @@ def _run_production_decode(; -165,9 +165,20 @@ def _run_production_decode(; symbols: _run_production_decode, test_optimized_decode_updates_indexed_recurrent_state_pool_in_place
  - `tests/unittest/_torch/modules/kimi_kda/kda_prefill_test_utils.py` modified +9/-4 (13 lines); hunks: -3,6 +3,7; -26,6 +27,8 @@ def run_indexed_prefill(; symbols: run_indexed_prefill, run_fla_prefill
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py
@@ -2,17 +2,25 @@
+from collections.abc import Mapping
+from fla.modules import ShortConvolution  # noqa: E402
+from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import (  # noqa: E402
+    copy_kda_replay_conv_window,
+    fused_kda_post_conv,
+)
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -10,7 +10,7 @@
-``[slots, 3D, W]`` bf16 pool and one delta-rule recurrent state in the
+``[slots, 3D, W - 1]`` bf16 pool and one delta-rule recurrent state in the
@@ -27,10 +27,11 @@
+from ..mamba.causal_conv1d import causal_conv1d_fn
-from ._kda_kernels import KDAKernelDispatch
+from ._kda_kernels import KDAKernelDispatch, fused_kda_post_conv
diff -- tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py
@@ -27,9 +27,10 @@
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py` modified +217/-32; `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +21/-10; `tests/unittest/_torch/modules/kimi_kda/kda_prefill_test_utils.py` modified +9/-4; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +5/-6; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +4/-4; `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` modified +4/-1
  - runtime: `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +101/-117; `tensorrt_llm/_torch/modules/kimi_kda/_kda_kernels.py` modified +201/-7
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/executor/test_mamba_cache_manager.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`, `tests/unittest/_torch/modules/kimi_kda/kda_prefill_test_utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17845 - [TRTLLM-14764][feat] trtllm-serve: Kimi K3 API compliance for the Kimi Vendor Verifier (KVV)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17845
- Status/date: merged / 2026-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`, `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py`, `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py`; associated commits `11d8ef198ce9`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 14 files, +1132/-43, 1515 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +99/-2 (101 lines); hunks: -28,6 +28,7; -42,6 +43,10 @@ def _unescape_attr(value: str) -> str:; symbols: _unescape_attr, _escape_attr, _parse_attrs, has_tool_call, touching `_unescape_attr, _escape_attr, _parse_attrs`; `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +3/-0 (3 lines); hunks: -443,6 +443,9 @@ class KimiK3InputProcessor(KimiK25InputProcessor):; symbols: KimiK3InputProcessor, KimiK3ForConditionalGeneration, touching `KimiK3InputProcessor, KimiK3ForConditionalGeneration`; `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py` added +572/-0 (572 lines); hunks: -0,0 +1,572; symbols: make_request, dynamic_system_msg, TestToolChoiceValidation, test_required_with_tools_accepted, touching `make_request, dynamic_system_msg, TestToolChoiceValidation`; `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +10/-0 (10 lines); hunks: -210,6 +210,16 @@ These options are set within the YAML file passed to `trtll....
- Code diff details:
  - `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +99/-2 (101 lines); hunks: -28,6 +28,7; -42,6 +43,10 @@ def _unescape_attr(value: str) -> str:; symbols: _unescape_attr, _escape_attr, _parse_attrs, has_tool_call
  - `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +3/-0 (3 lines); hunks: -443,6 +443,9 @@ class KimiK3InputProcessor(KimiK25InputProcessor):; symbols: KimiK3InputProcessor, KimiK3ForConditionalGeneration
  - `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py` added +572/-0 (572 lines); hunks: -0,0 +1,572; symbols: make_request, dynamic_system_msg, TestToolChoiceValidation, test_required_with_tools_accepted
  - `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +10/-0 (10 lines); hunks: -210,6 +210,16 @@ These options are set within the YAML file passed to `trtll...
- Key code excerpts:

```diff
diff -- tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py
@@ -28,6 +28,7 @@
+import os
@@ -42,6 +43,10 @@ def _unescape_attr(value: str) -> str:
+def _escape_attr(value: str) -> str:
+    return value.replace("&", "&amp;").replace('"', "&quot;")
@@ -96,15 +101,107 @@ def has_tool_call(self, text: str) -> bool:
-        # JSON-schema-driven structural-tag constrained decoding used for
diff -- tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py
@@ -443,6 +443,9 @@ class KimiK3InputProcessor(KimiK25InputProcessor):
+        # K3's reference renderer concatenates content parts with no
+        # separator; the default "\n" join skews prompt-token parity.
+        placeholders_separator="",
diff -- tests/unittest/llmapi/apps/test_kimi_serve_extensions.py
@@ -0,0 +1,572 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/serve/tool_parser/kimi_k3_tool_parser.py` modified +99/-2; `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +3/-0
  - tests: `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py` added +572/-0
  - docs: `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +10/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_cpu.yml`, `tests/unittest/api_stability/references/trtllm_serve_api.yaml`, `tests/unittest/llmapi/apps/test_kimi_serve_extensions.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18164 - [TRTLLM-14705][fix] Kimi K3 B200 enablement: MLA decode dispatch fix, L0 wiring, docs

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18164
- Status/date: merged / 2026-08-31
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/README.md`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`; associated commits `3d466dba8dcf`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 7 files, +197/-7, 329 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +39/-3 (42 lines); hunks: -10,7 +10,15 @@ This guide uses Slurm and the `trtllm-llmapi-launch` multi-no...; -26,7 +34,7 @@ The checkpoint and the configuration file must live on a share...; symbols: per, touching `per`; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +33/-0 (33 lines); hunks: -61,6 +61,34 @@ def _select_mla_generation_backend(quant_config: Optional[Qua...; -298,6 +326,11 @@ def __init__(; symbols: _select_mla_generation_backend, _validate_mla_generation_backend, _kimi_k3_mla_decode_backend_policy, __init__, touching `_select_mla_generation_backend, _validate_mla_generation_backend, _kimi_k3_mla_decode_backend_policy`; `examples/kimi_k3/README.md` modified +24/-4 (28 lines); hunks: -5,8 +5,15 @@ start and configuration for GSM8K evaluation.; -27,12 +34,14 @@ other GPU architectures may be added in a future release.; symbols: per, touching `per`.
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +39/-3 (42 lines); hunks: -10,7 +10,15 @@ This guide uses Slurm and the `trtllm-llmapi-launch` multi-no...; -26,7 +34,7 @@ The checkpoint and the configuration file must live on a share...; symbols: per
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +33/-0 (33 lines); hunks: -61,6 +61,34 @@ def _select_mla_generation_backend(quant_config: Optional[Qua...; -298,6 +326,11 @@ def __init__(; symbols: _select_mla_generation_backend, _validate_mla_generation_backend, _kimi_k3_mla_decode_backend_policy, __init__
  - `examples/kimi_k3/README.md` modified +24/-4 (28 lines); hunks: -5,8 +5,15 @@ start and configuration for GSM8K evaluation.; -27,12 +34,14 @@ other GPU architectures may be added in a future release.; symbols: per
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md
@@ -10,7 +10,15 @@ This guide uses Slurm and the `trtllm-llmapi-launch` multi-node launcher. The co
-* GPU: NVIDIA Blackwell GPUs. DEP16 and TEP16 use 16 GPUs; the TEP8 recipe uses 8 GPUs. These deployment recipes were validated on GB300 NVL GPUs. The repository's Slurm examples
+* GPU: NVIDIA Blackwell GPUs. DEP16 and TEP16 use 16 GPUs; the TEP8 recipe uses 8 GPUs. These deployment recipes were validated on GB300 NVL GPUs. The repository's Slurm examples
+  | Recipe | Attention layout | Per-rank weights | Requires |
+  | :-- | :-- | --: | :-- |
+  | DEP16 | attention-DP (replicated) | 210 GB | GB300-class per-GPU memory |
+  | TEP8 | attention-TP, EP8 | 213 GB | GB300-class per-GPU memory |
diff -- tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py
@@ -61,6 +61,34 @@ def _select_mla_generation_backend(quant_config: Optional[QuantConfig]) -> str:
+def _validate_mla_generation_backend(backend: str, num_heads: int) -> None:
+    """Fail fast when `backend` can never run at this per-rank head count.
+    FlashInfer's `trtllm_batch_decode_with_kv_cache_mla` rejects
+    `64 < num_heads_q < 128` for every batch shape, and the per-batch policy
+    never demotes away from an explicit `trtllm-gen` selection (the
+    FP8-KV-cache override or a `TLLM_K3_MLA_GEN_BACKEND=trtllm-gen` request).
diff -- examples/kimi_k3/README.md
@@ -5,8 +5,15 @@ start and configuration for GSM8K evaluation.
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +39/-3; `examples/kimi_k3/README.md` modified +24/-4
  - runtime: `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +33/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_dgx_b200.yml`, `tests/unittest/_torch/modules/test_kimi_k3_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17921 - [TRTLLM-15035][test] Wire Kimi K3 spec-dec and suffix-automaton tests into L0 CI

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17921
- Status/date: merged / 2026-09-01
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/kimi_k3_disagg_parity.py`, `tests/integration/defs/test_kimi_k3_specdec.py`; associated commits `0f94be2600b5`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +166/-167, 718 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/test_kimi_k3_specdec.py` modified +50/-13 (63 lines); hunks: -10,10 +10,14; -38,13 +42,48 @@ def _find_checkpoint():; symbols: _find_checkpoint, _find_lfs_pointer_files, test_kimi_k3_sa_specdec_logits_parity, test_kimi_k3_disagg_parity_selftest, touching `_find_checkpoint, _find_lfs_pointer_files, test_kimi_k3_sa_specdec_logits_parity`; `tests/integration/defs/kimi_k3_disagg_parity.py` modified +1/-1 (2 lines); hunks: -39,7 +39,7.
- Code diff details:
  - `tests/integration/defs/test_kimi_k3_specdec.py` modified +50/-13 (63 lines); hunks: -10,10 +10,14; -38,13 +42,48 @@ def _find_checkpoint():; symbols: _find_checkpoint, _find_lfs_pointer_files, test_kimi_k3_sa_specdec_logits_parity, test_kimi_k3_disagg_parity_selftest
  - `tests/integration/defs/kimi_k3_disagg_parity.py` modified +1/-1 (2 lines); hunks: -39,7 +39,7
- Key code excerpts:

```diff
diff -- tests/integration/defs/test_kimi_k3_specdec.py
@@ -10,10 +10,14 @@
-<LLM_MODELS_ROOT>/Kimi-K3). Skips cleanly when the
-checkpoint is absent. The MoE backend defaults to VANILLA (the reference
-dequant path — the bit-parity oracle; slow but fine at 4 layers) so the
-test has no fused-kernel dependency and runs on any arch.
+<LLM_MODELS_ROOT>/Kimi-K3). Fails — deliberately does not skip — when the
+checkpoint is absent: the test is CI-listed (GB300 post-merge), and a
diff -- tests/integration/defs/kimi_k3_disagg_parity.py
@@ -39,7 +39,7 @@
-    # examples/kimi_k3/run_gsm8k_kimi_k3.sbatch, but served)
+    # examples/kimi_k3/run_eval_kimi_k3.sbatch, but served)
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/test_kimi_k3_specdec.py` modified +50/-13; `tests/integration/defs/kimi_k3_disagg_parity.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/kimi_k3_disagg_parity.py`, `tests/integration/defs/test_kimi_k3_specdec.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_cpu.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18251 - [TRTLLM-15498][perf] optimize Kimi KDA decode data flow

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18251
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/modules/kimi_kda/_kda_decode.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`; associated commits `16881d4fa8ed`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +356/-228, 1098 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +84/-0 (84 lines); hunks: -468,6 +468,90 @@ def _make_direct_decode_args(; symbols: _make_direct_decode_args, test_decode_reads_row_strided_projection_slices, clone_args, _profile_decode_backend, touching `_make_direct_decode_args, test_decode_reads_row_strided_projection_slices, clone_args`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +19/-63 (82 lines); hunks: -194,9 +194,6 @@ def __init__(; -533,15 +530,12 @@ def forward_decode(; symbols: __init__, forward_decode, _project_qkvg, touching `__init__, forward_decode, _project_qkvg`; `tests/unittest/_torch/thop/parallel/test_kda_decode.py` modified +57/-24 (81 lines); hunks: -31,6 +31,9 @@ class KdaInputs:; -75,22 +78,32 @@ def _make_inputs(; symbols: KdaInputs, _make_inputs, make_conv_state, test_kda_decode_matches_fla, touching `KdaInputs, _make_inputs, make_conv_state`; `tensorrt_llm/_torch/modules/kimi_kda/_kda_decode.py` modified +24/-16 (40 lines); hunks: -33,6 +33,13 @@ def _require_cuda_fp32(name: str, tensor: torch.Tensor) -> None:; -159,19 +166,20 @@ def run_kda_decode_fusion_cuda(; symbols: _require_cuda_fp32, _as_token_rows, _dummy_tensor, run_kda_decode_fusion_cuda, touching `_require_cuda_fp32, _as_token_rows, _dummy_tensor`.
- Code diff details:
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +84/-0 (84 lines); hunks: -468,6 +468,90 @@ def _make_direct_decode_args(; symbols: _make_direct_decode_args, test_decode_reads_row_strided_projection_slices, clone_args, _profile_decode_backend
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +19/-63 (82 lines); hunks: -194,9 +194,6 @@ def __init__(; -533,15 +530,12 @@ def forward_decode(; symbols: __init__, forward_decode, _project_qkvg
  - `tests/unittest/_torch/thop/parallel/test_kda_decode.py` modified +57/-24 (81 lines); hunks: -31,6 +31,9 @@ class KdaInputs:; -75,22 +78,32 @@ def _make_inputs(; symbols: KdaInputs, _make_inputs, make_conv_state, test_kda_decode_matches_fla
  - `tensorrt_llm/_torch/modules/kimi_kda/_kda_decode.py` modified +24/-16 (40 lines); hunks: -33,6 +33,13 @@ def _require_cuda_fp32(name: str, tensor: torch.Tensor) -> None:; -159,19 +166,20 @@ def run_kda_decode_fusion_cuda(; symbols: _require_cuda_fp32, _as_token_rows, _dummy_tensor, run_kda_decode_fusion_cuda
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +9/-3 (12 lines); hunks: -196,13 +196,15 @@ def test_kda_prefill_matches_reference():; -243,8 +245,12 @@ def test_kda_qkvg_multistream_decode_matches_separate_proje...; symbols: test_kda_prefill_matches_reference, test_kda_qkvg_multistream_decode_matches_separate_projections, test_kda_decode_matches_reference
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py
@@ -468,6 +468,90 @@ def _make_direct_decode_args(
+@torch.no_grad()
+@pytest.mark.parametrize("num_heads", [2, 32], ids=["compact-heads", "many-heads"])
+def test_decode_reads_row_strided_projection_slices(num_heads: int) -> None:
+    """Fused-projection views match packed inputs with direct W-1 updates."""
+    torch.manual_seed(2)
+    batch_size = 5
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -194,9 +194,6 @@ def __init__(
-        # Persistent batch-row-dense staging for the fused decode kernel's
-        # W - 1 convolution windows. It is allocated once and never reallocated.
-        self._cs_dense: Optional[torch.Tensor] = None
@@ -533,15 +530,12 @@ def forward_decode(
-        * conv windows gathered and repacked into a persistent dense
-          per-section buffer;
diff -- tests/unittest/_torch/thop/parallel/test_kda_decode.py
@@ -31,6 +31,9 @@ class KdaInputs:
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +84/-0; `tests/unittest/_torch/thop/parallel/test_kda_decode.py` modified +57/-24; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +9/-3
  - runtime: `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +19/-63; `tensorrt_llm/_torch/modules/kimi_kda/_kda_decode.py` modified +24/-16
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`, `tests/unittest/_torch/thop/parallel/test_kda_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18491 - [None][test] Add coverage for KimiLinearForCausalLM._setup_helix_mappings and related paths

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18491
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py`; associated commits `0c30590bbd8c`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +276/-0, 277 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _make_model_config, _make_cfg, _make_helix_mapping, _set_moe_attrs, touching `_make_model_config, _make_cfg, _make_helix_mapping`.
- Code diff details:
  - `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py` added +276/-0 (276 lines); hunks: -0,0 +1,276; symbols: _make_model_config, _make_cfg, _make_helix_mapping, _set_moe_attrs
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py
@@ -0,0 +1,276 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py` added +276/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_linear_helix_mappings.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18064 - [TRTLLM-15498][perf] prepare Kimi K3 prefill metadata

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18064
- Status/date: merged / 2026-09-02
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py`; associated commits `42903f9a9496`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +173/-6, 267 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-0 (3 lines); hunks: -112,6 +112,7; -1986,6 +1987,8 @@ class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[Ki...; symbols: KimiLinearForCausalLM, __init__, touching `KimiLinearForCausalLM, __init__`; `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +1/-0 (1 lines); hunks: -458,6 +458,7 @@ class KimiK3ForConditionalGeneration(KimiK25ForConditionalGe...; symbols: KimiK3ForConditionalGeneration, __init__, touching `KimiK3ForConditionalGeneration, __init__`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _prepare_kda_chunk_indices, KimiK3MambaMetadata, __init__, prepare, touching `_prepare_kda_chunk_indices, KimiK3MambaMetadata, __init__`; `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: _make_attention_metadata, test_k3_prepare_metadata_match_chunk_indices, touching `_make_attention_metadata, test_k3_prepare_metadata_match_chunk_indices`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-0 (3 lines); hunks: -112,6 +112,7; -1986,6 +1987,8 @@ class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[Ki...; symbols: KimiLinearForCausalLM, __init__
  - `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +1/-0 (1 lines); hunks: -458,6 +458,7 @@ class KimiK3ForConditionalGeneration(KimiK25ForConditionalGe...; symbols: KimiK3ForConditionalGeneration, __init__
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: _prepare_kda_chunk_indices, KimiK3MambaMetadata, __init__, prepare
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: _make_attention_metadata, test_k3_prepare_metadata_match_chunk_indices
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -112,6 +112,7 @@
+from ..modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
@@ -1986,6 +1987,8 @@ class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel, Any]):
+    mamba_metadata_cls = KimiK3MambaMetadata
diff -- tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py
@@ -458,6 +458,7 @@ class KimiK3ForConditionalGeneration(KimiK25ForConditionalGeneration):
+    mamba_metadata_cls = KimiLinearForCausalLM.mamba_metadata_cls
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py
@@ -0,0 +1,83 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+"""Kimi K3-specific recurrent-state metadata."""
+import torch
+from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
+from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
diff -- tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py
@@ -0,0 +1,56 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +3/-0; `tensorrt_llm/_torch/models/modeling_kimi_k3_vl.py` modified +1/-0; `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` added +83/-0
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` added +56/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18643 - [None][perf] Give the K3 KDA prefill conv input its layout without a repack

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18643
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`; associated commits `dec2efcf062d`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +40/-20, 90 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +40/-20 (60 lines); hunks: -28,6 +28,7; -384,6 +385,44 @@ def _sync_kda_replay_conv_window(self, layer_cache, slot_in...; symbols: _sync_kda_replay_conv_window, _project_packed_conv_input, forward_prefill, touching `_sync_kda_replay_conv_window, _project_packed_conv_input, forward_prefill`.
- Code diff details:
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +40/-20 (60 lines); hunks: -28,6 +28,7; -384,6 +385,44 @@ def _sync_kda_replay_conv_window(self, layer_cache, slot_in...; symbols: _sync_kda_replay_conv_window, _project_packed_conv_input, forward_prefill
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -28,6 +28,7 @@
+from ..mamba.fuse_elementwise_ops import extract_transpose_prefill_slice
@@ -384,6 +385,44 @@ def _sync_kda_replay_conv_window(self, layer_cache, slot_indices, conv_pool) ->
+    def _project_packed_conv_input(
+        self, x: torch.Tensor, x2d: torch.Tensor
+    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
+        """Project the block input into the ``[3D, T]`` the convolution needs.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +40/-20
- Risk and verification: Runtime changes concentrate in `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17816 - [None][feat] Kimi k3 Support bcg

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/17816
- Status/date: merged / 2026-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` and 9 files; associated commits `d773557c7557`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 10 files, +338/-133, 866 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +20/-13 (33 lines); hunks: -120,6 +120,7; -1067,6 +1068,7 @@ def __init__(; symbols: __init__, touching `__init__`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +226/-73 (299 lines); hunks: -19,6 +19,7; -27,6 +28,9; symbols: _kda_expand_fla_conv_cache, _extract_kda_extra_attrs, kda_core_inplace, KimiKDALinearAttention, touching `_kda_expand_fla_conv_cache, _extract_kda_extra_attrs, kda_core_inplace`; `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` modified +30/-24 (54 lines); hunks: -141,29 +141,33 @@ def _hook(_module: nn.Module, _inputs: tuple, _output: obj...; -193,14 +197,16 @@ def _hook(_module: nn.Module, _inputs: tuple, _output: obj...; symbols: _hook, touching `_hook`; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +23/-8 (31 lines); hunks: -211,12 +211,16 @@ def tokens(scale=0.5):; -228,13 +232,24 @@ def tokens(scale=0.5):; symbols: tokens, touching `tokens`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +20/-13 (33 lines); hunks: -120,6 +120,7; -1067,6 +1068,7 @@ def __init__(; symbols: __init__
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +226/-73 (299 lines); hunks: -19,6 +19,7; -27,6 +28,9; symbols: _kda_expand_fla_conv_cache, _extract_kda_extra_attrs, kda_core_inplace, KimiKDALinearAttention
  - `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` modified +30/-24 (54 lines); hunks: -141,29 +141,33 @@ def _hook(_module: nn.Module, _inputs: tuple, _output: obj...; -193,14 +197,16 @@ def _hook(_module: nn.Module, _inputs: tuple, _output: obj...; symbols: _hook
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +23/-8 (31 lines); hunks: -211,12 +211,16 @@ def tokens(scale=0.5):; -228,13 +232,24 @@ def tokens(scale=0.5):; symbols: tokens
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +17/-3 (20 lines); hunks: -81,11 +81,12 @@ def forward_decode(hidden_states, conv_pool, ssm_pool, slot_...; -119,6 +120,7 @@ def _decode_metadata(layer_cache: SimpleNamespace, slot_indi...; symbols: forward_decode, _decode_metadata, _prefill_metadata, test_kda_verify_matches_sequential_decode
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -120,6 +120,7 @@
+from ..utils import AuxStreamType
@@ -1067,6 +1068,7 @@ def __init__(
+        moe_aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
@@ -1117,6 +1119,7 @@ def __init__(
+            aux_stream_dict=moe_aux_stream_dict,
@@ -1507,6 +1510,7 @@ def __init__(
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py
@@ -19,6 +19,7 @@
+import weakref
@@ -27,6 +28,9 @@
+from ...model_config import ModelConfig
+from ...pyexecutor.breakable_cuda_graph import eager_on_graph, is_in_breakable_cuda_graph
+from ...utils import get_model_extra_attrs
@@ -77,6 +81,34 @@ def _kda_expand_fla_conv_cache(conv_state: torch.Tensor) -> torch.Tensor:
diff -- tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py
@@ -141,29 +141,33 @@ def _hook(_module: nn.Module, _inputs: tuple, _output: object) -> None:
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +20/-13; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +226/-73; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +3/-4
  - tests: `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py` modified +30/-24; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py` modified +23/-8; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +17/-3; `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` modified +15/-2; `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_kda_fp8_packed_prefill.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_decode_op.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_prefill_op.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_fused_verify_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18682 - [https://nvbugs/6707518][fix] Fix Kimi K3 spec dec test

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18682
- Status/date: merged / 2026-09-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/integration/defs/test_kimi_k3_specdec.py`; associated commits `709dd417f1fc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +25/-9, 71 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/kimi_k3_sa_harness.py` modified +18/-4 (22 lines); hunks: -46,7 +46,10; -295,7 +298,9 @@ def _dump_completions(path, completions, want_logprobs):; symbols: _dump_completions, main, touching `_dump_completions, main`; `tests/integration/defs/test_kimi_k3_specdec.py` modified +7/-5 (12 lines); hunks: -3,11 +3,13.
- Code diff details:
  - `tests/integration/defs/kimi_k3_sa_harness.py` modified +18/-4 (22 lines); hunks: -46,7 +46,10; -295,7 +298,9 @@ def _dump_completions(path, completions, want_logprobs):; symbols: _dump_completions, main
  - `tests/integration/defs/test_kimi_k3_specdec.py` modified +7/-5 (12 lines); hunks: -3,11 +3,13
- Key code excerpts:

```diff
diff -- tests/integration/defs/kimi_k3_sa_harness.py
@@ -46,7 +46,10 @@
-                               check at divergence; truncated models)
+                               check at divergence; truncated models).
+                               Note that we have no log probs when spec
+                               dec is enabled (the sampler does not support it),
+                               so only the baseline side is scored for that case.
@@ -295,7 +298,9 @@ def _dump_completions(path, completions, want_logprobs):
diff -- tests/integration/defs/test_kimi_k3_specdec.py
@@ -3,11 +3,13 @@
-the first MLA layer) with SA spec dec and logits-parity checking: baseline
-and spec logprobs must agree along the shared output prefix (hard failure
-on drift — the state-bug signature), while non-tie divergences only warn
-(benign reduction-order rounding flips argmax on truncated-model noise
-logits; see the harness docstring).
+the first MLA layer) with SA spec dec and logits-parity checking.
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/kimi_k3_sa_harness.py` modified +18/-4; `tests/integration/defs/test_kimi_k3_specdec.py` modified +7/-5
- Risk and verification: The diff ships test coverage in `tests/integration/defs/kimi_k3_sa_harness.py`, `tests/integration/defs/test_kimi_k3_specdec.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18699 - [None][test] add Kimi K3 feature matrix coverage

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18699
- Status/date: merged / 2026-09-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/accuracy/test_kimi3.py`; associated commits `5fa39642b9bc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 6 files, +354/-163, 693 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/accuracy/test_kimi3.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: TestKimiK3, device, test_w4a16_mxfp4, _assert_checkpoint_routing, touching `TestKimiK3, device, test_w4a16_mxfp4`.
- Code diff details:
  - `tests/integration/defs/accuracy/test_kimi3.py` added +289/-0 (289 lines); hunks: -0,0 +1,289; symbols: TestKimiK3, device, test_w4a16_mxfp4, _assert_checkpoint_routing
- Key code excerpts:

```diff
diff -- tests/integration/defs/accuracy/test_kimi3.py
@@ -0,0 +1,289 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/accuracy/test_kimi3.py` added +289/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/accuracy_core.py`, `tests/integration/defs/accuracy/test_glm52.py`, `tests/integration/defs/accuracy/test_kimi3.py`, `tests/integration/defs/accuracy/test_llm_api_pytorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18294 - [None][feat] support Kimi K3 KDA replay with KV cache manager V2

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18294
- Status/date: merged / 2026-09-07
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py`, `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` and 6 files; associated commits `4b9f3ae3de92`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 9 files, +886/-24, 1170 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +74/-0 (74 lines); hunks: -26,6 +26,7; -92,6 +93,79 @@ def forward_decode(hidden_states, conv_pool, ssm_pool, slot_i...; symbols: forward_decode, test_forward_uses_metadata_aligned_generation_state_indices, forward_prefill, forward_verify, touching `forward_decode, test_forward_uses_metadata_aligned_generation_state_indices, forward_prefill`; `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` modified +48/-0 (48 lines); hunks: -54,3 +54,51 @@ def test_k3_prepare_metadata_match_chunk_indices() -> None:; symbols: test_k3_prepare_metadata_match_chunk_indices, test_k3_prepare_materializes_stable_aligned_generation_indices, KdaCacheManager, __init__, touching `test_k3_prepare_metadata_match_chunk_indices, test_k3_prepare_materializes_stable_aligned_generation_indices, KdaCacheManager`; `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` modified +21/-0 (21 lines); hunks: -47,6 +47,13 @@ def __init__(self, max_batch_size: int, chunk_size: int, max_...; -56,6 +63,20 @@ def prepare(self, attn_metadata: AttentionMetadata) -> None:; symbols: __init__, prepare, touching `__init__, prepare`; `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +18/-0 (18 lines); hunks: -534,6 +534,24 @@ def test_zero_accepted_hint_variant(B, H):; symbols: test_zero_accepted_hint_variant, test_misaligned_state_indices_rejected_after_aligned_warmup, touching `test_zero_accepted_hint_variant, test_misaligned_state_indices_rejected_after_aligned_warmup`.
- Code diff details:
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +74/-0 (74 lines); hunks: -26,6 +26,7; -92,6 +93,79 @@ def forward_decode(hidden_states, conv_pool, ssm_pool, slot_i...; symbols: forward_decode, test_forward_uses_metadata_aligned_generation_state_indices, forward_prefill, forward_verify
  - `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` modified +48/-0 (48 lines); hunks: -54,3 +54,51 @@ def test_k3_prepare_metadata_match_chunk_indices() -> None:; symbols: test_k3_prepare_metadata_match_chunk_indices, test_k3_prepare_materializes_stable_aligned_generation_indices, KdaCacheManager, __init__
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` modified +21/-0 (21 lines); hunks: -47,6 +47,13 @@ def __init__(self, max_batch_size: int, chunk_size: int, max_...; -56,6 +63,20 @@ def prepare(self, attn_metadata: AttentionMetadata) -> None:; symbols: __init__, prepare
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +18/-0 (18 lines); hunks: -534,6 +534,24 @@ def test_zero_accepted_hint_variant(B, H):; symbols: test_zero_accepted_hint_variant, test_misaligned_state_indices_rejected_after_aligned_warmup
  - `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +6/-5 (11 lines); hunks: -376,6 +376,9 @@ def _forward_impl(; -399,13 +402,11 @@ def _forward_impl(; symbols: _forward_impl
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py
@@ -26,6 +26,7 @@
+from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
@@ -92,6 +93,79 @@ def forward_decode(hidden_states, conv_pool, ssm_pool, slot_indices, *args, **kw
+@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
+def test_forward_uses_metadata_aligned_generation_state_indices(
+    monkeypatch: pytest.MonkeyPatch,
+) -> None:
diff -- tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py
@@ -54,3 +54,51 @@ def test_k3_prepare_metadata_match_chunk_indices() -> None:
+def test_k3_prepare_materializes_stable_aligned_generation_indices() -> None:
+    class KdaCacheManager:
+        use_kda_replay_update = True
+        def __init__(self) -> None:
+            self.state_indices = torch.tensor([9, 4, 7], dtype=torch.int32, device="cuda")
+        def get_state_indices(self, request_ids: list[int], is_padding: list[bool]) -> torch.Tensor:
diff -- tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py
@@ -47,6 +47,13 @@ def __init__(self, max_batch_size: int, chunk_size: int, max_num_tokens: int) ->
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py` modified +74/-0; `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py` modified +48/-0; `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +18/-0
  - runtime: `tensorrt_llm/_torch/modules/kimi_kda/kimi_k3_mamba_metadata.py` modified +21/-0; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +6/-5; `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/executor/kv_cache/test_mamba_cache_manager.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_k3_mamba_metadata.py`, `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_verify_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18709 - [None][fix] Kimi K3: admit every trtllm-gen SiTu quant format, not just MXFP4

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18709
- Status/date: merged / 2026-09-09
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`; associated commits `3992241b737b`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 3 files, +106/-21, 169 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +42/-20 (62 lines); hunks: -118,7 +118,7; -1133,25 +1133,9 @@ def __init__(; symbols: __init__, _resolve_routed_quant_config, _check_trtllm_situ_quant, _routed_moe_model_config, touching `__init__, _resolve_routed_quant_config, _check_trtllm_situ_quant`; `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +51/-0 (51 lines); hunks: -610,6 +610,57 @@ def test_kimi_k3_moe_auto_backend_defaults_to_trtllm(archit...; symbols: test_kimi_k3_moe_auto_backend_defaults_to_trtllm, test_kimi_k3_trtllm_situ_admits_every_backend_supported_quant, test_kimi_k3_trtllm_situ_rejects_quant_without_fused_cubin, touching `test_kimi_k3_moe_auto_backend_defaults_to_trtllm, test_kimi_k3_trtllm_situ_admits_every_backend_supported_quant, test_kimi_k3_trtllm_situ_rejects_quant_without_fused_cubin`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +42/-20 (62 lines); hunks: -118,7 +118,7; -1133,25 +1133,9 @@ def __init__(; symbols: __init__, _resolve_routed_quant_config, _check_trtllm_situ_quant, _routed_moe_model_config
  - `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +51/-0 (51 lines); hunks: -610,6 +610,57 @@ def test_kimi_k3_moe_auto_backend_defaults_to_trtllm(archit...; symbols: test_kimi_k3_moe_auto_backend_defaults_to_trtllm, test_kimi_k3_trtllm_situ_admits_every_backend_supported_quant, test_kimi_k3_trtllm_situ_rejects_quant_without_fused_cubin
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -118,7 +118,7 @@
-from ..moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
+from ..moe.fused_moe import ConfigurableMoE, SiTuActivation, TRTLLMGenFusedMoE, create_moe
@@ -1133,25 +1133,9 @@ def __init__(
-        # trtllm-gen ships SiTu cubins for exactly one dtype combination
-        # (``Bmm_MxE4m3_MxE2m1MxE4m3`` = MXFP8 act x MXFP4 weight) and has no
-        # standalone SiTu activation kernel to fall back on, so a non-MXFP4
diff -- tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py
@@ -610,6 +610,57 @@ def test_kimi_k3_moe_auto_backend_defaults_to_trtllm(architecture):
+# ---------------------------------------------------------------------------
+# The K3 model layer and TRTLLMGenFusedMoE both have to know which routed-expert
+# formats trtllm-gen has a fused SiTu FC1 cubin for. They disagreed once: the
+# model's copy was written when MXFP4 was the only drop (#17865) and #17940 then
+# shipped the NVFP4 cubins and updated only the backend, so for a week an NVFP4
+# K3 checkpoint raised at construction on a path that the kernels supported.
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +42/-20
  - tests: `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +51/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18728 - [None][fix] share Kimi auxiliary streams

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18728
- Status/date: merged / 2026-09-09
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py`, `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py`, `tests/unittest/_torch/moe/test_kimi_k3_mlp.py`; associated commits `eabb0c8a1513`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +106/-30, 288 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +22/-25 (47 lines); hunks: -1067,8 +1067,7 @@ def __init__(; -1119,7 +1118,7 @@ def __init__(; symbols: __init__, _routed_output, touching `__init__, _routed_output`; `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: test_kimi_linear_model_builds_shared_aux_stream_registry, _FakeDecoderLayer, __init__, touching `test_kimi_linear_model_builds_shared_aux_stream_registry, _FakeDecoderLayer, __init__`; `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` modified +25/-5 (30 lines); hunks: -26,10 +26,21; -207,7 +218,6 @@ def test_kimi_k3_shared_expert_parallel_construction(; symbols: _make_aux_stream_dict, _UnfusedKimiMLP, test_kimi_k3_shared_expert_parallel_construction, __init__, touching `_make_aux_stream_dict, _UnfusedKimiMLP, test_kimi_k3_shared_expert_parallel_construction`; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -159,6 +160,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +22/-25 (47 lines); hunks: -1067,8 +1067,7 @@ def __init__(; -1119,7 +1118,7 @@ def __init__(; symbols: __init__, _routed_output
  - `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py` added +56/-0 (56 lines); hunks: -0,0 +1,56; symbols: test_kimi_linear_model_builds_shared_aux_stream_registry, _FakeDecoderLayer, __init__
  - `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` modified +25/-5 (30 lines); hunks: -26,10 +26,21; -207,7 +218,6 @@ def test_kimi_k3_shared_expert_parallel_construction(; symbols: _make_aux_stream_dict, _UnfusedKimiMLP, test_kimi_k3_shared_expert_parallel_construction, __init__
  - `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +3/-0 (3 lines); hunks: -20,6 +20,7; -159,6 +160,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -1067,8 +1067,7 @@ def __init__(
-        aux_stream: Optional[torch.cuda.Stream] = None,
-        moe_aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
+        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
@@ -1119,7 +1118,7 @@ def __init__(
-            aux_stream_dict=moe_aux_stream_dict,
+            aux_stream_dict=aux_stream_dict,
diff -- tests/unittest/_torch/modeling/test_kimi_linear_modeling.py
@@ -0,0 +1,56 @@
+# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+# SPDX-License-Identifier: Apache-2.0
+import pytest
+import torch
+from torch import nn
+from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
diff -- tests/unittest/_torch/moe/test_kimi_k3_mlp.py
@@ -26,10 +26,21 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +22/-25; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +3/-0
  - tests: `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py` added +56/-0; `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` modified +25/-5
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_linear_modeling.py`, `tests/unittest/_torch/moe/test_kimi_k3_mlp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19088 - [None][test] Supply auxiliary streams in Kimi K3 NVFP4 regression

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19088
- Status/date: merged / 2026-09-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`; associated commits `a3848cc0d8ec`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +7/-2, 23 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +7/-2 (9 lines); hunks: -49,7 +49,7; -633,7 +633,12 @@ def test_kimi_k3_trtllm_accepts_nvfp4_routed_experts():; symbols: test_kimi_k3_trtllm_accepts_nvfp4_routed_experts, touching `test_kimi_k3_trtllm_accepts_nvfp4_routed_experts`.
- Code diff details:
  - `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +7/-2 (9 lines); hunks: -49,7 +49,7; -633,7 +633,12 @@ def test_kimi_k3_trtllm_accepts_nvfp4_routed_experts():; symbols: test_kimi_k3_trtllm_accepts_nvfp4_routed_experts
- Key code excerpts:

```diff
diff -- tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py
@@ -49,7 +49,7 @@
-from tensorrt_llm._torch.utils import ActType_TrtllmGen
+from tensorrt_llm._torch.utils import ActType_TrtllmGen, AuxStreamType
@@ -633,7 +633,12 @@ def test_kimi_k3_trtllm_accepts_nvfp4_routed_experts():
-    runtime = KimiK3MoERuntime(model_config, cfg, layer_idx=0)
+    runtime = KimiK3MoERuntime(
+        model_config,
```

- Extracted files (not manually reviewed):
  - tests: `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +7/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18461 - [TRTLLM-16020][test] Add Kimi K3 GSM8K accuracy tests to GB300 multi-node post-merge CI

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/18461
- Status/date: merged / 2026-09-15
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/accuracy/test_kimi3.py`; associated commits `a494ef678a97`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +27/-1, 60 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/accuracy/test_kimi3.py` modified +7/-1 (8 lines); hunks: -54,7 +54,8 @@ class TestKimiK3(LlmapiAccuracyTestHarness):; -66,6 +67,11 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.Mon...; symbols: TestKimiK3, device, test_w4a16_mxfp4, touching `TestKimiK3, device, test_w4a16_mxfp4`.
- Code diff details:
  - `tests/integration/defs/accuracy/test_kimi3.py` modified +7/-1 (8 lines); hunks: -54,7 +54,8 @@ class TestKimiK3(LlmapiAccuracyTestHarness):; -66,6 +67,11 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.Mon...; symbols: TestKimiK3, device, test_w4a16_mxfp4
- Key code excerpts:

```diff
diff -- tests/integration/defs/accuracy/test_kimi3.py
@@ -54,7 +54,8 @@ class TestKimiK3(LlmapiAccuracyTestHarness):
-    # platform selection, not by this marker.
+    # platform selection and by the CI stage's gb300-only gpu wildcard,
+    # not by this marker.
@@ -66,6 +67,11 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
+        The baseline and sa modes run post-merge in the GB300 16-GPU 4-node CI
+        stage (test-db list l0_gb300_multi_nodes_node4_gpu16.yml); all four
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/accuracy/test_kimi3.py` modified +7/-1
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/test_kimi3.py`, `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_node4_gpu16.yml`, `tests/integration/test_lists/waives.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19191 - [TRTLLM-16188][test] Add Kimi K3 short-context perf cases and fix stale disagg note

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19191
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`; associated commits `166b51894fc4`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +4/-1, 19 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +1/-1 (2 lines); hunks: -39,7 +39,7 @@ The checkpoint and the configuration file must live on a share....
- Code diff details:
  - `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +1/-1 (2 lines); hunks: -39,7 +39,7 @@ The checkpoint and the configuration file must live on a share...
- Key code excerpts:

```diff
diff -- docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md
@@ -39,7 +39,7 @@ The checkpoint and the configuration file must live on a shared filesystem visib
-* **Speculative decoding and disaggregated serving are not yet available** for Kimi K3; support is under development. See the "Current limitations" section of `examples/kimi_k3/RE
+* **Disaggregated serving and suffix-automaton speculative decoding are available.** Disaggregated serving is validated end-to-end on GB300 with matched DEP16 context/generation s
```

- Extracted files (not manually reviewed):
  - docs: `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md` modified +1/-1
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/qa/llm_perf_core.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19084 - [https://nvbugs/6656598][tests] Deprecate K2 E2E test for K3

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19084
- Status/date: merged / 2026-09-16
- Trace source: `git log --name-only -- <model-files>` found it through `tests/integration/defs/accuracy/test_kimi3.py`; associated commits `6882e320e8bc`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 4 files, +60/-1, 91 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tests/integration/defs/accuracy/test_kimi3.py` modified +50/-0 (50 lines); hunks: -34,6 +34,7; -154,6 +155,55 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.M...; symbols: test_w4a16_mxfp4, test_gpqa_diamond_w4a16_mxfp4, _assert_checkpoint_routing, touching `test_w4a16_mxfp4, test_gpqa_diamond_w4a16_mxfp4, _assert_checkpoint_routing`.
- Code diff details:
  - `tests/integration/defs/accuracy/test_kimi3.py` modified +50/-0 (50 lines); hunks: -34,6 +34,7; -154,6 +155,55 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.M...; symbols: test_w4a16_mxfp4, test_gpqa_diamond_w4a16_mxfp4, _assert_checkpoint_routing
- Key code excerpts:

```diff
diff -- tests/integration/defs/accuracy/test_kimi3.py
@@ -34,6 +34,7 @@
+    GPQADiamond,
@@ -154,6 +155,55 @@ def test_w4a16_mxfp4(self, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
+    @skip_pre_blackwell
+    @pytest.mark.skip_less_mpi_world_size(16)
+    @pytest.mark.skip_less_device_memory(200000)
+    def test_gpqa_diamond_w4a16_mxfp4(self) -> None:
```

- Extracted files (not manually reviewed):
  - tests: `tests/integration/defs/accuracy/test_kimi3.py` modified +50/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/accuracy/references/gpqa_diamond.yaml`, `tests/integration/defs/accuracy/test_kimi3.py`, `tests/integration/test_lists/qa/llm_function_multinode.txt`, `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_node4_gpu16.yml`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19277 - [None][fix] Make MoonViT replication helix-aware

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19277
- Status/date: merged / 2026-09-18
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_k25.py`, `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py`; associated commits `34495fa8bf68`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 2 files, +25/-4, 63 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +13/-2 (15 lines); hunks: -372,19 +372,30 @@ def _vision_requires_replication(model_config: ModelConfig...; symbols: _vision_requires_replication, _get_vision_tp_mapping, touching `_vision_requires_replication, _get_vision_tp_mapping`; `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py` modified +12/-2 (14 lines); hunks: -84,10 +84,12 @@ def test_other_model_type_with_subconfigs(self):; -108,6 +110,14 @@ def test_k25_16_heads_under_tp8_shards(self):; symbols: test_other_model_type_with_subconfigs, _vision_model_config, test_k25_16_heads_under_tp8_shards, test_attention_dp_always_replicates, touching `test_other_model_type_with_subconfigs, _vision_model_config, test_k25_16_heads_under_tp8_shards`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +13/-2 (15 lines); hunks: -372,19 +372,30 @@ def _vision_requires_replication(model_config: ModelConfig...; symbols: _vision_requires_replication, _get_vision_tp_mapping
  - `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py` modified +12/-2 (14 lines); hunks: -84,10 +84,12 @@ def test_other_model_type_with_subconfigs(self):; -108,6 +110,14 @@ def test_k25_16_heads_under_tp8_shards(self):; symbols: test_other_model_type_with_subconfigs, _vision_model_config, test_k25_16_heads_under_tp8_shards, test_attention_dp_always_replicates
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_k25.py
@@ -372,19 +372,30 @@ def _vision_requires_replication(model_config: ModelConfig, num_heads: int) -> b
+    # Helix carries its parallelism in cp with tp_size=1, so a tp-only test
+    # never trips; the tower has no context-parallel form, so any cp > 1
+    # must replicate.
+    if mapping.cp_size > 1:
+        return True
+    # Fold every parallel dimension (incl. helix cp) into pp so each rank
diff -- tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py
@@ -84,10 +84,12 @@ def test_other_model_type_with_subconfigs(self):
-def _vision_model_config(tp_size, enable_attention_dp):
+def _vision_model_config(tp_size, enable_attention_dp, cp_size=1):
-        mapping=SimpleNamespace(tp_size=tp_size, enable_attention_dp=enable_attention_dp)
+        mapping=SimpleNamespace(
+            tp_size=tp_size, enable_attention_dp=enable_attention_dp, cp_size=cp_size
+        )
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_k25.py` modified +13/-2
  - tests: `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py` modified +12/-2
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/modeling/test_kimi_k3_config_routing.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19182 - [None][feat] Kimi K3 attention-residual RMSNorm fusion + KDA beta-cache alignment

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19182
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `cpp/tensorrt_llm/kernels/kimiK3AttnRes/CMakeLists.txt`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu`, `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.h` and 16 files; associated commits `5958f7c70046`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 24 files, +4476/-219, 5488 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +373/-48 (421 lines); hunks: -48,16 +48,16; -119,6 +119,7; symbols: _is_mla_layer, _read_attn_res_topology, _read_fused_attn_res_max_tokens, _persistent_attn_res_applicable, touching `_is_mla_layer, _read_attn_res_topology, _read_fused_attn_res_max_tokens`; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu` added +1223/-0 (1223 lines); hunks: -0,0 +1,1223; `tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py` added +598/-0 (598 lines); hunks: -0,0 +1,598; symbols: _has_supported_gpu, _production_rms_norm, _make_inputs, _unfused_reference, touching `_has_supported_gpu, _production_rms_norm, _make_inputs`; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` modified +456/-85 (541 lines); hunks: -30,6 +30,7; -72,6 +73,14 @@ __inline__ __device__ float block_reduce_sum(float val, float....
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +373/-48 (421 lines); hunks: -48,16 +48,16; -119,6 +119,7; symbols: _is_mla_layer, _read_attn_res_topology, _read_fused_attn_res_max_tokens, _persistent_attn_res_applicable
  - `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu` added +1223/-0 (1223 lines); hunks: -0,0 +1,1223
  - `tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py` added +598/-0 (598 lines); hunks: -0,0 +1,598; symbols: _has_supported_gpu, _production_rms_norm, _make_inputs, _unfused_reference
  - `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` modified +456/-85 (541 lines); hunks: -30,6 +30,7; -72,6 +73,14 @@ __inline__ __device__ float block_reduce_sum(float val, float...
  - `tests/microbenchmarks/kimi_k3_attn_res_add_rmsnorm.py` added +338/-0 (338 lines); hunks: -0,0 +1,338; symbols: CaseInputs, _parse_candidates, _make_inputs, _attn_res
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -48,16 +48,16 @@
-``routed_expert_norm`` / ``routed_expert_up_proj``, which are
-nonlinear/linear layers applied to the full sum). When attention DP is off,
-the shared experts use standard MLP TP over the model TP group: gate/up are
-column-sharded and down is row-sharded. Direct MoE-TP combines the shared
-hidden-width partial and routed latent partial into one all-reduce after the
-two streams join, then splits them before the routed norm/up projection.
diff -- cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu
@@ -0,0 +1,1223 @@
+/*
+ * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
+ *
+ * Licensed under the Apache License, Version 2.0 (the "License");
+ * you may not use this file except in compliance with the License.
+ * You may obtain a copy of the License at
diff -- tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py
@@ -0,0 +1,598 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +373/-48; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwdPersistentFused.cu` added +1223/-0; `cpp/tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.cu` modified +456/-85
  - tests: `tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py` added +598/-0; `tests/microbenchmarks/kimi_k3_attn_res_add_rmsnorm.py` added +338/-0; `tests/unittest/_torch/moe/test_kimi_k3_mlp.py` modified +274/-4; `tests/unittest/_torch/modules/kimi_kda/test_kimi_kda_bf16_state_pool.py` added +219/-0; `tests/unittest/_torch/modules/kimi_kda/test_kda_host_metadata.py` added +181/-0
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/integration/test_lists/test-db/l0_gb300_multi_gpus.yml`, `tests/microbenchmarks/kimi_k3_attn_res_add_rmsnorm.py`, `tests/unittest/_torch/executor/kv_cache/test_mamba_cache_manager.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19179 - [None][refactor] Clean up Kimi checkpoint FP8 attention loading

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19179
- Status/date: merged / 2026-09-19
- Trace source: `git log --name-only -- <model-files>` found it through `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `examples/kimi_k3/eval_extra_llm_options_nvfp4_dep16.yaml`, `examples/kimi_k3/perf_sweep/acc_sweep.sbatch`, `examples/kimi_k3/perf_sweep/perf_sweep.sbatch`, `examples/kimi_k3/run_eval_kimi_k3.sbatch` and 16 files; associated commits `347f5f172f37`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 19 files, +1683/-962, 3229 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +395/-509 (904 lines); hunks: -85,6 +85,7; -145,57 +146,15; symbols: KimiK3MoEGate, forward, _resolve_fp8_weight_read_gates, _resolve_kimi_situ_betas, touching `KimiK3MoEGate, forward, _resolve_fp8_weight_read_gates`; `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +427/-1 (428 lines); hunks: -1,24 +1,37; -182,3 +195,416 @@ def test_helix_cp_v_b_shard_noop_without_cp():; symbols: test_checkpoint_plan_preserves_external_attention_names, _PlanHarness, _trunk_parameters, test_helix_cp_v_b_shard_noop_without_cp, touching `test_checkpoint_plan_preserves_external_attention_names, _PlanHarness, _trunk_parameters`; `tests/unittest/_torch/modeling/kimi_k3_mla_reference.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: KimiRMSNorm, __init__, forward, repeat_kv, touching `KimiRMSNorm, __init__, forward`; `tests/unittest/_torch/modeling/test_kimi_k3_checkpoint.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _config, load_mla_checkpoint, _weights, test_checkpoint_loads_fp8_mla_values, touching `_config, load_mla_checkpoint, _weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +395/-509 (904 lines); hunks: -85,6 +85,7; -145,57 +146,15; symbols: KimiK3MoEGate, forward, _resolve_fp8_weight_read_gates, _resolve_kimi_situ_betas
  - `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +427/-1 (428 lines); hunks: -1,24 +1,37; -182,3 +195,416 @@ def test_helix_cp_v_b_shard_noop_without_cp():; symbols: test_checkpoint_plan_preserves_external_attention_names, _PlanHarness, _trunk_parameters, test_helix_cp_v_b_shard_noop_without_cp
  - `tests/unittest/_torch/modeling/kimi_k3_mla_reference.py` added +234/-0 (234 lines); hunks: -0,0 +1,234; symbols: KimiRMSNorm, __init__, forward, repeat_kv
  - `tests/unittest/_torch/modeling/test_kimi_k3_checkpoint.py` added +223/-0 (223 lines); hunks: -0,0 +1,223; symbols: _config, load_mla_checkpoint, _weights, test_checkpoint_loads_fp8_mla_values
  - `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +0/-214 (214 lines); hunks: -122,14 +122,6 @@ def test_kimi_situ_betas_must_be_positive(situ_beta, situ_l...; -1876,212 +1868,6 @@ def test_nvfp4_streaming_drains_staging_per_expert():; symbols: test_kimi_situ_betas_must_be_positive, test_clear_checkpoint_fp8_pairs_releases_unconsumed_stashes, _init_block_weights, test_nvfp4_streaming_drains_staging_per_expert
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -85,6 +85,7 @@
+import weakref
@@ -145,57 +146,15 @@
+    ".self_attn.mixer.k_b_proj_trans_scale",
+    ".self_attn.mixer.v_b_proj_scale",
+    ".self_attn.mixer.k_b_proj_trans_dequant",
+    ".self_attn.mixer.v_b_proj_dequant",
diff -- tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py
@@ -1,24 +1,37 @@
-"""Checkpoint-name and weight-shard tests for the Kimi Linear model."""
+"""Checkpoint naming, weight sharding, and FP8 loading for the Kimi Linear model."""
+import copy
+from unittest.mock import patch
+from torch import nn
+from tensorrt_llm._torch.model_config import ModelConfig
diff -- tests/unittest/_torch/modeling/kimi_k3_mla_reference.py
@@ -0,0 +1,234 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +395/-509; `tensorrt_llm/_torch/modules/kimi_kda/kimi_kda_mixer.py` modified +93/-47; `tensorrt_llm/_torch/modules/kimi_k3_mla/kimi_k3_mla_attention.py` modified +31/-28
  - tests: `tests/unittest/_torch/modeling/test_kimi_linear_checkpoint.py` modified +427/-1; `tests/unittest/_torch/modeling/kimi_k3_mla_reference.py` added +234/-0; `tests/unittest/_torch/modeling/test_kimi_k3_checkpoint.py` added +223/-0; `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +0/-214; `tests/unittest/_torch/modeling/test_kimi_k3_mla.py` added +189/-0
- Risk and verification: The diff ships test coverage in `tests/integration/defs/perf/test_perf.py`, `tests/integration/test_lists/test-db/l0_b200.yml`, `tests/unittest/_torch/modeling/kimi_k3_mla_reference.py`, `tests/unittest/_torch/modeling/test_kimi_k3_checkpoint.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19040 - [TRTLLM-16564][feat] MLA-backboned standalone DSpark drafter (Inferact/Kimi-K3-DSpark)

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19040
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py`, `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py`; associated commits `129655ff5674`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 31 files, +4224/-302, 5796 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +54/-2 (56 lines); hunks: -295,6 +295,44 @@ def _is_mla_layer(cfg, layer_idx: int) -> bool:; -1748,7 +1786,11 @@ def forward(; symbols: _is_mla_layer, forward, __init__, touching `_is_mla_layer, forward, __init__`; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` modified +980/-24 (1004 lines); hunks: -19,6 +19,7; -34,8 +35,34; symbols: _has_absorbed_mla_kernel, test_swa_window_conventions, _build_drafter, _rms, touching `_has_absorbed_mla_kernel, test_swa_window_conventions, _build_drafter`; `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` modified +88/-69 (157 lines); hunks: -53,6 +53,7; -320,30 +321,50 @@ def _fits_32bit_stride(tensor: torch.Tensor) -> bool:; symbols: _fits_32bit_stride, _from_dlpack_arg, _dlpack_arg, _beta_cache_assumed_align, touching `_fits_32bit_stride, _from_dlpack_arg, _dlpack_arg`; `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +124/-15 (139 lines); hunks: -260,7 +260,7 @@ def cpu_reference(data):; -275,7 +275,7 @@ def cute_run(data, zero_accepted_hint=False):; symbols: cpu_reference, cute_run, test_zero_accepted_hint_variant, touching `cpu_reference, cute_run, test_zero_accepted_hint_variant`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +54/-2 (56 lines); hunks: -295,6 +295,44 @@ def _is_mla_layer(cfg, layer_idx: int) -> bool:; -1748,7 +1786,11 @@ def forward(; symbols: _is_mla_layer, forward, __init__
  - `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` modified +980/-24 (1004 lines); hunks: -19,6 +19,7; -34,8 +35,34; symbols: _has_absorbed_mla_kernel, test_swa_window_conventions, _build_drafter, _rms
  - `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` modified +88/-69 (157 lines); hunks: -53,6 +53,7; -320,30 +321,50 @@ def _fits_32bit_stride(tensor: torch.Tensor) -> bool:; symbols: _fits_32bit_stride, _from_dlpack_arg, _dlpack_arg, _beta_cache_assumed_align
  - `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +124/-15 (139 lines); hunks: -260,7 +260,7 @@ def cpu_reference(data):; -275,7 +275,7 @@ def cute_run(data, zero_accepted_hint=False):; symbols: cpu_reference, cute_run, test_zero_accepted_hint_variant
  - `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py` modified +133/-0 (133 lines); hunks: -462,3 +462,136 @@ def __call__(self, hidden_states, block_residual, num_snap...; symbols: __call__, test_aux_capture_taps_the_selected_stream, test_aux_capture_tail_follows_the_same_switch, _Layer
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -295,6 +295,44 @@ def _is_mla_layer(cfg, layer_idx: int) -> bool:
+KIMI_K3_AUX_ATTN_RES_STREAM_ENV = "KIMI_K3_AUX_ATTN_RES_STREAM"
+"""Which residual-stream value the DFlash/DSpark hidden-state tap captures.
+``1`` (default) captures the pre-norm attn_res mixture -- the value the next
+consumer actually reads. ``0`` captures the raw running prefix sum instead.
+Both conventions exist in the wild and a drafter distilled against one scores
+lower on the other with nothing raised, so this is a property of the DRAFTER
diff -- tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py
@@ -19,6 +19,7 @@
+import math
@@ -34,8 +35,34 @@
+needs_cuda = pytest.mark.skipif(
+    not torch.cuda.is_available(), reason="drafter module construction needs CUDA"
+)
+def _has_absorbed_mla_kernel() -> bool:
diff -- tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py
@@ -53,6 +53,7 @@
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +54/-2; `tensorrt_llm/_torch/custom_ops/cute_dsl_kimi_k3_kda_mtp_ops.py` modified +88/-69
  - tests: `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dspark_semantics.py` modified +980/-24; `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py` modified +124/-15; `tests/unittest/_torch/speculative/hw_agnostic/test_kimi_k3_dflash_scaffold.py` modified +133/-0
- Risk and verification: The diff ships test coverage in `tests/unittest/_torch/executor/kv_cache/test_kv_cache_budget_split.py`, `tests/unittest/_torch/modules/kimi_kda/test_kda_mtp_decode_cute_parity.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_eplb_config.py`, `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_heads.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19003 - [None][feat] Kimi K3: unlock the CUTEDSL MoE backend for NVFP4 SiTU

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19003
- Status/date: merged / 2026-09-21
- Trace source: `git log --name-only -- <model-files>` found it through `tensorrt_llm/_torch/models/modeling_kimi_linear.py`, `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`; associated commits `77628ee84805`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 8 files, +589/-60, 956 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +27/-5 (32 lines); hunks: -1073,6 +1073,16 @@ def __init__(; -1129,12 +1139,20 @@ def __init__(; symbols: __init__, _check_trtllm_situ_quant, _routed_moe_model_config, touching `__init__, _check_trtllm_situ_quant, _routed_moe_model_config`; `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +429/-29 (458 lines); hunks: -122,6 +122,45 @@ def test_kimi_situ_betas_must_be_positive(situ_beta, situ_l...; -583,6 +622,116 @@ def test_kimi_k3_routed_config_logs_megamoe_capacity_overr...; symbols: test_kimi_situ_betas_must_be_positive, test_cutedsl_kernel_rejects_unusable_situ_betas, _init_block_weights, test_kimi_k3_routed_config_logs_megamoe_capacity_override, touching `test_kimi_situ_betas_must_be_positive, test_cutedsl_kernel_rejects_unusable_situ_betas, _init_block_weights`.
- Code diff details:
  - `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +27/-5 (32 lines); hunks: -1073,6 +1073,16 @@ def __init__(; -1129,12 +1139,20 @@ def __init__(; symbols: __init__, _check_trtllm_situ_quant, _routed_moe_model_config
  - `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +429/-29 (458 lines); hunks: -122,6 +122,45 @@ def test_kimi_situ_betas_must_be_positive(situ_beta, situ_l...; -583,6 +622,116 @@ def test_kimi_k3_routed_config_logs_megamoe_capacity_overr...; symbols: test_kimi_situ_betas_must_be_positive, test_cutedsl_kernel_rejects_unusable_situ_betas, _init_block_weights, test_kimi_k3_routed_config_logs_megamoe_capacity_override
- Key code excerpts:

```diff
diff -- tensorrt_llm/_torch/models/modeling_kimi_linear.py
@@ -1073,6 +1073,16 @@ def __init__(
+        """Build the routed experts and the shared expert for one MoE layer.
+        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
+        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
+        that ``ModelConfig`` does not carry.
+        ``aux_stream_dict`` is shared across every layer of the model, so the
+        streams reached through it are borrowed and must not be synchronized
diff -- tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py
@@ -122,6 +122,45 @@ def test_kimi_situ_betas_must_be_positive(situ_beta, situ_linear_beta):
+@pytest.mark.parametrize(
+    "situ_beta,situ_linear_beta",
+    [
+        (float("nan"), 25.0),
+        (4.0, float("nan")),
+        (float("inf"), 25.0),
```

- Extracted files (not manually reviewed):
  - runtime: `tensorrt_llm/_torch/models/modeling_kimi_linear.py` modified +27/-5
  - tests: `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py` modified +429/-29
- Risk and verification: The diff ships test coverage in `tests/integration/test_lists/test-db/l0_b300.yml`, `tests/unittest/_torch/moe/test_kimi_k3_situ_moe.py`, `tests/unittest/_torch/moe/test_moe_backend.py`, `tests/unittest/_torch/thop/parallel/test_cute_dsl_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19784 - [https://nvbugs/6783973][doc] Stop listing supported SA speculation as a Kimi K3 limitation

- Link: https://github.com/NVIDIA/TensorRT-LLM/pull/19784
- Status/date: merged / 2026-10-02
- Trace source: `git log --name-only -- <model-files>` found it through `examples/kimi_k3/README.md`; associated commits `f1a48e6c47b6`
- Extracted diff scope (not a manual audit): GitHub Pull Request files API returned 1 files, +27/-4, 49 readable patch lines; API patches may be truncated or absent; inspect the full diff before using this entry as optimization evidence.
- Motivation: Manual review pending; the PR title and file inventory are discovery evidence, not an inferred rationale.
- Key implementation inventory (machine-extracted): `examples/kimi_k3/README.md` modified +27/-4 (31 lines); hunks: -115,8 +115,9 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \; -201,7 +202,21 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \.
- Code diff details:
  - `examples/kimi_k3/README.md` modified +27/-4 (31 lines); hunks: -115,8 +115,9 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \; -201,7 +202,21 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
- Key code excerpts:

```diff
diff -- examples/kimi_k3/README.md
@@ -115,8 +115,9 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
-This selects `eval_extra_llm_options_sa.yaml` (see Current limitations
-below for what SA changes) and logs a speculative-decoding acceptance
+This selects `eval_extra_llm_options_sa.yaml` (see the suffix-automaton
+section and its restrictions under Current limitations below for what SA
+changes) and logs a speculative-decoding acceptance
@@ -201,7 +202,21 @@ sbatch examples/kimi_k3/run_eval_kimi_k3.sbatch \
```

- Extracted files (not manually reviewed):
  - docs: `examples/kimi_k3/README.md` modified +27/-4
- Risk and verification: This is mostly docs/examples in `examples/kimi_k3/README.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
