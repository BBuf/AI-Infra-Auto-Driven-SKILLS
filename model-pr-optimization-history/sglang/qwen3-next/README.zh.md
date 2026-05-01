# sglang Qwen3 Next 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `4197c55968`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `python/sglang/srt/configs/qwen3_next.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#11585](https://github.com/sgl-project/sglang/pull/11585), [#12525](https://github.com/sgl-project/sglang/pull/12525) |
| `python/sglang/srt/models/qwen3_next.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#10322](https://github.com/sgl-project/sglang/pull/10322), [#10379](https://github.com/sgl-project/sglang/pull/10379), [#10622](https://github.com/sgl-project/sglang/pull/10622), [#11969](https://github.com/sgl-project/sglang/pull/11969), [#12525](https://github.com/sgl-project/sglang/pull/12525), [#13081](https://github.com/sgl-project/sglang/pull/13081), [#14607](https://github.com/sgl-project/sglang/pull/14607), [#16164](https://github.com/sgl-project/sglang/pull/16164), [#17016](https://github.com/sgl-project/sglang/pull/17016), [#17373](https://github.com/sgl-project/sglang/pull/17373), [#17613](https://github.com/sgl-project/sglang/pull/17613), ... (21 total) |
| `python/sglang/srt/models/qwen3_next_mtp.py` | [#10233](https://github.com/sgl-project/sglang/pull/10233), [#10322](https://github.com/sgl-project/sglang/pull/10322), [#10392](https://github.com/sgl-project/sglang/pull/10392) |
| `test/registered/4-gpu-models/test_qwen3_next_models.py` | 无直接 PR 号提交 |
| `test/registered/4-gpu-models/test_qwen3_next_models_mtp.py` | 无直接 PR 号提交 |
| `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_auto_qwen3_next.py` | 无直接 PR 号提交 |
| `test/registered/ascend/basic_function/parallel_strategy/expert_parallelism/test_npu_deepep_low_latency_qwen3_next.py` | 无直接 PR 号提交 |
| `test/registered/core/test_qwen3_next_deterministic.py` | 无直接 PR 号提交 |
| `test/registered/models/test_qwen3_next_models_fp4.py` | [#17627](https://github.com/sgl-project/sglang/pull/17627) |

## PR 覆盖总览

- git 追溯 PR 数: 23
- 原文档显式引用补充 PR 数: 37
- 当前文档总 PR 数: 60
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-09-11 | [#10233](https://github.com/sgl-project/sglang/pull/10233) | merged | Qwen3-Next support | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-11 | [#10322](https://github.com/sgl-project/sglang/pull/10322) | merged | [bugfix] fix norm type error in qwen3_next model | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-12 | [#10379](https://github.com/sgl-project/sglang/pull/10379) | merged | Support Qwen3-Next on Ascend NPU | `python/sglang/srt/models/qwen3_next.py` |
| 2025-09-13 | [#10392](https://github.com/sgl-project/sglang/pull/10392) | merged | [Fix] Support qwen3-next MTP+DP | `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2025-09-16 | [#10466](https://github.com/sgl-project/sglang/pull/10466) | merged | feat: update support for qwen3next model | `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-09-18 | [#10624](https://github.com/sgl-project/sglang/pull/10624) | merged | update deepep version for qwen3-next deepep moe | `docker/Dockerfile`, `scripts/ci/ci_install_deepep.sh` |
| 2025-09-18 | [#10622](https://github.com/sgl-project/sglang/pull/10622) | merged | support qwen3-next-fp8 deepep | `python/sglang/srt/models/qwen3_next.py` |
| 2025-09-19 | [#10657](https://github.com/sgl-project/sglang/pull/10657) | open | feat: add eagle3 support for qwen3-next model | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-10-15 | [#11585](https://github.com/sgl-project/sglang/pull/11585) | merged | Clean up some Qwen3-Next and deterministic code | `python/sglang/srt/configs/qwen3_next.py` |
| 2025-10-16 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py` |
| 2025-10-21 | [#11487](https://github.com/sgl-project/sglang/pull/11487) | merged | init support for KTransformers Heterogeneous Computing | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-30 | [#11969](https://github.com/sgl-project/sglang/pull/11969) | merged | [NPU] bugfix for Qwen3-Next and performance update | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-06 | [#12508](https://github.com/sgl-project/sglang/pull/12508) | merged | [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup | `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2025-11-08 | [#12892](https://github.com/sgl-project/sglang/pull/12892) | open | [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup | `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/fla/fused_recurrent.py` |
| 2025-11-21 | [#13708](https://github.com/sgl-project/sglang/pull/13708) | merged | [Fix] Qwen3Next lmhead dtype | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-25 | [#13081](https://github.com/sgl-project/sglang/pull/13081) | merged | Support piecewise cuda graph for Qwen3-next | `python/sglang/srt/models/qwen3_next.py` |
| 2025-11-26 | [#13924](https://github.com/sgl-project/sglang/pull/13924) | closed | [performance]Qwen3 Next kernel performance optimize | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-11-26 | [#13964](https://github.com/sgl-project/sglang/pull/13964) | open | [Performance]Qwen3 Next kernel performance optimize | `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` |
| 2025-12-05 | [#14502](https://github.com/sgl-project/sglang/pull/14502) | open | [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2025-12-13 | [#14855](https://github.com/sgl-project/sglang/pull/14855) | merged | Clean up GDN Init | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-03 | [#16164](https://github.com/sgl-project/sglang/pull/16164) | merged | [NPU] Adapt qwen3-next W8A8 on NPU | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-05 | [#16488](https://github.com/sgl-project/sglang/pull/16488) | open | Two-Batch Overlap (TBO) support to Qwen3-Next Models | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-01-10 | [#16863](https://github.com/sgl-project/sglang/pull/16863) | merged | tiny refactor pcg split op registration | `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/compilation/compilation_config.py` |
| 2026-01-16 | [#17016](https://github.com/sgl-project/sglang/pull/17016) | merged | [bugfix] fix qwen3-next alt_stream none issue | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-18 | [#15631](https://github.com/sgl-project/sglang/pull/15631) | merged | [jit-kernel] Add CuTe DSL GDN Decode Kernel | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` |
| 2026-01-20 | [#17403](https://github.com/sgl-project/sglang/pull/17403) | merged | Use attn_tp_group for all reduce in token embedding | `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-01-22 | [#17373](https://github.com/sgl-project/sglang/pull/17373) | merged | refactor Qwen3-Next with a new RadixLinearAttention | `python/sglang/srt/models/qwen3_next.py` |
| 2026-01-24 | [#17570](https://github.com/sgl-project/sglang/pull/17570) | merged | Use attn tp group in embedding for more models | `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py` |
| 2026-01-27 | [#17660](https://github.com/sgl-project/sglang/pull/17660) | merged | [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention | `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py` |
| 2026-01-30 | [#12525](https://github.com/sgl-project/sglang/pull/12525) | merged | [CPU] Optimize Qwen3-next model on CPU | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py` |
| 2026-01-30 | [#17981](https://github.com/sgl-project/sglang/pull/17981) | open | [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell. | `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`, `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py` |
| 2026-01-30 | [#17983](https://github.com/sgl-project/sglang/pull/17983) | open | [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel | `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` |
| 2026-02-01 | [#14607](https://github.com/sgl-project/sglang/pull/14607) | merged | support qwen3-next eagle3 | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-08 | [#18224](https://github.com/sgl-project/sglang/pull/18224) | merged | [ModelOPT] Support Qwen 3 Next Coder NVFP4 | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-09 | [#18489](https://github.com/sgl-project/sglang/pull/18489) | merged | [MODEL] Adding Support for Qwen3.5 Models | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py` |
| 2026-02-14 | [#17613](https://github.com/sgl-project/sglang/pull/17613) | merged | [Perf] refactor piecewise cuda graph support of Qwen3-Next | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-22 | [#18917](https://github.com/sgl-project/sglang/pull/18917) | merged | [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-25 | [#18355](https://github.com/sgl-project/sglang/pull/18355) | merged | [AMD] Support Qwen3-Coder-Next on AMD platform | `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-26 | [#19220](https://github.com/sgl-project/sglang/pull/19220) | merged | [PCG] fix piecewise cuda graph for Qwen3.5 | `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-27 | [#19434](https://github.com/sgl-project/sglang/pull/19434) | merged | [Qwen3-Next] Support gdn fused_rms_norm_gated | `python/sglang/srt/models/qwen3_next.py` |
| 2026-02-28 | [#17627](https://github.com/sgl-project/sglang/pull/17627) | merged | [feat] Support nvfp4 quantized model of Qwen3-Next | `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-04 | [#19812](https://github.com/sgl-project/sglang/pull/19812) | open | Fix Qwen3.5/Qwen3Next MTP EPLB compatibility | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-03-09 | [#19767](https://github.com/sgl-project/sglang/pull/19767) | merged | Fix qwen3.5 mtp eplb related issues | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py` |
| 2026-03-12 | [#20397](https://github.com/sgl-project/sglang/pull/20397) | open | [NPU] Qwen3 next Ascend Support MTP | `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` |
| 2026-03-20 | [#19321](https://github.com/sgl-project/sglang/pull/19321) | merged | [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-23 | [#21019](https://github.com/sgl-project/sglang/pull/21019) | merged | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py` |
| 2026-03-26 | [#21313](https://github.com/sgl-project/sglang/pull/21313) | merged | bugfix for weight loading for qwen3-next | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-26 | [#21496](https://github.com/sgl-project/sglang/pull/21496) | merged | Revert "bugfix for weight loading for qwen3-next" | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21662](https://github.com/sgl-project/sglang/pull/21662) | merged | [Fix] Fix weight_loader property assignment for qwen3-next FP8 models | `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-30 | [#21684](https://github.com/sgl-project/sglang/pull/21684) | open | [bugfix] fix Qwen3-next memory leak | `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-07 | [#22073](https://github.com/sgl-project/sglang/pull/22073) | merged | [Feature] Adding Qwen3-asr Model Support | `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22458](https://github.com/sgl-project/sglang/pull/22458) | merged | Fix NCCL AllGather hanging issue for Qwen3 Next MTP | `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py` |
| 2026-04-15 | [#22876](https://github.com/sgl-project/sglang/pull/22876) | open | Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac | `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py` |
| 2026-04-17 | [#23075](https://github.com/sgl-project/sglang/pull/23075) | open | [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view. | `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py` |
| 2026-04-18 | [#22664](https://github.com/sgl-project/sglang/pull/22664) | merged | Qwen3next flashinfer allreduce auto enable | `python/sglang/srt/server_args.py` |
| 2026-04-20 | [#23273](https://github.com/sgl-project/sglang/pull/23273) | open | [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell) | `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py` |
| 2026-04-22 | [#23474](https://github.com/sgl-project/sglang/pull/23474) | open | [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models | `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py` |
| 2026-04-27 | [#21698](https://github.com/sgl-project/sglang/pull/21698) | merged | [npu]fix: qwen3-next w8a8 precision bugs | `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-29 | [#23619](https://github.com/sgl-project/sglang/pull/23619) | merged | Enable Qwen3-Next MoE all-reduce fusion | `python/sglang/srt/models/qwen3_next.py` |

## 逐 PR diff 审计卡

### PR #10233 - Qwen3-Next support

- 链接: https://github.com/sgl-project/sglang/pull/10233
- 状态/时间: merged / 2025-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；关联提交 `30c6e1f56967`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+3224/-8，可读 patch 3452 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Qwen3-Next support」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；技术摘要: 覆盖「Qwen3-Next support」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunks: -0,0 +1,1072; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating，涉及 `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel`；`python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunks: -0,0 +1,326; symbols: HybridLayerType, Qwen3NextConfig, to, __init__，涉及 `HybridLayerType, Qwen3NextConfig, to`；`python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: Qwen3NextForCausalLMMTP, __init__, forward, load_weights，涉及 `Qwen3NextForCausalLMMTP, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` added +1072/-0 (1072 lines); hunks: -0,0 +1,1072; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/configs/qwen3_next.py` added +326/-0 (326 lines); hunks: -0,0 +1,326; symbols: HybridLayerType, Qwen3NextConfig, to, __init__
  - `python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0 (117 lines); hunks: -0,0 +1,117; symbols: Qwen3NextForCausalLMMTP, __init__, forward, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -0,0 +1,1072 @@
+import enum
+import logging
+from typing import Any, Dict, Iterable, Optional, Set, Tuple
+import torch
+import torch.nn.functional as F
+from torch import nn
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -0,0 +1,326 @@
+# coding=utf-8
+# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -0,0 +1,117 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` added +1072/-0; `python/sglang/srt/configs/qwen3_next.py` added +326/-0; `python/sglang/srt/models/qwen3_next_mtp.py` added +117/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10322 - [bugfix] fix norm type error in qwen3_next model

- 链接: https://github.com/sgl-project/sglang/pull/10322
- 状态/时间: merged / 2025-09-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；关联提交 `4a0e0be2a2b4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+10/-51，可读 patch 89 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix norm type error in qwen3_next model」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunks: -518,24 +518,10 @@ def __init__(; -685,23 +671,10 @@ def __init__(; symbols: __init__, get_layer, forward，涉及 `__init__, get_layer, forward`；`python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunks: -54,15 +54,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +9/-42 (51 lines); hunks: -518,24 +518,10 @@ def __init__(; -685,23 +671,10 @@ def __init__(; symbols: __init__, get_layer, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9 (10 lines); hunks: -54,15 +54,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -518,24 +518,10 @@ def __init__(
-        if getattr(
-            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
-        ):
-            logger.warning_once(
-                "Using Gemma RMSNorm for input normalization and post attn normalization."
-            )
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -54,15 +54,7 @@ def __init__(
-        if getattr(
-            config, "use_gemma_rms_norm", getattr(config, "apply_layernorm_1p", False)
-        ):
-            logger.warning_once(
-                "Using Gemma RMSNorm for input normalization and post attn normalization."
-            )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +9/-42; `python/sglang/srt/models/qwen3_next_mtp.py` modified +1/-9
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10379 - Support Qwen3-Next on Ascend NPU

- 链接: https://github.com/sgl-project/sglang/pull/10379
- 状态/时间: merged / 2025-09-12
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `16cd550c8554`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+79/-26，可读 patch 281 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Qwen3-Next on Ascend NPU」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_next.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +6/-3 (9 lines); hunks: -46,10 +46,11; -327,7 +328,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, forward，涉及 `__init__, fix_query_key_value_ordering, _forward_input_proj`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-3 (9 lines); hunks: -46,10 +46,11; -327,7 +328,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -46,10 +46,11 @@
-from sglang.srt.utils import add_prefix, is_cuda, make_layers, set_weight_attrs
+from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+_is_npu = is_npu()
@@ -327,7 +328,7 @@ def __init__(
-            device=torch.cuda.current_device(),
+            device=torch.get_device_module().current_device(),
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +6/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10392 - [Fix] Support qwen3-next MTP+DP

- 链接: https://github.com/sgl-project/sglang/pull/10392
- 状态/时间: merged / 2025-09-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next_mtp.py`；关联提交 `9752861002d7`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+29/-18，可读 patch 120 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Support qwen3-next MTP+DP」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next_mtp.py`；技术摘要: 覆盖「[Fix] Support qwen3-next MTP+DP」；主要实现面是 `python/sglang/srt/models/qwen3_next_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunks: -85,8 +85,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2 (7 lines); hunks: -85,8 +85,11 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -85,8 +85,11 @@ def forward(
-        input_embeds = self.pre_fc_norm_embedding(input_embeds)
-        hidden_states = self.pre_fc_norm_hidden(forward_batch.spec_info.hidden_states)
+        hidden_states = forward_batch.spec_info.hidden_states
+        # Some idle batch has 0 batch size. GemmaRMSNorm.forward would fail due to bs=0.
+        if not forward_batch.forward_mode.is_idle():
+            input_embeds = self.pre_fc_norm_embedding(input_embeds)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next_mtp.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/layers/logits_processor.py`, `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10466 - feat: update support for qwen3next model

- 链接: https://github.com/sgl-project/sglang/pull/10466
- 状态/时间: merged / 2025-09-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+11/-7，可读 patch 74 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: update support for qwen3next model」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；技术摘要: 覆盖「feat: update support for qwen3next model」；主要实现面是 `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunks: -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel，涉及 `fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel`；`python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -239,6 +239,7 @@ def __init__(; -278,13 +279,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunks: -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel，涉及 `fused_sigmoid_gating_delta_rule_update_kernel`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4 (8 lines); hunks: -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(; -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule_fwd_kernel, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -239,6 +239,7 @@ def __init__(; -278,13 +279,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2 (4 lines); hunks: -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_recurrent.py
@@ -86,8 +86,8 @@ def fused_recurrent_gated_delta_rule_fwd_kernel(
-            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
-            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
+            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
+            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
@@ -411,8 +411,8 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(
-            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
diff -- python/sglang/srt/models/qwen3_next.py
@@ -239,6 +239,7 @@ def __init__(
+        quant_config: Optional[QuantizationConfig] = None,
@@ -278,13 +279,15 @@ def __init__(
+            quant_config=quant_config,
+            quant_config=None,
@@ -336,6 +339,7 @@ def __init__(
+            quant_config=quant_config,
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -119,8 +119,8 @@ def fused_sigmoid_gating_delta_rule_update_kernel(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +4/-4; `python/sglang/srt/models/qwen3_next.py` modified +5/-1; `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10624 - update deepep version for qwen3-next deepep moe

- 链接: https://github.com/sgl-project/sglang/pull/10624
- 状态/时间: merged / 2025-09-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「update deepep version for qwen3-next deepep moe」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `docker/Dockerfile`, `scripts/ci/ci_install_deepep.sh`；技术摘要: 覆盖「update deepep version for qwen3-next deepep moe」；主要实现面是 `docker/Dockerfile`, `scripts/ci/ci_install_deepep.sh`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `docker/Dockerfile` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 as base；`scripts/ci/ci_install_deepep.sh` modified +1/-1 (2 lines); hunks: -58,7 +58,7 @@ cd build。
- 代码 diff 细节:
  - `docker/Dockerfile` modified +1/-1 (2 lines); hunks: -3,7 +3,7 @@ FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 as base
  - `scripts/ci/ci_install_deepep.sh` modified +1/-1 (2 lines); hunks: -58,7 +58,7 @@ cd build
- 关键代码摘录:

```diff
diff -- docker/Dockerfile
@@ -3,7 +3,7 @@ FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 as base
-ARG DEEPEP_COMMIT=b92d0d4860ce6866cd6d31bfbae937f9a7a3772b
+ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
diff -- scripts/ci/ci_install_deepep.sh
@@ -58,7 +58,7 @@ cd build
-rm -rf /root/.cache/deepep && git clone https://github.com/deepseek-ai/DeepEP.git /root/.cache/deepep && cd /root/.cache/deepep && git checkout b92d0d4860ce6866cd6d31bfbae937f9a7a
+rm -rf /root/.cache/deepep && git clone https://github.com/deepseek-ai/DeepEP.git /root/.cache/deepep && cd /root/.cache/deepep && git checkout 9af0e0d0e74f3577af1979c9b9e1ac2cad0
```

- 已读文件:
  - other: `docker/Dockerfile` modified +1/-1; `scripts/ci/ci_install_deepep.sh` modified +1/-1
- 验证与风险: 未看到显式测试文件；下一次修改同一区域时需要补足模型加载、短文本生成和 parser/多模态输入的回归验证。

### PR #10622 - support qwen3-next-fp8 deepep

- 链接: https://github.com/sgl-project/sglang/pull/10622
- 状态/时间: merged / 2025-09-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `1344ebc8333d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+93/-9，可读 patch 204 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3-next-fp8 deepep」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「support qwen3-next-fp8 deepep」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunks: -13,6 +13,7; -46,7 +47,14; symbols: forward, __init__, routed_experts_weights_of_layer，涉及 `forward, __init__, routed_experts_weights_of_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +29/-8 (37 lines); hunks: -13,6 +13,7; -46,7 +47,14; symbols: forward, __init__, routed_experts_weights_of_layer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -13,6 +13,7 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
@@ -46,7 +47,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +29/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10657 - feat: add eagle3 support for qwen3-next model

- 链接: https://github.com/sgl-project/sglang/pull/10657
- 状态/时间: open / 2025-09-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+45/-3，可读 patch 113 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add eagle3 support for qwen3-next model」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`；技术摘要: 覆盖「feat: add eagle3 support for qwen3-next model」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/model_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunks: -1,6 +1,6; -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, forward, HybridLayerType，涉及 `get_layer, forward, HybridLayerType`；`python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):; -1743,6 +1744,12 @@ def _get_attention_backend(self):; symbols: initialize, _get_attention_backend，涉及 `initialize, _get_attention_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +38/-3 (41 lines); hunks: -1,6 +1,6; -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):; symbols: get_layer, forward, HybridLayerType
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):; -1743,6 +1744,12 @@ def _get_attention_backend(self):; symbols: initialize, _get_attention_backend
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -1,6 +1,6 @@
-from typing import Any, Dict, Iterable, Optional, Set, Tuple
+from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
@@ -837,6 +837,9 @@ def get_layer(idx: int, prefix: str):
+        # For EAGLE3 support
+        self.layers_to_capture = []
@@ -855,9 +858,16 @@ def forward(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -347,6 +347,7 @@ def initialize(self, min_per_gpu_memory: float):
+            self.server_args.full_attention_backend = self.server_args.attention_backend
@@ -1743,6 +1744,12 @@ def _get_attention_backend(self):
+        elif self.is_draft_worker and hasattr(
+            self.server_args, "full_attention_backend"
+        ):
+            attn_backend = self._get_attention_backend_from_str(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +38/-3; `python/sglang/srt/model_executor/model_runner.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11585 - Clean up some Qwen3-Next and deterministic code

- 链接: https://github.com/sgl-project/sglang/pull/11585
- 状态/时间: merged / 2025-10-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_next.py`；关联提交 `6b143d62a297`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1/-15，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Clean up some Qwen3-Next and deterministic code」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/configs/qwen3_next.py`；技术摘要: 覆盖「Clean up some Qwen3-Next and deterministic code」；主要实现面是 `python/sglang/srt/configs/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/configs/qwen3_next.py` modified +0/-3 (3 lines); hunks: -27,12 +27,9; symbols: HybridLayerType, Qwen3NextConfig，涉及 `HybridLayerType, Qwen3NextConfig`。
- 代码 diff 细节:
  - `python/sglang/srt/configs/qwen3_next.py` modified +0/-3 (3 lines); hunks: -27,12 +27,9; symbols: HybridLayerType, Qwen3NextConfig
- 关键代码摘录:

```diff
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -27,12 +27,9 @@
-# NOTE: HybridLayerType
-    swa_attention = "swa_attention"
-    mamba2 = "mamba"
```

- 已读文件:
  - runtime: `python/sglang/srt/configs/qwen3_next.py` modified +0/-3
- 验证与风险: diff 自带测试面 `python/sglang/test/test_deterministic.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- 链接: https://github.com/sgl-project/sglang/pull/10912
- 状态/时间: merged / 2025-10-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+727/-186，可读 patch 1433 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`；技术摘要: 覆盖「[PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)」；主要实现面是 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool，涉及 `init_memory_pool`；`python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from，涉及 `__init__, for, get_speculative_mamba2_params_all_layers`；`python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo，涉及 `TransferKVChunk, TransferInfo, from_zmq`；`python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest，涉及 `clear, HybridMambaDecodeReqToTokenPool, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +25/-9 (34 lines); hunks: -1670,19 +1670,34 @@ def init_memory_pool(; -1808,6 +1823,7 @@ def init_memory_pool(; symbols: init_memory_pool
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunks: -142,72 +142,93 @@ def __init__(; -253,6 +274,22 @@ def fork_from(self, src_index: torch.Tensor) -> Optional[to...; symbols: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunks: -58,6 +58,7 @@ class TransferKVChunk:; -69,6 +70,7 @@ class TransferInfo:; symbols: TransferKVChunk, TransferInfo, from_zmq, KVArgsRegisterInfo
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunks: -25,11 +25,12; -47,9 +48,19; symbols: clear, HybridMambaDecodeReqToTokenPool, __init__, DecodeRequest
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunks: -0,0 +1,83; symbols: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +25/-9; `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137; `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17; `python/sglang/srt/disaggregation/decode.py` modified +113/-8; `python/sglang/srt/disaggregation/prefill.py` modified +71/-1; `python/sglang/srt/disaggregation/base/conn.py` modified +17/-4
  - tests: `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/test_disaggregation_hybrid_attention.py`, `test/srt/test_mamba_unittest.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #11487 - init support for KTransformers Heterogeneous Computing

- 链接: https://github.com/sgl-project/sglang/pull/11487
- 状态/时间: merged / 2025-10-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+547/-17，可读 patch 842 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「init support for KTransformers Heterogeneous Computing」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py`；技术摘要: 覆盖「init support for KTransformers Heterogeneous Computing」；主要实现面是 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunks: -4,24 +4,47; -51,6 +74,18; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, __new__，涉及 `_mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunks: -28,6 +28,11; -123,7 +128,6 @@ def __init__(; symbols: __init__, _weight_loader_physical, _weight_loader_impl, forward，涉及 `__init__, _weight_loader_physical, _weight_loader_impl`；`python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunks: -44,6 +44,7; -82,7 +83,12; symbols: forward_normal_dual_stream, __init__, post_load_weights，涉及 `forward_normal_dual_stream, __init__, post_load_weights`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunks: -19,11 +19,13; -38,6 +40,7; symbols: to_int, CompressedTensorsConfig, __init__, get_quant_method，涉及 `to_int, CompressedTensorsConfig, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8 (416 lines); hunks: -4,24 +4,47; -51,6 +74,18; symbols: _mask_topk_ids_cpu_experts, mask_cpu_expert_ids, GPTQMarlinState, __new__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3 (28 lines); hunks: -28,6 +28,11; -123,7 +128,6 @@ def __init__(; symbols: __init__, _weight_loader_physical, _weight_loader_impl, forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +21/-5 (26 lines); hunks: -44,6 +44,7; -82,7 +83,12; symbols: forward_normal_dual_stream, __init__, post_load_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1 (11 lines); hunks: -19,11 +19,13; -38,6 +40,7; symbols: to_int, CompressedTensorsConfig, __init__, get_quant_method
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0 (9 lines); hunks: -64,6 +64,13; -247,6 +254,8 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -4,24 +4,47 @@
+import re
+try:
+    from sgl_kernel import fused_marlin_moe
+    FUSED_MARLIN_MOE_AVAILABLE = True
+except ImportError:
+    FUSED_MARLIN_MOE_AVAILABLE = False
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -28,6 +28,11 @@
+from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
+    CompressedTensorsWNA16AMXEPMoEMethod,
+    CompressedTensorsWNA16AMXMoEMethod,
+    CompressedTensorsWNA16MoEMethod,
+)
@@ -123,7 +128,6 @@ def __init__(
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -44,6 +44,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +408/-8; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +25/-3; `python/sglang/srt/models/deepseek_v2.py` modified +21/-5; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +10/-1; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-0; `python/sglang/srt/layers/quantization/compressed_tensors/__init__.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/environ.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11969 - [NPU] bugfix for Qwen3-Next and performance update

- 链接: https://github.com/sgl-project/sglang/pull/11969
- 状态/时间: merged / 2025-10-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `cafebef1546e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+68/-21，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] bugfix for Qwen3-Next and performance update」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[NPU] bugfix for Qwen3-Next and performance update」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -478,6 +478,13 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +7/-0 (7 lines); hunks: -478,6 +478,13 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -478,6 +478,13 @@ def forward(
+        # Add padding for DP-Attn
+        if is_dp_attention_enabled():
+            core_attn_out_pad = torch.zeros_like(z)
+            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
+            core_attn_out = core_attn_out_pad
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `python/sglang/srt/layers/attention/mamba/mamba.py`, `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12508 - [GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup

- 链接: https://github.com/sgl-project/sglang/pull/12508
- 状态/时间: merged / 2025-11-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+71/-51，可读 patch 151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；技术摘要: 覆盖「[GDN] Fuse b.sigmoid(), fused_gdn_gating and unsqueeze into one kernel: up to 0.85% e2e speedup」；主要实现面是 `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: fused_gdn_gating_kernel, fused_gdn_gating，涉及 `fused_gdn_gating_kernel, fused_gdn_gating`；`python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunks: -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(; symbols: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet，涉及 `fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunks: -5,6 +5,7; -30,7 +31,6; symbols: forward_extend，涉及 `forward_extend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0 (69 lines); hunks: -0,0 +1,69; symbols: fused_gdn_gating_kernel, fused_gdn_gating
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-45 (45 lines); hunks: -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(; symbols: fused_qkvzba_split_reshape_cat, fused_gdn_gating_kernel, fused_gdn_gating, Qwen3GatedDeltaNet
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6 (8 lines); hunks: -5,6 +5,7; -30,7 +31,6; symbols: forward_extend
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_gdn_gating.py
@@ -0,0 +1,69 @@
+from typing import Tuple
+import torch
+import triton
+import triton.language as tl
+# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
+# beta_output = b.sigmoid()
diff -- python/sglang/srt/models/qwen3_next.py
@@ -190,51 +190,6 @@ def fused_qkvzba_split_reshape_cat(
-# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
-@triton.jit
-def fused_gdn_gating_kernel(
-    g,
-    A_log,
-    a,
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py` added +69/-0; `python/sglang/srt/models/qwen3_next.py` modified +0/-45; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +2/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_gdn_gating.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12892 - [GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup

- 链接: https://github.com/sgl-project/sglang/pull/12892
- 状态/时间: open / 2025-11-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+172/-241，可读 patch 840 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/fla/fused_recurrent.py`；技术摘要: 覆盖「[GDN/Qwen3-Next] Avoid SSM and conv state copy for speculative decoding - up to 9.47% e2e speedup」；主要实现面是 `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/fla/fused_recurrent.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunks: -460,14 +460,14 @@ def causal_conv1d_fn(; -576,8 +576,7 @@ def _causal_conv1d_update_kernel(; symbols: causal_conv1d_fn, _causal_conv1d_update_kernel，涉及 `causal_conv1d_fn, _causal_conv1d_update_kernel`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunks: -36,24 +36,21; -630,50 +627,39 @@ def forward_extend(; symbols: forward_extend, forward, update_mamba_state_after_mtp_verify，涉及 `forward_extend, forward, update_mamba_state_after_mtp_verify`；`python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunks: -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(; -366,9 +365,9 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel，涉及 `fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel`；`python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunks: -140,8 +140,16 @@ def mem_usage_bytes(self):; -171,32 +179,40 @@ def __init__(; symbols: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__，涉及 `mem_usage_bytes, SpeculativeState, at_layer_idx`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126 (179 lines); hunks: -460,14 +460,14 @@ def causal_conv1d_fn(; -576,8 +576,7 @@ def _causal_conv1d_update_kernel(; symbols: causal_conv1d_fn, _causal_conv1d_update_kernel
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55 (88 lines); hunks: -36,24 +36,21; -630,50 +627,39 @@ def forward_extend(; symbols: forward_extend, forward, update_mamba_state_after_mtp_verify
  - `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22 (50 lines); hunks: -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(; -366,9 +365,9 @@ def fused_recurrent_gated_delta_rule_update_fwd_kernel(; symbols: fused_recurrent_gated_delta_rule, fused_recurrent_gated_delta_rule_update_fwd_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35 (90 lines); hunks: -140,8 +140,16 @@ def mem_usage_bytes(self):; -171,32 +179,40 @@ def __init__(; symbols: mem_usage_bytes, SpeculativeState, at_layer_idx, __init__
  - `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2 (4 lines); hunks: -182,8 +182,8 @@ void causal_conv1d_fwd(const at::Tensor &x, const at::Tensor...
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
@@ -460,14 +460,14 @@ def causal_conv1d_fn(
+        stride_istate_seq = conv_states.stride(0)
+        stride_istate_dim = conv_states.stride(-2)
+        stride_istate_token = conv_states.stride(-1)
-            and dim == conv_states.shape[1]
-            and width - 1 <= conv_states.shape[2]
+            and dim == conv_states.shape[-2]
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -36,24 +36,21 @@
-    from sglang.srt.layers.attention.mamba.causal_conv1d import (
-        causal_conv1d_fn as causal_conv1d_fn_cuda,
-    )
+    pass
-    causal_conv1d_fn = causal_conv1d_fn_cuda
+    # TODO: add this back. This is temporarily commented to wait for sgl-kernel new version release
diff -- python/sglang/srt/layers/attention/fla/fused_recurrent.py
@@ -348,8 +348,7 @@ def fused_recurrent_gated_delta_rule(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` modified +53/-126; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +33/-55; `python/sglang/srt/layers/attention/fla/fused_recurrent.py` modified +28/-22; `python/sglang/srt/mem_cache/memory_pool.py` modified +55/-35; `python/sglang/srt/speculative/eagle_worker.py` modified +1/-1
  - other: `sgl-kernel/csrc/mamba/causal_conv1d.cu` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_recurrent.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13708 - [Fix] Qwen3Next lmhead dtype

- 链接: https://github.com/sgl-project/sglang/pull/13708
- 状态/时间: merged / 2025-11-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+0/-1，可读 patch 8 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Qwen3Next lmhead dtype」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Fix] Qwen3Next lmhead dtype」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -864,7 +864,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-1 (1 lines); hunks: -864,7 +864,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -864,7 +864,6 @@ def __init__(
-        self.lm_head = self.lm_head.float()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +0/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13081 - Support piecewise cuda graph for Qwen3-next

- 链接: https://github.com/sgl-project/sglang/pull/13081
- 状态/时间: merged / 2025-11-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `d64bf6c6ce70`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+112/-3，可读 patch 191 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support piecewise cuda graph for Qwen3-next」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Support piecewise cuda graph for Qwen3-next」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunks: -5,6 +5,7; -53,9 +54,13; symbols: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward，涉及 `fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +62/-1 (63 lines); hunks: -5,6 +5,7; -53,9 +54,13; symbols: fused_qkvzba_split_reshape_cat_kernel, fix_query_key_value_ordering, _forward_input_proj, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -53,9 +54,13 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
+from sglang.srt.utils import direct_register_custom_op
@@ -349,7 +354,11 @@ def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
-        DUAL_STREAM_TOKEN_THRESHOLD = 1024 if not _is_npu else 0
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +62/-1
- 验证与风险: diff 自带测试面 `test/srt/models/test_qwen3_next_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13924 - [performance]Qwen3 Next kernel performance optimize

- 链接: https://github.com/sgl-project/sglang/pull/13924
- 状态/时间: closed / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+38/-22，可读 patch 117 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[performance]Qwen3 Next kernel performance optimize」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；技术摘要: 覆盖「[performance]Qwen3 Next kernel performance optimize」；主要实现面是 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +38/-22 (60 lines); hunks: -13,6 +13,21; -77,6 +92,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update，涉及 `fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +38/-22 (60 lines); hunks: -13,6 +13,21; -77,6 +92,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -13,6 +13,21 @@
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +38/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13964 - [Performance]Qwen3 Next kernel performance optimize

- 链接: https://github.com/sgl-project/sglang/pull/13964
- 状态/时间: open / 2025-11-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+34/-24，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Performance]Qwen3 Next kernel performance optimize」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；技术摘要: 覆盖「[Performance]Qwen3 Next kernel performance optimize」；主要实现面是 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunks: -7,6 +7,17; -71,6 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update，涉及 `fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24 (58 lines); hunks: -7,6 +7,17; -71,6 +82,10 @@ def fused_sigmoid_gating_delta_rule_update_kernel(; symbols: fused_sigmoid_gating_delta_rule_update_kernel, fused_sigmoid_gating_delta_rule_update
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
@@ -7,6 +7,17 @@
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` modified +34/-24
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14502 - [Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next

- 链接: https://github.com/sgl-project/sglang/pull/14502
- 状态/时间: open / 2025-12-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+248/-123，可读 patch 498 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；技术摘要: 覆盖「[Qwen3-Next]Optimize piecewise CUDA graph for Qwen3-Next」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunks: -5,6 +5,7; -37,6 +38,7; symbols: forward_extend, _causal_conv1d_gdn_core，涉及 `forward_extend, _causal_conv1d_gdn_core`；`python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunks: -5,7 +5,6; -49,7 +48,6; symbols: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location，涉及 `fused_qkvzba_split_reshape_cat_kernel, forward, _forward`；`python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunks: -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):; -477,9 +481,13 @@ def capture_one_batch_size(self, num_tokens: int):; symbols: warmup_torch_compile, capture_one_batch_size，涉及 `warmup_torch_compile, capture_one_batch_size`；`python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunks: -131,10 +131,15 @@ class State:; -151,6 +156,15 @@ class SpeculativeState(State):; symbols: State, at_layer_idx, SpeculativeState, __init__，涉及 `State, at_layer_idx, SpeculativeState`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75 (294 lines); hunks: -5,6 +5,7; -37,6 +38,7; symbols: forward_extend, _causal_conv1d_gdn_core
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-41 (41 lines); hunks: -5,7 +5,6; -49,7 +48,6; symbols: fused_qkvzba_split_reshape_cat_kernel, forward, _forward, get_model_config_for_expert_location
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4 (16 lines); hunks: -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):; -477,9 +481,13 @@ def capture_one_batch_size(self, num_tokens: int):; symbols: warmup_torch_compile, capture_one_batch_size
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2 (18 lines); hunks: -131,10 +131,15 @@ class State:; -151,6 +156,15 @@ class SpeculativeState(State):; symbols: State, at_layer_idx, SpeculativeState, __init__
  - `python/sglang/srt/compilation/compilation_config.py` modified +1/-1 (2 lines); hunks: -17,7 +17,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -37,6 +38,7 @@
+from sglang.srt.utils.custom_op import register_custom_op
@@ -957,6 +959,9 @@ def forward_extend(
+        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
@@ -968,13 +973,11 @@ def forward_extend(
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,7 +5,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -49,7 +48,6 @@
-from sglang.srt.utils.custom_op import register_custom_op
@@ -59,8 +57,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -373,22 +369,6 @@ def forward(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -330,9 +330,13 @@ def warmup_torch_compile(self, num_tokens: int):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +219/-75; `python/sglang/srt/models/qwen3_next.py` modified +0/-41; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +12/-4; `python/sglang/srt/mem_cache/memory_pool.py` modified +16/-2; `python/sglang/srt/compilation/compilation_config.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14855 - Clean up GDN Init

- 链接: https://github.com/sgl-project/sglang/pull/14855
- 状态/时间: merged / 2025-12-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-13，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Clean up GDN Init」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Clean up GDN Init」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunks: -7,7 +7,7; -221,7 +221,6 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-13 (18 lines); hunks: -7,7 +7,7; -221,7 +221,6 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -7,7 +7,7 @@
-from sglang.srt.distributed import divide, get_pp_group
+from sglang.srt.distributed import get_pp_group
@@ -221,7 +221,6 @@ def __init__(
-        # QKV
@@ -232,7 +231,6 @@ def __init__(
-        # projection of the input hidden states
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +5/-13
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16164 - [NPU] Adapt qwen3-next W8A8 on NPU

- 链接: https://github.com/sgl-project/sglang/pull/16164
- 状态/时间: merged / 2026-01-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `6bc5a52fd2d4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+18/-5，可读 patch 121 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Adapt qwen3-next W8A8 on NPU」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[NPU] Adapt qwen3-next W8A8 on NPU」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunks: -202,6 +202,7 @@ def __init__(; -229,6 +230,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering，涉及 `__init__, fix_query_key_value_ordering`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +18/-5 (23 lines); hunks: -202,6 +202,7 @@ def __init__(; -229,6 +230,7 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -202,6 +202,7 @@ def __init__(
+        prefix: str = "",
@@ -229,6 +230,7 @@ def __init__(
+            prefix=add_prefix("conv1d", prefix),
@@ -241,14 +243,16 @@ def __init__(
+            prefix=add_prefix("in_proj_qkvz", prefix),
-            quant_config=None,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +18/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #16488 - Two-Batch Overlap (TBO) support to Qwen3-Next Models

- 链接: https://github.com/sgl-project/sglang/pull/16488
- 状态/时间: open / 2026-01-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+484/-13，可读 patch 594 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Two-Batch Overlap (TBO) support to Qwen3-Next Models」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；技术摘要: 覆盖「Two-Batch Overlap (TBO) support to Qwen3-Next Models」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunks: -12,7 +12,7; -50,6 +50,8; symbols: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer，涉及 `_forward, op_prepare, op_core`；`python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunks: -78,6 +78,7; -324,6 +325,96 @@ def forward(; symbols: forward, op_gate, op_shared_experts, op_select_experts，涉及 `forward, op_gate, op_shared_experts`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunks: -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; -1040,7 +1043,8 @@ def forward_extend(; symbols: _forward_metadata, forward_extend，涉及 `_forward_metadata, forward_extend`；`python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunks: -51,6 +51,16 @@ def init_new_tbo(; -209,3 +219,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill，涉及 `init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +293/-11 (304 lines); hunks: -12,7 +12,7; -50,6 +50,8; symbols: _forward, op_prepare, op_core, Qwen3HybridLinearDecoderLayer
  - `python/sglang/srt/models/qwen2_moe.py` modified +91/-0 (91 lines); hunks: -78,6 +78,7; -324,6 +325,96 @@ def forward(; symbols: forward, op_gate, op_shared_experts, op_select_experts
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1 (6 lines); hunks: -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; -1040,7 +1043,8 @@ def forward_extend(; symbols: _forward_metadata, forward_extend
  - `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0 (85 lines); hunks: -51,6 +51,16 @@ def init_new_tbo(; -209,3 +219,78 @@ def _compute_moe_qwen3_decode(layer):; symbols: init_new_tbo, _compute_moe_qwen3_decode, _compute_moe_qwen3_next_layer_operations_strategy_tbo, _compute_moe_qwen3_next_prefill
  - `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0 (9 lines); hunks: -79,6 +79,15 @@ def compute_split_seq_index(; symbols: compute_split_seq_index
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -12,7 +12,7 @@
-from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
+from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes, ScatterMode
@@ -50,6 +50,8 @@
+from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
@@ -468,7 +470,100 @@ def _forward(
+    def op_prepare(self, state):
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -78,6 +78,7 @@
+    is_non_idle_and_non_empty
@@ -324,6 +325,96 @@ def forward(
+    def op_gate(self, state):
+        if is_non_idle_and_non_empty(
+            state.forward_batch.forward_mode, state.hidden_states_mlp_input
+        ):
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -223,6 +223,9 @@ def _forward_metadata(self, forward_batch: ForwardBatch):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +293/-11; `python/sglang/srt/models/qwen2_moe.py` modified +91/-0; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +5/-1; `python/sglang/srt/batch_overlap/operations_strategy.py` modified +85/-0; `python/sglang/srt/batch_overlap/two_batch_overlap.py` modified +9/-0
  - tests: `test/manual/test_two_batch_overlap.py` modified +1/-1
- 验证与风险: diff 自带测试面 `test/manual/test_two_batch_overlap.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16863 - tiny refactor pcg split op registration

- 链接: https://github.com/sgl-project/sglang/pull/16863
- 状态/时间: merged / 2026-01-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+20/-6，可读 patch 81 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「tiny refactor pcg split op registration」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/compilation/compilation_config.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunks: -20,6 +20,7; -132,6 +133,7 @@ def forward(; symbols: forward, unified_attention_with_output，涉及 `forward, unified_attention_with_output`；`python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -5,6 +5,7; -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):; symbols: get_model_config_for_expert_location, gdn_with_output，涉及 `get_model_config_for_expert_location, gdn_with_output`；`python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunks: -1,6 +1,17; -15,11 +26,8 @@ def __init__(; symbols: register_split_op, decorator, __init__, add_split_op，涉及 `register_split_op, decorator, __init__`；`python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunks: -39,6 +39,7; -125,6 +126,7 @@ def _register_group(group: "GroupCoordinator") -> None:; symbols: _register_group, inplace_all_reduce，涉及 `_register_group, inplace_all_reduce`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/radix_attention.py` modified +2/-0 (2 lines); hunks: -20,6 +20,7; -132,6 +133,7 @@ def forward(; symbols: forward, unified_attention_with_output
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-0 (2 lines); hunks: -5,6 +5,7; -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):; symbols: get_model_config_for_expert_location, gdn_with_output
  - `python/sglang/srt/compilation/compilation_config.py` modified +14/-6 (20 lines); hunks: -1,6 +1,17; -15,11 +26,8 @@ def __init__(; symbols: register_split_op, decorator, __init__, add_split_op
  - `python/sglang/srt/distributed/parallel_state.py` modified +2/-0 (2 lines); hunks: -39,6 +39,7; -125,6 +126,7 @@ def _register_group(group: "GroupCoordinator") -> None:; symbols: _register_group, inplace_all_reduce
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/radix_attention.py
@@ -20,6 +20,7 @@
+from sglang.srt.compilation.compilation_config import register_split_op
@@ -132,6 +133,7 @@ def forward(
+@register_split_op()
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,6 +5,7 @@
+from sglang.srt.compilation.compilation_config import register_split_op
@@ -1060,6 +1061,7 @@ def get_model_config_for_expert_location(cls, config):
+@register_split_op()
diff -- python/sglang/srt/compilation/compilation_config.py
@@ -1,6 +1,17 @@
-from typing import List
+from typing import Callable, List, Optional
+SPLIT_OPS = []
+def register_split_op(op_name: Optional[str] = None):
+    def decorator(op_func: Callable):
+        name = op_name or op_func.__name__
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/radix_attention.py` modified +2/-0; `python/sglang/srt/models/qwen3_next.py` modified +2/-0; `python/sglang/srt/compilation/compilation_config.py` modified +14/-6; `python/sglang/srt/distributed/parallel_state.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/compilation/compilation_config.py`, `python/sglang/srt/distributed/parallel_state.py`, `python/sglang/srt/layers/radix_attention.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17016 - [bugfix] fix qwen3-next alt_stream none issue

- 链接: https://github.com/sgl-project/sglang/pull/17016
- 状态/时间: merged / 2026-01-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `6f10e17b4a68`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix qwen3-next alt_stream none issue」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[bugfix] fix qwen3-next alt_stream none issue」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj，涉及 `_forward_input_proj`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +5/-1 (6 lines); hunks: -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -358,7 +358,11 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):
-        if seq_len < DUAL_STREAM_TOKEN_THRESHOLD:
+        if (
+            seq_len < DUAL_STREAM_TOKEN_THRESHOLD
+            and self.alt_stream is not None
+            and get_is_capture_mode()
+        ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15631 - [jit-kernel] Add CuTe DSL GDN Decode Kernel

- 链接: https://github.com/sgl-project/sglang/pull/15631
- 状态/时间: merged / 2026-01-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1804/-1，可读 patch 1842 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[jit-kernel] Add CuTe DSL GDN Decode Kernel」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`；技术摘要: 覆盖「[jit-kernel] Add CuTe DSL GDN Decode Kernel」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/jit_kernel/cutedsl_gdn.py`, `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunks: -5,6 +5,8; -38,6 +40,7; symbols: __init__, forward_decode，涉及 `__init__, forward_decode`；`python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunks: -0,0 +1,1494; symbols: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch，涉及 `_define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen`；`python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl，涉及 `run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance`；`python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -175,6 +175,9 @@ class Envs:; symbols: Envs，涉及 `Envs`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1 (13 lines); hunks: -5,6 +5,8; -38,6 +40,7; symbols: __init__, forward_decode
  - `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0 (1494 lines); hunks: -0,0 +1,1494; symbols: _define_kernels, gdn_kernel_small_batch, gdn_kernel_small_batch_varlen, gdn_kernel_large_batch
  - `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0 (295 lines); hunks: -0,0 +1,295; symbols: run_triton_kernel, test_cutedsl_gdn_precision, test_cutedsl_gdn_performance, run_cutedsl
  - `python/sglang/srt/environ.py` modified +3/-0 (3 lines); hunks: -175,6 +175,9 @@ class Envs:; symbols: Envs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -5,6 +5,8 @@
+from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
+from sglang.srt.environ import Envs
@@ -38,6 +40,7 @@
+from sglang.srt.utils.common import rank0_log
@@ -843,6 +846,14 @@ def __init__(self, model_runner: ModelRunner):
+        use_cutedsl = Envs.SGLANG_USE_CUTEDSL_GDN_DECODE.get()
diff -- python/sglang/jit_kernel/cutedsl_gdn.py
@@ -0,0 +1,1494 @@
+"""CuTe DSL Fused Sigmoid Gating Delta Rule Kernel for GDN Decode."""
+import logging
+from typing import Dict, Optional, Tuple
+import cuda.bindings.driver as cuda
+import cutlass
+import cutlass.cute as cute
diff -- python/sglang/jit_kernel/tests/test_cutedsl_gdn.py
@@ -0,0 +1,295 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +12/-1; `python/sglang/jit_kernel/cutedsl_gdn.py` added +1494/-0; `python/sglang/srt/environ.py` modified +3/-0
  - tests: `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` added +295/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17403 - Use attn_tp_group for all reduce in token embedding

- 链接: https://github.com/sgl-project/sglang/pull/17403
- 状态/时间: merged / 2026-01-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+17/-5，可读 patch 64 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use attn_tp_group for all reduce in token embedding」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/deepseek_nextn.py`；技术摘要: 覆盖「Use attn_tp_group for all reduce in token embedding」；主要实现面是 `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/deepseek_nextn.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +11/-3 (14 lines); hunks: -19,7 +19,11; -210,6 +214,7 @@ def __init__(; symbols: __init__, forward, extra_repr，涉及 `__init__, forward, extra_repr`；`python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -564,6 +564,10 @@ def attn_tp_reduce_scatter_tensor(output: torch.Tensor, inp...; symbols: attn_tp_reduce_scatter_tensor, attn_tp_all_reduce, attn_tp_all_gather_into_tensor，涉及 `attn_tp_reduce_scatter_tensor, attn_tp_all_reduce, attn_tp_all_gather_into_tensor`；`python/sglang/srt/models/deepseek_nextn.py` modified +1/-1 (2 lines); hunks: -86,7 +86,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -2474,7 +2474,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +11/-3 (14 lines); hunks: -19,7 +19,11; -210,6 +214,7 @@ def __init__(; symbols: __init__, forward, extra_repr
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: -564,6 +564,10 @@ def attn_tp_reduce_scatter_tensor(output: torch.Tensor, inp...; symbols: attn_tp_reduce_scatter_tensor, attn_tp_all_reduce, attn_tp_all_gather_into_tensor
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-1 (2 lines); hunks: -86,7 +86,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: -2474,7 +2474,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/vocab_parallel_embedding.py
@@ -19,7 +19,11 @@
-from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_reduce,
+    get_attention_tp_rank,
+    get_attention_tp_size,
+)
diff -- python/sglang/srt/layers/dp_attention.py
@@ -564,6 +564,10 @@ def attn_tp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
+def attn_tp_all_reduce(input: torch.Tensor):
+    return get_attention_tp_group().all_reduce(input)
diff -- python/sglang/srt/models/deepseek_nextn.py
@@ -86,7 +86,7 @@ def __init__(
-            enable_tp=not is_dp_attention_enabled(),
+            use_attn_tp_group=is_dp_attention_enabled(),
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2474,7 +2474,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/vocab_parallel_embedding.py` modified +11/-3; `python/sglang/srt/layers/dp_attention.py` modified +4/-0; `python/sglang/srt/models/deepseek_nextn.py` modified +1/-1; `python/sglang/srt/models/deepseek_v2.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/deepseek_nextn.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17373 - refactor Qwen3-Next with a new RadixLinearAttention

- 链接: https://github.com/sgl-project/sglang/pull/17373
- 状态/时间: merged / 2026-01-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `419bbcee10a5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+200/-106，可读 patch 444 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「refactor Qwen3-Next with a new RadixLinearAttention」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「refactor Qwen3-Next with a new RadixLinearAttention」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunks: -29,6 +29,7; -60,8 +61,6; symbols: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward，涉及 `fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +21/-37 (58 lines); hunks: -29,6 +29,7; -60,8 +61,6; symbols: fused_qkvzba_split_reshape_cat_kernel, __init__, fix_query_key_value_ordering, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -29,6 +29,7 @@
+from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
@@ -60,8 +61,6 @@
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -305,6 +304,20 @@ def __init__(
+        self.linear_attn = RadixLinearAttention(
+            layer_id=layer_id,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +21/-37
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17570 - Use attn tp group in embedding for more models

- 链接: https://github.com/sgl-project/sglang/pull/17570
- 状态/时间: merged / 2026-01-24
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+19/-19，可读 patch 171 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use attn tp group in embedding for more models」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`；技术摘要: 覆盖「Use attn tp group in embedding for more models」；主要实现面是 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer，涉及 `__init__, get_layer`；`python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/bailing_moe.py` modified +1/-1 (2 lines); hunks: -717,7 +717,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1 (2 lines); hunks: -62,7 +62,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/falcon_h1.py` modified +1/-1 (2 lines); hunks: -394,7 +394,7 @@ def __init__(; symbols: __init__, get_layer
  - `python/sglang/srt/models/glm4.py` modified +1/-1 (2 lines); hunks: -307,7 +307,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +1/-1 (2 lines); hunks: -895,7 +895,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

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

- 已读文件:
  - runtime: `python/sglang/srt/models/bailing_moe.py` modified +1/-1; `python/sglang/srt/models/bailing_moe_nextn.py` modified +1/-1; `python/sglang/srt/models/falcon_h1.py` modified +1/-1; `python/sglang/srt/models/glm4.py` modified +1/-1; `python/sglang/srt/models/glm4_moe.py` modified +1/-1; `python/sglang/srt/models/glm4_moe_lite.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/bailing_moe_nextn.py`, `python/sglang/srt/models/falcon_h1.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17660 - [hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention

- 链接: https://github.com/sgl-project/sglang/pull/17660
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+54/-105，可读 patch 298 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention」；模型线: Qwen3 Next；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`；技术摘要: 覆盖「[hybrid-model] clean up and consolidate redundant fields in RadixLinearAttention」；主要实现面是 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunks: -635,15 +635,7 @@ def forward_decode(; -678,18 +670,18 @@ def forward_decode(; symbols: forward_decode, forward_extend，涉及 `forward_decode, forward_extend`；`python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunks: -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):; symbols: RadixLinearAttention, __init__，涉及 `RadixLinearAttention, __init__`；`python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -315,11 +315,12 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunks: -306,11 +306,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81 (115 lines); hunks: -635,15 +635,7 @@ def forward_decode(; -678,18 +670,18 @@ def forward_decode(; symbols: forward_decode, forward_extend
  - `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18 (30 lines); hunks: -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):; symbols: RadixLinearAttention, __init__
  - `python/sglang/srt/models/kimi_linear.py` modified +4/-3 (7 lines); hunks: -315,11 +315,12 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +4/-3 (7 lines); hunks: -306,11 +306,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -635,15 +635,7 @@ def forward_decode(
-        head_dim = layer.head_qk_dim
-        layer_id = layer.layer_id
-        beta = b
-        g = a
-        A_log = layer.A_log
-        dt_bias = layer.dt_bias
diff -- python/sglang/srt/layers/radix_linear_attention.py
@@ -31,36 +31,30 @@ class RadixLinearAttention(nn.Module):
-        num_qk_heads: int,
+        num_q_heads: int,
+        num_k_heads: int,
-        head_qk_dim: int,
+        head_q_dim: int,
+        head_k_dim: int,
diff -- python/sglang/srt/models/kimi_linear.py
@@ -315,11 +315,12 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +34/-81; `python/sglang/srt/layers/radix_linear_attention.py` modified +12/-18; `python/sglang/srt/models/kimi_linear.py` modified +4/-3; `python/sglang/srt/models/qwen3_next.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/radix_linear_attention.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12525 - [CPU] Optimize Qwen3-next model on CPU

- 链接: https://github.com/sgl-project/sglang/pull/12525
- 状态/时间: merged / 2026-01-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/models/qwen3_next.py`；关联提交 `336dc4579e94`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+366/-41，可读 patch 685 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] Optimize Qwen3-next model on CPU」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`；技术摘要: 覆盖「[CPU] Optimize Qwen3-next model on CPU」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/configs/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +31/-5 (36 lines); hunks: -46,6 +46,8; -56,6 +58,8; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, _forward，涉及 `__init__, fix_query_key_value_ordering, _forward_input_proj`；`python/sglang/srt/configs/qwen3_next.py` modified +7/-0 (7 lines); hunks: -20,8 +20,11; -276,6 +279,10 @@ def full_attention_layer_ids(self):; symbols: HybridLayerType, full_attention_layer_ids, mamba2_cache_params，涉及 `HybridLayerType, full_attention_layer_ids, mamba2_cache_params`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +31/-5 (36 lines); hunks: -46,6 +46,8; -56,6 +58,8; symbols: __init__, fix_query_key_value_ordering, _forward_input_proj, _forward
  - `python/sglang/srt/configs/qwen3_next.py` modified +7/-0 (7 lines); hunks: -20,8 +20,11; -276,6 +279,10 @@ def full_attention_layer_ids(self):; symbols: HybridLayerType, full_attention_layer_ids, mamba2_cache_params
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -46,6 +46,8 @@
+    cpu_has_amx_support,
+    is_cpu,
@@ -56,6 +58,8 @@
+_is_cpu = is_cpu()
+_is_amx_available = cpu_has_amx_support()
@@ -209,8 +213,16 @@ def __init__(
diff -- python/sglang/srt/configs/qwen3_next.py
@@ -20,8 +20,11 @@
+from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
+from sglang.srt.utils import is_cpu
+_is_cpu = is_cpu()
@@ -276,6 +279,10 @@ def full_attention_layer_ids(self):
+        if _is_cpu:
+            world_size = get_attention_tp_size()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +31/-5; `python/sglang/srt/configs/qwen3_next.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/qwen3_next.py`, `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/layers/amx_utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17981 - [Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.

- 链接: https://github.com/sgl-project/sglang/pull/17981
- 状态/时间: open / 2026-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+2128/-88，可读 patch 2504 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.」；模型线: Qwen3 Next；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`, `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`；技术摘要: 覆盖「[Qwen3-Next] Add cutedsl decode/mtp kernel with transposed ssm_state and prefill gluon kernel for blackwell.」；主要实现面是 `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py`, `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: CuteDSLGDNTransposeKernel, decode, extend, target_verify，涉及 `CuteDSLGDNTransposeKernel, decode, extend`；`python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunks: -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h，涉及 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h`；`python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunks: -4,6 +4,9; -57,6 +60,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0 (6 lines); hunks: -33,6 +33,7 @@ def chunk_gated_delta_rule_fwd(; -56,6 +57,7 @@ def chunk_gated_delta_rule_fwd(; symbols: chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule，涉及 `chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0 (115 lines); hunks: -0,0 +1,115; symbols: CuteDSLGDNTransposeKernel, decode, extend, target_verify
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28 (107 lines); hunks: -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1 (24 lines); hunks: -4,6 +4,9; -57,6 +60,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0 (6 lines); hunks: -33,6 +33,7 @@ def chunk_gated_delta_rule_fwd(; -56,6 +57,7 @@ def chunk_gated_delta_rule_fwd(; symbols: chunk_gated_delta_rule_fwd, forward, chunk_gated_delta_rule
  - `python/sglang/srt/layers/attention/linear/utils.py` modified +4/-0 (4 lines); hunks: -15,13 +15,17; symbols: LinearAttnKernelBackend, is_triton, is_cutedsl, is_cutedsl_transpose
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py
@@ -0,0 +1,115 @@
+import logging
+import torch
+from sglang.jit_kernel.cutedsl_gdn_transpose import (
+    cutedsl_transpose_fused_recurrent_gated_delta_rule_update,
+    cutedsl_transpose_fused_sigmoid_gated_delta_rule_update,
+)
diff -- python/sglang/srt/layers/attention/fla/chunk_delta_h.py
@@ -55,6 +55,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
+    TRANSPOSE_STATE: tl.constexpr,
@@ -101,23 +102,47 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
-        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
-        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
-        if K > 64:
-            p_h0_2 = tl.make_block_ptr(
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -4,6 +4,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl_transpose.py` added +115/-0; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +79/-28; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +23/-1; `python/sglang/srt/layers/attention/fla/chunk.py` modified +6/-0; `python/sglang/srt/layers/attention/linear/utils.py` modified +4/-0; `python/sglang/jit_kernel/cutedsl_gdn_transpose.py` added +1038/-0
  - tests: `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py` modified +858/-57
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_cutedsl_gdn.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17983 - [Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel

- 链接: https://github.com/sgl-project/sglang/pull/17983
- 状态/时间: open / 2026-01-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+1248/-97，可读 patch 1469 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`；技术摘要: 覆盖「[Qwen3-Next] Optimize Prefill Kernel, add GDN Gluon kernel and optimize cumsum kernel」；主要实现面是 `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py`, `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunks: -0,0 +1,293; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon，涉及 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon`；`python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: recompute_w_u_fwd_kernel_gluon，涉及 `recompute_w_u_fwd_kernel_gluon`；`python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunks: -0,0 +1,210; symbols: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon，涉及 `_mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon`；`python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunks: -13,7 +13,14; -55,6 +62,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h，涉及 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0 (293 lines); hunks: -0,0 +1,293; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0 (245 lines); hunks: -0,0 +1,245; symbols: recompute_w_u_fwd_kernel_gluon
  - `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0 (210 lines); hunks: -0,0 +1,210; symbols: _mask_scalar, _apply_causal_mask, chunk_fwd_kernel_o_gluon
  - `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29 (207 lines); hunks: -13,7 +13,14; -55,6 +62,7 @@ def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(; symbols: chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_gated_delta_rule_fwd_h
  - `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18 (124 lines); hunks: -9,11 +9,77; -178,23 +244,45 @@ def chunk_local_cumsum_scalar(; symbols: chunk_local_cumsum_scalar_vectorization_kernel, chunk_local_cumsum_scalar
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py
@@ -0,0 +1,293 @@
+from sglang.srt.layers.attention.fla.gluon import (
+    TensorMemoryLayout,
+    allocate_tensor_memory,
+    fence_async_shared,
+    get_tmem_reg_layout,
+    gl,
diff -- python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py
@@ -0,0 +1,245 @@
+from sglang.srt.layers.attention.fla.gluon import (
+    TensorMemoryLayout,
+    allocate_tensor_memory,
+    fence_async_shared,
+    get_tmem_reg_layout,
+    gl,
diff -- python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py
@@ -0,0 +1,210 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/fla/gluon/chunk_delta_h_gluon.py` added +293/-0; `python/sglang/srt/layers/attention/fla/gluon/wy_fast_gluon.py` added +245/-0; `python/sglang/srt/layers/attention/fla/gluon/chunk_o_gluon.py` added +210/-0; `python/sglang/srt/layers/attention/fla/chunk_delta_h.py` modified +178/-29; `python/sglang/srt/layers/attention/fla/cumsum.py` modified +106/-18; `python/sglang/srt/layers/attention/fla/chunk_o.py` modified +87/-28
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/chunk_delta_h.py`, `python/sglang/srt/layers/attention/fla/chunk_o.py`, `python/sglang/srt/layers/attention/fla/cumsum.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #14607 - support qwen3-next eagle3

- 链接: https://github.com/sgl-project/sglang/pull/14607
- 状态/时间: merged / 2026-02-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `3ca29dffc72a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+73/-6，可读 patch 156 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3-next eagle3」；模型线: Qwen3 Next；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「support qwen3-next eagle3」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunks: -531,12 +531,18 @@ def forward(; -753,10 +759,16 @@ def forward(; symbols: forward, get_layer, set_eagle3_layers_to_capture，涉及 `forward, get_layer, set_eagle3_layers_to_capture`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +73/-6 (79 lines); hunks: -531,12 +531,18 @@ def forward(; -753,10 +759,16 @@ def forward(; symbols: forward, get_layer, set_eagle3_layers_to_capture
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -531,12 +531,18 @@ def forward(
+        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +73/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18224 - [ModelOPT] Support Qwen 3 Next Coder NVFP4

- 链接: https://github.com/sgl-project/sglang/pull/18224
- 状态/时间: merged / 2026-02-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+35/-6，可读 patch 95 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ModelOPT] Support Qwen 3 Next Coder NVFP4」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[ModelOPT] Support Qwen 3 Next Coder NVFP4」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM，涉及 `__init__, HybridLayerType, Qwen3NextForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +35/-6 (41 lines); hunks: -665,6 +665,7 @@ def __init__(; -921,6 +922,15 @@ class HybridLayerType(enum.Enum):; symbols: __init__, HybridLayerType, Qwen3NextForCausalLM
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -665,6 +665,7 @@ def __init__(
+            quant_config=quant_config,
@@ -921,6 +922,15 @@ class HybridLayerType(enum.Enum):
+    # Map fused module names to their checkpoint (unfused) counterparts.
+    # This is needed so the quantization exclusion logic can match
+    # checkpoint-style names (e.g. "q_proj") against the fused sglang
+    # module names (e.g. "qkv_proj").
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +35/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18489 - [MODEL] Adding Support for Qwen3.5 Models

- 链接: https://github.com/sgl-project/sglang/pull/18489
- 状态/时间: merged / 2026-02-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+1923/-9，可读 patch 2159 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MODEL] Adding Support for Qwen3.5 Models」；模型线: Qwen3 Next；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`；技术摘要: 覆盖「[MODEL] Adding Support for Qwen3.5 Models」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/configs/qwen3_5.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward，涉及 `Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering`；`python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward，涉及 `Qwen3_5MultiTokenPredictor, __init__, embed_input_ids`；`python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config，涉及 `Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__`；`python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: -38,6 +38,8; -1498,8 +1500,15 @@ def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope，涉及 `qwen3_next_config, hybrid_gdn_config, compute_logprobs_only`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` added +1310/-0 (1310 lines); hunks: -0,0 +1,1310; symbols: Qwen3_5GatedDeltaNet, __init__, fix_query_key_value_ordering, forward
  - `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0 (415 lines); hunks: -0,0 +1,415; symbols: Qwen3_5MultiTokenPredictor, __init__, embed_input_ids, forward
  - `python/sglang/srt/configs/qwen3_5.py` added +113/-0 (113 lines); hunks: -0,0 +1,113; symbols: Qwen3_5VisionConfig, Qwen3_5TextConfig, __init__, Qwen3_5Config
  - `python/sglang/srt/model_executor/model_runner.py` modified +14/-3 (17 lines); hunks: -38,6 +38,8; -1498,8 +1500,15 @@ def qwen3_next_config(self):; symbols: qwen3_next_config, hybrid_gdn_config, compute_logprobs_only, model_is_mrope
  - `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1 (17 lines); hunks: -7,6 +7,7; -15,6 +16,10; symbols: preprocess_video, QwenVLImageProcessor, process_mm_data_async
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -0,0 +1,1310 @@
+# Copyright 2025 Qwen Team
+# Copyright 2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -0,0 +1,415 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/configs/qwen3_5.py
@@ -0,0 +1,113 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` added +1310/-0; `python/sglang/srt/models/qwen3_5_mtp.py` added +415/-0; `python/sglang/srt/configs/qwen3_5.py` added +113/-0; `python/sglang/srt/model_executor/model_runner.py` modified +14/-3; `python/sglang/srt/multimodal/processors/qwen_vl.py` modified +16/-1; `python/sglang/srt/layers/logits_processor.py` modified +11/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17613 - [Perf] refactor piecewise cuda graph support of Qwen3-Next

- 链接: https://github.com/sgl-project/sglang/pull/17613
- 状态/时间: merged / 2026-02-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `8be18c655d0f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+80/-34，可读 patch 196 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Perf] refactor piecewise cuda graph support of Qwen3-Next」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Perf] refactor piecewise cuda graph support of Qwen3-Next」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunks: -316,7 +316,7 @@ def __init__(; -405,23 +405,6 @@ def forward(; symbols: __init__, forward, _forward，涉及 `__init__, forward, _forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-19 (21 lines); hunks: -316,7 +316,7 @@ def __init__(; -405,23 +405,6 @@ def forward(; symbols: __init__, forward, _forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -316,7 +316,7 @@ def __init__(
-        self.linear_attn = RadixLinearAttention(
+        self.attn = RadixLinearAttention(
@@ -405,23 +405,6 @@ def forward(
-        if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
-            output = torch.empty_like(hidden_states)
-            gdn_with_output(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-19
- 验证与风险: diff 自带测试面 `test/registered/models/test_qwen3_next_models_pcg.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18917 - [Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill

- 链接: https://github.com/sgl-project/sglang/pull/18917
- 状态/时间: merged / 2026-02-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `5995bfec6368`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-7，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Qwen3-Next] Enable fused_qkvzba_split_reshape_cat also for prefill」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunks: -405,17 +405,11 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +1/-7 (8 lines); hunks: -405,17 +405,11 @@ def forward(; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -405,17 +405,11 @@ def forward(
-        is_cuda_graph = forward_batch.forward_mode.is_cuda_graph()
-        if (
-            self.num_v_heads // self.num_k_heads in [1, 2, 4]
-            and is_cuda_graph
-            and not _is_cpu
-        ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +1/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18355 - [AMD] Support Qwen3-Coder-Next on AMD platform

- 链接: https://github.com/sgl-project/sglang/pull/18355
- 状态/时间: merged / 2026-02-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+213/-74，可读 patch 395 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Support Qwen3-Coder-Next on AMD platform」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[AMD] Support Qwen3-Coder-Next on AMD platform」；主要实现面是 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata，涉及 `ForwardMetadata, __init__, init_forward_metadata`；`python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj，涉及 `_forward_input_proj`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72 (283 lines); hunks: -89,6 +89,9 @@ class ForwardMetadata:; -123,7 +126,6 @@ def __init__(; symbols: ForwardMetadata, __init__, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):; symbols: _forward_input_proj
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/aiter_backend.py
@@ -89,6 +89,9 @@ class ForwardMetadata:
+    custom_mask: Optional[torch.Tensor] = None
+    mask_indptr: Optional[torch.Tensor] = None
+    max_extend_len: Optional[int] = None
@@ -123,7 +126,6 @@ def __init__(
-        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
@@ -133,6 +135,21 @@ def __init__(
diff -- python/sglang/srt/models/qwen3_next.py
@@ -385,9 +385,9 @@ def _forward_input_proj(self, hidden_states: torch.Tensor):
-            seq_len < DUAL_STREAM_TOKEN_THRESHOLD
-            and self.alt_stream is not None
+            self.alt_stream is not None
+            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/aiter_backend.py` modified +211/-72; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19220 - [PCG] fix piecewise cuda graph for Qwen3.5

- 链接: https://github.com/sgl-project/sglang/pull/19220
- 状态/时间: merged / 2026-02-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+9/-46，可读 patch 115 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PCG] fix piecewise cuda graph for Qwen3.5」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`；技术摘要: 覆盖「[PCG] fix piecewise cuda graph for Qwen3.5」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: -5,8 +5,6; -53,7 +51,6; symbols: set_eagle3_layers_to_capture, gdn_with_output，涉及 `set_eagle3_layers_to_capture, gdn_with_output`；`python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward，涉及 `forward, _forward`；`python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b,...; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract，涉及 `_fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract`；`python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -987,6 +987,7 @@ def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward，涉及 `get_input_embeddings, should_apply_lora, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +0/-25 (25 lines); hunks: -5,8 +5,6; -53,7 +51,6; symbols: set_eagle3_layers_to_capture, gdn_with_output
  - `python/sglang/srt/models/qwen3_5.py` modified +1/-21 (22 lines); hunks: -22,9 +22,6; -72,7 +69,6; symbols: forward, _forward
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0 (7 lines); hunks: -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b,...; symbols: _fp8_scaled_mm_abstract, _fp8_blockwise_scaled_mm_abstract
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-0 (1 lines); hunks: -987,6 +987,7 @@ def get_input_embeddings(self):; symbols: get_input_embeddings, should_apply_lora, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -5,8 +5,6 @@
-from sglang.srt.compilation.compilation_config import register_split_op
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -53,7 +51,6 @@
-from sglang.srt.utils.custom_op import register_custom_op
@@ -1149,25 +1146,3 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):
-@register_custom_op(mutates_args=["output"])
diff -- python/sglang/srt/models/qwen3_5.py
@@ -22,9 +22,6 @@
-# Model Executor
-from sglang.srt.compilation.piecewise_context_manager import get_forward_context
@@ -72,7 +69,6 @@
-from sglang.srt.models.qwen3_next import gdn_with_output
@@ -253,22 +249,6 @@ def forward(
-    ):
diff -- python/sglang/srt/layers/quantization/fp8_utils.py
@@ -71,6 +71,13 @@ def _fp8_scaled_mm_abstract(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=No
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +0/-25; `python/sglang/srt/models/qwen3_5.py` modified +1/-21; `python/sglang/srt/layers/quantization/fp8_utils.py` modified +7/-0; `python/sglang/srt/models/qwen3_vl.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19434 - [Qwen3-Next] Support gdn fused_rms_norm_gated

- 链接: https://github.com/sgl-project/sglang/pull/19434
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `d2885a9094c3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+411/-299，可读 patch 758 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Support gdn fused_rms_norm_gated」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Qwen3-Next] Support gdn fused_rms_norm_gated」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunks: -53,6 +53,9; -291,14 +294,23 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-8 (28 lines); hunks: -53,6 +53,9; -291,14 +294,23 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -53,6 +53,9 @@
+from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
@@ -291,14 +294,23 @@ def __init__(
-        self.norm = RMSNormGated(
-            self.head_v_dim,
-            eps=self.layer_norm_epsilon,
-            group_size=None,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +20/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/fla/fused_norm_gate.py`, `python/sglang/srt/layers/attention/fla/kda.py`, `python/sglang/srt/models/kimi_linear.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #17627 - [feat] Support nvfp4 quantized model of Qwen3-Next

- 链接: https://github.com/sgl-project/sglang/pull/17627
- 状态/时间: merged / 2026-02-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`, `test/registered/models/test_qwen3_next_models_fp4.py`；关联提交 `a2ea5941d562`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+83/-1，可读 patch 105 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] Support nvfp4 quantized model of Qwen3-Next」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[feat] Support nvfp4 quantized model of Qwen3-Next」；主要实现面是 `test/registered/models/test_qwen3_next_models_fp4.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k，涉及 `TestQwen3NextFp4, setUpClass, tearDownClass`；`python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunks: -625,13 +625,19 @@ def __init__(; -1123,6 +1129,11 @@ def load_weights(; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0 (71 lines); hunks: -0,0 +1,71; symbols: TestQwen3NextFp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/models/qwen3_next.py` modified +12/-1 (13 lines); hunks: -625,13 +625,19 @@ def __init__(; -1123,6 +1129,11 @@ def load_weights(; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- test/registered/models/test_qwen3_next_models_fp4.py
@@ -0,0 +1,71 @@
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import get_device_sm, kill_process_tree
+from sglang.test.ci.ci_register import register_cuda_ci
+from sglang.test.few_shot_gsm8k import run_eval
+from sglang.test.test_utils import (
diff -- python/sglang/srt/models/qwen3_next.py
@@ -625,13 +625,19 @@ def __init__(
+        # qkv_proj is not quantized for fp4
-            quant_config=quant_config,
+            quant_config=(
+                quant_config
+                if quant_config is not None
+                and quant_config.get_name() != "modelopt_fp4"
```

- 已读文件:
  - tests: `test/registered/models/test_qwen3_next_models_fp4.py` added +71/-0
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +12/-1
- 验证与风险: diff 自带测试面 `test/registered/models/test_qwen3_next_models_fp4.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19812 - Fix Qwen3.5/Qwen3Next MTP EPLB compatibility

- 链接: https://github.com/sgl-project/sglang/pull/19812
- 状态/时间: open / 2026-03-04
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-0，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Qwen3.5/Qwen3Next MTP EPLB compatibility」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`；技术摘要: 覆盖「Fix Qwen3.5/Qwen3Next MTP EPLB compatibility」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen2_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunks: -888,6 +888,27 @@ def __init__(; -1200,6 +1221,10 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights，涉及 `__init__, routed_experts_weights_of_layer, get_model_config_for_expert_location`；`python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -156,6 +156,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +25/-0 (25 lines); hunks: -888,6 +888,27 @@ def __init__(; -1200,6 +1221,10 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, get_model_config_for_expert_location, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +1/-0 (1 lines); hunks: -156,6 +156,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -888,6 +888,27 @@ def __init__(
+        self._routed_experts_weights_of_layer = LazyValue(
+            lambda: {
+                layer_id: layer.mlp.get_moe_weights()
+                for layer_id, layer in enumerate(self.layers)
+                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
+            }
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -156,6 +156,7 @@ def __init__(
+        self.is_nextn = is_nextn
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +25/-0; `python/sglang/srt/models/qwen2_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19767 - Fix qwen3.5 mtp eplb related issues

- 链接: https://github.com/sgl-project/sglang/pull/19767
- 状态/时间: merged / 2026-03-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+79/-16，可读 patch 272 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix qwen3.5 mtp eplb related issues」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`；技术摘要: 覆盖「Fix qwen3.5 mtp eplb related issues」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`, `python/sglang/srt/models/qwen3_next_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward，涉及 `__init__, get_model_config_for_expert_location, get_embed_and_head`；`python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: -22,6 +22,7; -62,7 +63,10 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: -150,6 +150,7 @@ def __init__(; -220,6 +221,7 @@ def __init__(; symbols: __init__, get_moe_weights, _forward_deepep，涉及 `__init__, get_moe_weights, _forward_deepep`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-1 (35 lines); hunks: -72,7 +72,14; -294,6 +301,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6 (25 lines); hunks: -22,6 +22,8; -69,6 +71,7 @@ def __init__(; symbols: __init__, get_model_config_for_expert_location, get_embed_and_head, forward
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7 (19 lines); hunks: -22,6 +22,7; -62,7 +63,10 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +8/-2 (10 lines); hunks: -150,6 +150,7 @@ def __init__(; -220,6 +221,7 @@ def __init__(; symbols: __init__, get_moe_weights, _forward_deepep
  - `python/sglang/srt/models/qwen3_next.py` modified +6/-0 (6 lines); hunks: -485,6 +485,7 @@ def __init__(; -513,6 +514,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -72,7 +72,14 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu, make_layers, set_weight_attrs
+from sglang.srt.utils import (
+    LazyValue,
+    add_prefix,
+    is_cuda,
+    is_npu,
diff -- python/sglang/srt/models/qwen3_5_mtp.py
@@ -22,6 +22,8 @@
+from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -69,6 +71,7 @@ def __init__(
+            is_nextn=True,
@@ -84,6 +87,15 @@ def __init__(
+    @classmethod
diff -- python/sglang/srt/models/qwen3_next_mtp.py
@@ -22,6 +22,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-1; `python/sglang/srt/models/qwen3_5_mtp.py` modified +19/-6; `python/sglang/srt/models/qwen3_next_mtp.py` modified +12/-7; `python/sglang/srt/models/qwen2_moe.py` modified +8/-2; `python/sglang/srt/models/qwen3_next.py` modified +6/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_5_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20397 - [NPU] Qwen3 next Ascend Support MTP

- 链接: https://github.com/sgl-project/sglang/pull/20397
- 状态/时间: open / 2026-03-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+985/-94，可读 patch 1352 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Qwen3 next Ascend Support MTP」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py`, `python/sglang/srt/layers/attention/linear/gdn_backend.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunks: -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(; symbols: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block，涉及 `fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu`；`python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunks: -1,7 +1,7; -37,9 +37,20; symbols: vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode, forward_extend，涉及 `vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunks: -22,13 +22,16; -142,6 +145,9 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata，涉及 `__init__, _forward_metadata, prepare_gdn_inputs`；`python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3 (12 lines); hunks: -16,7 +16,7; -46,6 +46,8 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0 (537 lines); hunks: -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(; symbols: fused_mamba_state_scatter_with_mask, fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat_npu, move_cache_dynamic_last_kernel_h_block
  - `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61 (343 lines); hunks: -1,7 +1,7; -37,9 +37,20; symbols: vllm_causal_conv1d_update, GDNKernelDispatcher, forward_decode, forward_extend
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3 (97 lines); hunks: -22,13 +22,16; -142,6 +145,9 @@ def __init__(self, model_runner: ModelRunner):; symbols: __init__, _forward_metadata, prepare_gdn_inputs, init_forward_metadata
  - `python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3 (12 lines); hunks: -16,7 +16,7; -46,6 +46,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/layernorm.py` modified +4/-3 (7 lines); hunks: -84,6 +84,7; -508,11 +509,11 @@ def forward_npu(; symbols: RMSNorm, forward_npu, forward_xpu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py
@@ -188,3 +188,540 @@ def fused_mamba_state_scatter_with_mask(
+# todo: move to sgl_kernel_npu
+from sgl_kernel_npu.utils.triton_utils import get_device_properties
+MAX_ROWS_PER_ITER = 64
+@triton.jit(do_not_specialize=["total_rows", "rows_per_vec"])
+def fused_qkvzba_split_reshape_cat_kernel(
+    mixed_qkv,
diff -- python/sglang/srt/layers/attention/linear/gdn_backend.py
@@ -1,7 +1,7 @@
-from typing import Tuple, Union
+from typing import Tuple, Union, Optional
+import torch.nn.functional as F
@@ -37,9 +37,20 @@
+    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
+    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -22,13 +22,16 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/mamba/mamba_state_scatter_triton.py` modified +537/-0; `python/sglang/srt/layers/attention/linear/gdn_backend.py` modified +282/-61; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +94/-3; `python/sglang/srt/models/qwen3_next_mtp.py` modified +9/-3; `python/sglang/srt/layers/layernorm.py` modified +4/-3; `python/sglang/srt/models/qwen3_next.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19321 - [Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj

- 链接: https://github.com/sgl-project/sglang/pull/19321
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `d9794ef9f760`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+107/-17，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Qwen3-Next] Fuse Qwen3-Next GDN's qkvz_proj and ba_proj」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader，涉及 `__init__, fix_query_key_value_ordering, _make_packed_weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +83/-11 (94 lines); hunks: -20,6 +20,7; -245,28 +246,38 @@ def __init__(; symbols: __init__, fix_query_key_value_ordering, _make_packed_weight_loader, weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -20,6 +20,7 @@
+    MergedColumnParallelLinear,
@@ -245,28 +246,38 @@ def __init__(
-        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
-        projection_size_ba = self.num_v_heads * 2
-        self.in_proj_qkvz = ColumnParallelLinear(
-            input_size=self.hidden_size,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +83/-11
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/linear.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21019 - [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel

- 链接: https://github.com/sgl-project/sglang/pull/21019
- 状态/时间: merged / 2026-03-23
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+597/-202，可读 patch 953 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`；技术摘要: 覆盖「[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`, `python/sglang/jit_kernel/triton/gdn_fused_proj.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: -3,6 +3,7; -55,6 +56,7; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__，涉及 `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet`；`python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: -0,0 +1,310; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous，涉及 `fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +285/-65 (350 lines); hunks: -20,6 +20,11; -54,6 +59,10; symbols: __init__
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-137 (139 lines); hunks: -3,6 +3,7; -55,6 +56,7; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, Qwen3GatedDeltaNet, __init__
  - `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0 (310 lines); hunks: -0,0 +1,310; symbols: fused_qkvzba_split_reshape_cat_kernel, fused_qkvzba_split_reshape_cat, fused_qkvzba_split_reshape_cat_contiguous_kernel, fused_qkvzba_split_reshape_cat_contiguous
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -20,6 +20,11 @@
+import triton
+from sglang.jit_kernel.triton.gdn_fused_proj import (
+    fused_qkvzba_split_reshape_cat_contiguous,
+)
@@ -54,6 +59,10 @@
+from sglang.srt.layers.parameter import (
diff -- python/sglang/srt/models/qwen3_next.py
@@ -3,6 +3,7 @@
+import triton
@@ -55,6 +56,7 @@
+from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkvzba_split_reshape_cat
@@ -63,143 +65,6 @@
-import triton
-import triton.language as tl
diff -- python/sglang/jit_kernel/triton/gdn_fused_proj.py
@@ -0,0 +1,310 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +285/-65; `python/sglang/srt/models/qwen3_next.py` modified +2/-137; `python/sglang/jit_kernel/triton/gdn_fused_proj.py` added +310/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/jit_kernel/triton/gdn_fused_proj.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21313 - bugfix for weight loading for qwen3-next

- 链接: https://github.com/sgl-project/sglang/pull/21313
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `0906e45cec97`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「bugfix for weight loading for qwen3-next」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「bugfix for weight loading for qwen3-next」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,10 +135,10 @@ def __init__(
-        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
+        self.in_proj_qkvz.weight._weight_loader = self._make_packed_weight_loader(
-        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
+        self.in_proj_ba.weight._weight_loader = self._make_packed_weight_loader(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21496 - Revert "bugfix for weight loading for qwen3-next"

- 链接: https://github.com/sgl-project/sglang/pull/21496
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `a93065679b63`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "bugfix for weight loading for qwen3-next"」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Revert "bugfix for weight loading for qwen3-next"」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +2/-2 (4 lines); hunks: -135,10 +135,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,10 +135,10 @@ def __init__(
-        self.in_proj_qkvz.weight._weight_loader = self._make_packed_weight_loader(
+        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
-        self.in_proj_ba.weight._weight_loader = self._make_packed_weight_loader(
+        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21662 - [Fix] Fix weight_loader property assignment for qwen3-next FP8 models

- 链接: https://github.com/sgl-project/sglang/pull/21662
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `62a63eeff76d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+17/-4，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Fix weight_loader property assignment for qwen3-next FP8 models」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[Fix] Fix weight_loader property assignment for qwen3-next FP8 models」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunks: -135,11 +135,11 @@ def __init__(; -216,6 +216,19 @@ def __init__(; symbols: __init__, _override_weight_loader, _make_packed_weight_loader，涉及 `__init__, _override_weight_loader, _make_packed_weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-4 (21 lines); hunks: -135,11 +135,11 @@ def __init__(; -216,6 +216,19 @@ def __init__(; symbols: __init__, _override_weight_loader, _make_packed_weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -135,11 +135,11 @@ def __init__(
-        self.in_proj_qkvz.weight.weight_loader = self._make_packed_weight_loader(
-            self.in_proj_qkvz
+        self._override_weight_loader(
+            self.in_proj_qkvz, self._make_packed_weight_loader(self.in_proj_qkvz)
-        self.in_proj_ba.weight.weight_loader = self._make_packed_weight_loader(
-            self.in_proj_ba
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +17/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #21684 - [bugfix] fix Qwen3-next memory leak

- 链接: https://github.com/sgl-project/sglang/pull/21684
- 状态/时间: open / 2026-03-30
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bugfix] fix Qwen3-next memory leak」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`；未提供可用技术摘要。
- 实现要点: `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7 @@ def alloc(self, need_size: int):; symbols: alloc, free，涉及 `alloc, free`；`python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunks: -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:; symbols: alloc, free，涉及 `alloc, free`。
- 代码 diff 细节:
  - `python/sglang/srt/mem_cache/allocator.py` modified +1/-1 (2 lines); hunks: -150,7 +150,7 @@ def alloc(self, need_size: int):; symbols: alloc, free
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1 (2 lines); hunks: -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:; symbols: alloc, free
- 关键代码摘录:

```diff
diff -- python/sglang/srt/mem_cache/allocator.py
@@ -150,7 +150,7 @@ def alloc(self, need_size: int):
-        return select_index
+        return select_index.clone()
diff -- python/sglang/srt/mem_cache/memory_pool.py
@@ -356,7 +356,7 @@ def alloc(self, need_size: int) -> Optional[torch.Tensor]:
-        return select_index
+        return select_index.clone()
```

- 已读文件:
  - runtime: `python/sglang/srt/mem_cache/allocator.py` modified +1/-1; `python/sglang/srt/mem_cache/memory_pool.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/mem_cache/memory_pool.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22073 - [Feature] Adding Qwen3-asr Model Support

- 链接: https://github.com/sgl-project/sglang/pull/22073
- 状态/时间: merged / 2026-04-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+571/-11，可读 patch 689 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Adding Qwen3-asr Model Support」；模型线: Qwen3 Next；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`；技术摘要: 覆盖「[Feature] Adding Qwen3-asr Model Support」；主要实现面是 `python/sglang/srt/models/qwen3_asr.py`, `python/sglang/srt/configs/qwen3_asr.py`, `python/sglang/srt/multimodal/processors/qwen3_asr.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature，涉及 `Qwen3ASRForConditionalGeneration, __init__, pad_input_ids`；`python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config，涉及 `Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig`；`python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions，涉及 `Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt`；`python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunks: -50,12 +50,22; -71,6 +81,27 @@ def _convert_to_internal_request(; symbols: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix，涉及 `_detect_model_family, OpenAIServingTranscription, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_asr.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: Qwen3ASRForConditionalGeneration, __init__, pad_input_ids, get_audio_feature
  - `python/sglang/srt/configs/qwen3_asr.py` added +172/-0 (172 lines); hunks: -0,0 +1,172; symbols: Qwen3ASRThinkerConfig, __init__, Qwen3ASRConfig, get_text_config
  - `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0 (95 lines); hunks: -0,0 +1,95; symbols: Qwen3ASRMultimodalProcessor, __init__, _build_transcription_prompt, compute_mrope_positions
  - `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7 (64 lines); hunks: -50,12 +50,22; -71,6 +81,27 @@ def _convert_to_internal_request(; symbols: _detect_model_family, OpenAIServingTranscription, __init__, _request_id_prefix
  - `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0 (29 lines); hunks: -51,9 +51,38 @@ in the GitHub search bar.
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_asr.py
@@ -0,0 +1,199 @@
+"""Qwen3-ASR model compatible with HuggingFace weights"""
+import logging
+from typing import Any, Iterable, List, Optional, Tuple
+import torch
+import torch.nn as nn
+from sglang.srt.configs.qwen3_asr import Qwen3ASRConfig
diff -- python/sglang/srt/configs/qwen3_asr.py
@@ -0,0 +1,172 @@
+import torch
+from transformers import (
+    AutoConfig,
+    AutoFeatureExtractor,
+    AutoTokenizer,
+    PretrainedConfig,
diff -- python/sglang/srt/multimodal/processors/qwen3_asr.py
@@ -0,0 +1,95 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_asr.py` added +199/-0; `python/sglang/srt/configs/qwen3_asr.py` added +172/-0; `python/sglang/srt/multimodal/processors/qwen3_asr.py` added +95/-0; `python/sglang/srt/entrypoints/openai/serving_transcription.py` modified +57/-7; `python/sglang/srt/configs/model_config.py` modified +12/-2; `python/sglang/srt/configs/__init__.py` modified +2/-0
  - docs: `docs/supported_models/text_generation/multimodal_language_models.md` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/configs/__init__.py`, `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/configs/qwen3_asr.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22358 - Enable DFLASH support for additional model backends

- 链接: https://github.com/sgl-project/sglang/pull/22358
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+152/-5，可读 patch 299 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DFLASH support for additional model backends」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Enable DFLASH support for additional model backends」；主要实现面是 `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture，涉及 `forward, get_layer, get_input_embeddings`；`python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`；`python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture，涉及 `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22458 - Fix NCCL AllGather hanging issue for Qwen3 Next MTP

- 链接: https://github.com/sgl-project/sglang/pull/22458
- 状态/时间: merged / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+38/-0，可读 patch 68 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix NCCL AllGather hanging issue for Qwen3 Next MTP」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；技术摘要: 覆盖「Fix NCCL AllGather hanging issue for Qwen3 Next MTP」；主要实现面是 `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunks: -7,8 +7,13; -377,6 +382,20 @@ def verify(; symbols: verify，涉及 `verify`；`python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunks: -8,6 +8,11; -370,6 +375,20 @@ def sample(; symbols: sample，涉及 `sample`。
- 代码 diff 细节:
  - `python/sglang/srt/speculative/eagle_info.py` modified +19/-0 (19 lines); hunks: -7,8 +7,13; -377,6 +382,20 @@ def verify(; symbols: verify
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0 (19 lines); hunks: -8,6 +8,11; -370,6 +375,20 @@ def sample(; symbols: sample
- 关键代码摘录:

```diff
diff -- python/sglang/srt/speculative/eagle_info.py
@@ -7,8 +7,13 @@
+from sglang.srt.distributed import get_tp_group
+from sglang.srt.layers.dp_attention import (
+    get_attention_tp_group,
+    is_dp_attention_enabled,
+)
@@ -377,6 +382,20 @@ def verify(
diff -- python/sglang/srt/speculative/eagle_info_v2.py
@@ -8,6 +8,11 @@
+from sglang.srt.distributed import get_tp_group
+from sglang.srt.layers.dp_attention import (
+    get_attention_tp_group,
+    is_dp_attention_enabled,
+)
@@ -370,6 +375,20 @@ def sample(
```

- 已读文件:
  - runtime: `python/sglang/srt/speculative/eagle_info.py` modified +19/-0; `python/sglang/srt/speculative/eagle_info_v2.py` modified +19/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/speculative/eagle_info.py`, `python/sglang/srt/speculative/eagle_info_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22876 - Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac

- 链接: https://github.com/sgl-project/sglang/pull/22876
- 状态/时间: open / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+42/-0，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Fix: Raise ValueError when --enable-mixed-chunk and --mamba-scheduler-strategy extra_buffer cause ac」；主要实现面是 `test/registered/unit/server_args/test_server_args.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunks: -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check，涉及 `test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args`；`python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache，涉及 `_handle_mamba_radix_cache`。
- 代码 diff 细节:
  - `test/registered/unit/server_args/test_server_args.py` modified +35/-0 (35 lines); hunks: -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):; symbols: test_external_corpus_max_tokens_must_be_positive, TestMambaRadixCacheArgs, _make_dummy_mamba_args, test_mamba_extra_buffer_rejects_mixed_chunk_before_cuda_check
  - `python/sglang/srt/server_args.py` modified +7/-0 (7 lines); hunks: -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(; symbols: _handle_mamba_radix_cache
- 关键代码摘录:

```diff
diff -- test/registered/unit/server_args/test_server_args.py
@@ -477,5 +477,40 @@ def test_external_corpus_max_tokens_must_be_positive(self):
+class TestMambaRadixCacheArgs(CustomTestCase):
+    def _make_dummy_mamba_args(self, **overrides) -> ServerArgs:
+        args = ServerArgs(model_path="dummy")
+        args.mamba_scheduler_strategy = "extra_buffer"
+        args.disable_radix_cache = False
+        args.enable_mixed_chunk = False
diff -- python/sglang/srt/server_args.py
@@ -2250,6 +2250,13 @@ def _handle_mamba_radix_cache(
+            if self.enable_mixed_chunk:
+                raise ValueError(
+                    "mamba extra_buffer is not compatible with --enable-mixed-chunk "
+                    "because this combination may reduce model accuracy. "
+                    "Please disable --enable-mixed-chunk or use "
+                    "--mamba-scheduler-strategy no_buffer instead."
```

- 已读文件:
  - tests: `test/registered/unit/server_args/test_server_args.py` modified +35/-0
  - runtime: `python/sglang/srt/server_args.py` modified +7/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/server_args/test_server_args.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23075 - [Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.

- 链接: https://github.com/sgl-project/sglang/pull/23075
- 状态/时间: open / 2026-04-17
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+51/-13，可读 patch 128 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`；技术摘要: 覆盖「[Fix] Mixed chunk query_start_loc and mamba_cache_indices to the prefill-only prefix so that the tracking helpers see a consistent, prefill-only view.」；主要实现面是 `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/managers/schedule_batch.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunks: -195,14 +195,21 @@ def prepare_mixed(; -213,6 +220,12 @@ def prepare_mixed(; symbols: prepare_mixed，涉及 `prepare_mixed`；`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunks: -1,4 +1,5; -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; symbols: _forward_metadata，涉及 `_forward_metadata`；`python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunks: -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):; -2341,7 +2341,13 @@ def filter_batch(; symbols: mix_with_running, filter_batch, merge_batch，涉及 `mix_with_running, filter_batch, merge_batch`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6 (25 lines); hunks: -195,14 +195,21 @@ def prepare_mixed(; -213,6 +220,12 @@ def prepare_mixed(; symbols: prepare_mixed
  - `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2 (23 lines); hunks: -1,4 +1,5; -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):; symbols: _forward_metadata
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-5 (16 lines); hunks: -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):; -2341,7 +2341,13 @@ def filter_batch(; symbols: mix_with_running, filter_batch, merge_batch
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/mamba/mamba2_metadata.py
@@ -195,14 +195,21 @@ def prepare_mixed(
-        num_prefills = len(forward_batch.extend_seq_lens)
-        num_prefill_tokens = forward_batch.extend_num_tokens
-        num_decodes = len(forward_batch.seq_lens) - num_prefills
+        # In MIXED mode (enable_mixed_chunk),We derive num_prefills by subtracting the number
+        # of decode requests (= total seq_lens count - extend_seq_lens count) from
+        # the total extend count.  In non-MIXED mode the two counts are equal.
diff -- python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
@@ -1,4 +1,5 @@
+import types
@@ -205,16 +206,34 @@ def _forward_metadata(self, forward_batch: ForwardBatch):
+                    # In MIXED mode (enable_mixed_chunk), slice query_start_loc and
+                    # mamba_cache_indices to the prefill-only prefix so that the
+                    # tracking helpers see a consistent, prefill-only view.
+                    if forward_batch.forward_mode.is_mixed():
diff -- python/sglang/srt/managers/schedule_batch.py
@@ -1946,7 +1946,7 @@ def mix_with_running(self, running_batch: "ScheduleBatch"):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py` modified +19/-6; `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` modified +21/-2; `python/sglang/srt/managers/schedule_batch.py` modified +11/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, `python/sglang/srt/layers/attention/mamba/mamba2_metadata.py`, `python/sglang/srt/managers/schedule_batch.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22664 - Qwen3next flashinfer allreduce auto enable

- 链接: https://github.com/sgl-project/sglang/pull/22664
- 状态/时间: merged / 2026-04-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Qwen3next flashinfer allreduce auto enable」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/server_args.py`；技术摘要: 覆盖「Qwen3next flashinfer allreduce auto enable」；主要实现面是 `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunks: -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):; -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments，涉及 `_handle_model_specific_adjustments`。
- 代码 diff 细节:
  - `python/sglang/srt/server_args.py` modified +3/-1 (4 lines); hunks: -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):; -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- 关键代码摘录:

```diff
diff -- python/sglang/srt/server_args.py
@@ -2171,7 +2171,8 @@ def _handle_model_specific_adjustments(self):
-        # for models with explicit support (DeepseekV3, GptOss, Glm4Moe, Qwen3Moe)
+        # for models with explicit support (DeepseekV3, GptOss, Glm4Moe,
+        # Qwen3/Qwen3Next/Qwen3.5 MoE families)
@@ -2189,6 +2190,7 @@ def _handle_model_specific_adjustments(self):
+                "Qwen3NextForCausalLM",
```

- 已读文件:
  - runtime: `python/sglang/srt/server_args.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23273 - [NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)

- 链接: https://github.com/sgl-project/sglang/pull/23273
- 状态/时间: open / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+54/-22，可读 patch 154 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`；技术摘要: 覆盖「[NVIDIA] [GDN] Enable FlashInfer MTP verify on SM100+ (Blackwell)」；主要实现面是 `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunks: -3,7 +3,7; -27,14 +27,15; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__，涉及 `_get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__`；`python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunks: -2715,17 +2715,14 @@ def _handle_mamba_backend(self):; symbols: _handle_mamba_backend, _handle_linear_attn_backend，涉及 `_handle_mamba_backend, _handle_linear_attn_backend`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16 (67 lines); hunks: -3,7 +3,7; -27,14 +27,15; symbols: _get_flashinfer_gdn_kernels, FlashInferGDNKernel, __init__
  - `python/sglang/srt/server_args.py` modified +3/-6 (9 lines); hunks: -2715,17 +2715,14 @@ def _handle_mamba_backend(self):; symbols: _handle_mamba_backend, _handle_linear_attn_backend
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
@@ -3,7 +3,7 @@
-SM100+ (Blackwell+): decode-only with bf16 state.  More support on the way.
+SM100+ (Blackwell+): decode + MTP verify with bf16 state.  Prefill on the way.
@@ -27,14 +27,15 @@
+_flashinfer_gated_delta_rule_mtp_bf16 = None
-    Returns (available, prefill_fn, mtp_fn, decode_fn).
+    Returns (available, prefill_fn, mtp_fn, decode_fn, mtp_bf16_fn).
diff -- python/sglang/srt/server_args.py
@@ -2715,17 +2715,14 @@ def _handle_mamba_backend(self):
-        # SM100+: default to FlashInfer GDN decode when the user hasn't
-        # explicitly chosen a decode backend and mamba-ssm-dtype is bf16
-        # (required by FlashInfer GDN on SM100+).
+        # SM100+: default to FlashInfer GDN decode (and MTP verify, via pool API)
+        # when the user hasn't explicitly chosen a decode backend and
+        # mamba-ssm-dtype is bf16 (required by FlashInfer GDN on SM100+).
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` modified +51/-16; `python/sglang/srt/server_args.py` modified +3/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23474 - [Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models

- 链接: https://github.com/sgl-project/sglang/pull/23474
- 状态/时间: open / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+284/-8，可读 patch 330 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`；技术摘要: 覆盖「[Bugfix] Try to fix --cpu-offload-gb on hybrid linear-attn models」；主要实现面是 `test/registered/unit/utils/test_offloader_tied_params.py`, `python/sglang/srt/utils/offloader.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent，涉及 `_TiedChild, __init__, forward`；`python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward，涉及 `maybe_offload_to_cpu, forward`。
- 代码 diff 细节:
  - `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0 (199 lines); hunks: -0,0 +1,199; symbols: _TiedChild, __init__, forward, _TiedParent
  - `python/sglang/srt/utils/offloader.py` modified +85/-8 (93 lines); hunks: -1,7 +1,7; -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) ->...; symbols: maybe_offload_to_cpu, forward
- 关键代码摘录:

```diff
diff -- test/registered/unit/utils/test_offloader_tied_params.py
@@ -0,0 +1,199 @@
+"""Tests for OffloaderV1 with tied parameters and view aliases (see issue #23150).
+Two failure modes caused the Qwen3-Next / Qwen3.5 CPU-offload regression:
+1. **Tied parameters**: a single nn.Parameter is registered under both a parent
+   and a child module (Qwen3GatedDeltaNet + RadixLinearAttention share
+   ``A_log`` / ``dt_bias``). state_dict() then lists the same tensor under
+   multiple keys, and functional_call(..., tie_weights=True) rejects it when
diff -- python/sglang/srt/utils/offloader.py
@@ -1,7 +1,7 @@
-from typing import Callable, Generator, List, Optional
+from typing import Callable, Dict, Generator, List, Optional
@@ -106,16 +106,52 @@ def maybe_offload_to_cpu(self, module: torch.nn.Module) -> torch.nn.Module:
+        # Record tensor views that alias each parameter's *original* storage
+        # BEFORE we rebind .data to pinned CPU memory. Some hybrid linear-attn
+        # models (e.g. Qwen3-Next) cache such views, which would otherwise point
```

- 已读文件:
  - tests: `test/registered/unit/utils/test_offloader_tied_params.py` added +199/-0
  - runtime: `python/sglang/srt/utils/offloader.py` modified +85/-8
- 验证与风险: diff 自带测试面 `test/registered/unit/utils/test_offloader_tied_params.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21698 - [npu]fix: qwen3-next w8a8 precision bugs

- 链接: https://github.com/sgl-project/sglang/pull/21698
- 状态/时间: merged / 2026-04-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `f2b84b90ac86`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+22/-5，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[npu]fix: qwen3-next w8a8 precision bugs」；模型线: Qwen3 Next；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「[npu]fix: qwen3-next w8a8 precision bugs」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunks: -65,6 +65,14; -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):; symbols: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader，涉及 `Qwen3GatedDeltaNet, __init__, _override_weight_loader`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +22/-5 (27 lines); hunks: -65,6 +65,14; -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):; symbols: Qwen3GatedDeltaNet, __init__, _override_weight_loader, _make_packed_weight_loader
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -65,6 +65,14 @@
+if _is_npu:
+    from sgl_kernel_npu.fla.utils import (
+        fused_qkvzba_split_reshape_cat as fused_qkvzba_split_reshape_cat_npu,
+    )
+    fused_qkvzba_split_reshape_cat = fused_qkvzba_split_reshape_cat_npu
@@ -223,11 +231,20 @@ def _override_weight_loader(module, new_loader):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +22/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23619 - Enable Qwen3-Next MoE all-reduce fusion

- 链接: https://github.com/sgl-project/sglang/pull/23619
- 状态/时间: merged / 2026-04-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_next.py`；关联提交 `13afe8acdff3`
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+46/-23，可读 patch 90 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Qwen3-Next MoE all-reduce fusion」；模型线: Qwen3 Next；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_next.py`；技术摘要: 覆盖「Enable Qwen3-Next MoE all-reduce fusion」；主要实现面是 `python/sglang/srt/models/qwen3_next.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `python/sglang/srt/models/qwen3_next.py` modified +46/-23 (69 lines); hunks: -428,6 +428,48 @@ def forward(; -510,18 +552,8 @@ def forward(; symbols: forward, _apply_qwen3_next_mlp, Qwen3HybridLinearDecoderLayer, __init__，涉及 `forward, _apply_qwen3_next_mlp, Qwen3HybridLinearDecoderLayer`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_next.py` modified +46/-23 (69 lines); hunks: -428,6 +428,48 @@ def forward(; -510,18 +552,8 @@ def forward(; symbols: forward, _apply_qwen3_next_mlp, Qwen3HybridLinearDecoderLayer, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_next.py
@@ -428,6 +428,48 @@ def forward(
+def _apply_qwen3_next_mlp(
+    layer: nn.Module,
+    hidden_states: torch.Tensor,
+    residual: Optional[torch.Tensor],
+    forward_batch: ForwardBatch,
+) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_next.py` modified +46/-23
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
