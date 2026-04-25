# DeepSeek V3/R1 PR History

Snapshot:

- SGLang `origin/main`: `929e00eea`
- sgl-cookbook `origin/main`: `8ec4d03`
- Date: `2026-04-21`

This history covers DeepSeek V3 and R1 only. DeepSeek V3.1 and V3.2 have separate skills because their optimization problems are different.

## Chronological Timeline

| Date | PR | State | Area | Main effect |
| --- | ---: | --- | --- | --- |
| 2024-12-26 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | AMD | Enabled DeepSeek V3 on AMD by touching Triton decode attention and fused MoE. |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD accuracy | Fixed DeepSeek V3 FP8 numerical behavior in `deepseek_v2.py`. |
| 2025-02-05 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | docs | Added DeepSeek usage and multi-node documentation. |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | docs | Refined DeepSeek V3 launch-server docs. |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | MTP | Added NextN/EAGLE speculative decoding support for DeepSeek V3/R1. |
| 2025-02-26 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | FP8 GEMM | Added DeepGEMM and SGLang FP8 block-wise GEMM benchmarks. |
| 2025-03-05 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | docs | Added INT8 launch example. |
| 2025-03-07 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGEMM | Integrated DeepGEMM into `sgl-kernel`. |
| 2025-03-08 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | DeepGEMM | Made linear layers support DeepGEMM. |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | MTP/MLA | Added NextN support for the FlashInfer MLA backend. |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | FlashMLA | Added the initial FlashMLA backend. |
| 2025-03-17 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | FlashMLA/graph | Added CUDA graph support for the FlashMLA backend. |
| 2025-03-18 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | fused MoE | Added the DeepSeek-style fused MoE group gate selection kernel. |
| 2025-03-20 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | DeepGEMM default | Set DeepGEMM as the default on Hopper. |
| 2025-03-20 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | ROCm MTP | Enabled MTP/NextN on AMD GPUs. |
| 2025-03-27 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | FA3 MLA | Added FA3 backend support for MLA. |
| 2025-04-05 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | MoE align | Reduced `moe_align_block_size_kernel` small-batch overhead. |
| 2025-04-07 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | MHA chunked prefill | Added `MHA_CHUNKED_KV` for DeepSeek chunked prefill. |
| 2025-04-09 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | FA3 default | Used FA3 MLA by default on Hopper. |
| 2025-04-11 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | DeepGEMM guard | Temporarily turned off DeepGEMM by default. |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | DeepGEMM guard | Limited DeepGEMM usage to Hopper. |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | fused MoE | Applied the fused MoE gate in DeepSeek V3/R1. |
| 2025-04-14 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | MLA kernel | Added the faster `merge_state_v2` CUDA merge-attention-state kernel. |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | RoPE | Applied DeepSeek CUDA RoPE. |
| 2025-04-14 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Cutlass MLA | Added the Cutlass MLA attention backend. |
| 2025-04-15 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | DeepGEMM BMM | Introduced DeepGEMM `group_gemm_masked` as an MLA BMM exploration. |
| 2025-04-16 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | FP8 quant | Switched to `sglang_per_token_group_quant_fp8` from `sgl-kernel`. |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | MLA FP8 quant | Removed one kernel in `per_tensor_quant_mla_fp8` with zero-scalar reuse. |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | shared experts | Enabled DeepSeek V3 shared-expert fusion on SM90. |
| 2025-04-20 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | MLA copy | Removed an extra copy in DeepSeek `forward_absorb`. |
| 2025-04-22 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | MLA projection | Fused `q_a_proj` and `kv_a_proj_with_mqa` for DeepSeek. |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | DeepGEMM default | Turned DeepGEMM back on by default and updated docs. |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | MTP/fusion | Fixed the R1 combination of MTP and shared-expert fusion. |
| 2025-04-24 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | MoE tuning | Updated H20 fused-MoE Triton configs for FP8 W8A8 DeepSeek/R1. |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | MoE tuning | Updated H200 Triton 3.2 fused-MoE configs and warning behavior. |
| 2025-04-25 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | MLA KV cache | Fused the MLA set-KV-cache kernel and removed K concat overhead. |
| 2025-04-27 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | MTP ergonomics | Auto-set the MTP draft model path. |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | MTP API | Updated CI tests and docs for the MTP API change. |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | MLA streams | Overlapped q/k norm with two streams. |
| 2025-05-05 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | docs | Documented MLA attention backend choices. |
| 2025-05-07 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | MTP/DP attention | Added MTP support with DP attention. |
| 2025-05-08 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | FlashMLA/MTP | Added FlashMLA backend support with MTP and FP8 KV cache. |
| 2025-05-09 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | hybrid attention | Explored hybrid attention backend wiring; this did not become the main V3/R1 path. |
| 2025-05-12 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | fused MoE | Fused the routed scaling factor into the top-k reduce kernel. |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | DeepGEMM/MLA | Replaced Triton with DeepGEMM for `fused_qkv_a_proj_with_mqa`. |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | routed scaling | Fused routed scaling factor in DeepSeek. |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | DeepGEMM format | Supported the new DeepGEMM format in per-token-group quantization. |
| 2025-06-13 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | DeepGEMM refactor | Refactored DeepGEMM integration. |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | DeepGEMM format | Added SRT-side support for the new DeepGEMM quant format. |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | DeepGEMM format | Re-quantized DeepSeek weights for the new DeepGEMM input format. |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | DeepGEMM | Completed support for the new DeepGEMM path. |
| 2025-06-20 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | MTP/FP4 | Fixed MTP with DeepSeek R1 FP4. |
| 2025-07-04 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | W4AFP8 | Added W4AFP8 config, Cutlass W4A8 MoE, and EP-MoE path for R1. |
| 2025-07-17 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | W4AFP8 TP | Added TP mode for R1-W4AFP8 and Cutlass grouped kernels. |
| 2025-07-22 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | W4A8 DeepEP | Added normal DeepEP support for R1 W4A8/W4AFP8. |
| 2025-07-28 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | W4A8 DeepEP LL | Added low-latency DeepEP support for R1 W4A8. |
| 2025-09-04 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | W4AFP8 perf | Optimized R1 W4AFP8 glue kernels. |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | DP/compile | Fixed GPU fault when DeepSeek V3 runs with DP and torch-compile. |
| 2025-10-12 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | FP4 defaults | Updated R1-FP4 default config on Blackwell. |
| 2025-10-16 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | FP4/SM120 | Enabled FP4 DeepSeek on SM120. |
| 2025-10-23 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | deterministic | Added deterministic inference support for DeepSeek-architecture models on a single GPU. |
| 2025-10-24 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | docs | Added W4FP8 usage example. |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | quant defaults | Updated DeepSeek V3 quantization auto setting for SM100. |
| 2025-11-09 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | W4AFP8 perf | Optimized W4AFP8 kernels on DeepSeek-V3-0324. |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | MTP/B200 | Fixed DeepSeek V3 MTP on B200. |
| 2025-11-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepEP LL | Made R1 W4A8 DeepEP low-latency dispatch use FP8 communication. |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | DP accuracy | Fixed DeepSeek V3 DP accuracy with BF16 KV. |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | MXFP4 | Fixed accuracy when running MXFP4 DeepSeek V3 with EP. |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | router/top-k | Switched to optimized `fused_topk_deepseek` instead of generic `moe_fused_gate`. |
| 2025-12-20 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | PCG/FP4 | Added piecewise CUDA graph support for DeepSeek V3 FP4. |
| 2026-01-07 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | loader refactor | Split DeepSeek V2/V3 weight loader into a mixin. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | Added H20/H20-3E fused MoE configs for V3.1/V3.2 shapes. |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | eval | Removed `deepseek-r1` from thinking-mode choices because R1 reasoning parser is different from V3-style thinking mode. |
| 2026-01-25 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | router benchmark | Added Blackwell benchmark for `dsv3_router_gemm`. |
| 2026-01-26 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | loader memory | Fixed OOM by deferring `dict(weights)` materialization. |
| 2026-02-03 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | ROCm | Optimized DeepSeek R1 on MI300X. |
| 2026-02-08 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | AMD router | Used `aiter_dsv3_router_gemm` when expert count is at most 256. |
| 2026-02-09 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | XPU | Enabled R1 inference on Intel GPU. |
| 2026-02-11 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | AMD MTP | Fixed TP4 DeepSeek V3 MTP accuracy on AMD. |
| 2026-02-22 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | Migrated DeepSeek MLA forward code into shared forward-method modules. |
| 2026-02-26 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | R1 MXFP4 | Fixed AMD R1-0528-MXFP4 weight-load shape mismatch. |
| 2026-03-04 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | AMD CI | Added MI35x tests for KV FP8 and all-reduce fusion in DeepSeek lanes. |
| 2026-03-04 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | AMD perf | Used BF16 correction bias in AITER FP8 path to avoid runtime dtype conversion. |
| 2026-03-18 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | DP bugfix | Fixed GPU fault when DeepSeek R1 runs with DP. |
| 2026-03-24 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | MXFP8 | Added MXFP8 DeepSeek V3 support in routed MoE. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | Added adaptive `speculative_num_steps` for EAGLE top-k=1. |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | revert | Reverted #14162. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | Allowed piecewise CUDA graph with speculative decoding. |
| 2026-04-08 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | reland | Relanded R1 W4A8 DeepEP low-latency FP8 communication. |
| 2026-04-08 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | LoRA | Refactored LoRA quant info and added DeepSeek V3 MLA LoRA support. |
| 2026-04-16 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | CPU shared expert | Expanded CPU shared-expert interface when scaling factor is absent. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | Explored parser-gated two-phase radix-cache stripping for reasoning tokens. |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | quant bugfix | Guards `.weight` access in DeepSeek MLA for AWQ/compressed-tensors. |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | MoE/DP | Fixes double-reduce with CuteDSL EP plus DP attention. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | Added opt-in stripping of thinking tokens from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | Extends adaptive speculative decoding to spec v2 EAGLE workers. |

## Code-Level Narrative

### 1. AMD and early FP8 enablement

[#2601](https://github.com/sgl-project/sglang/pull/2601) and [#2667](https://github.com/sgl-project/sglang/pull/2667) made the DeepSeek V3 path viable on AMD. The changes landed in Triton decode attention, fused MoE, and `deepseek_v2.py`. ROCm regressions should therefore be checked against attention, MoE, and the model file.

The later AMD line is not one PR:

- [#15304](https://github.com/sgl-project/sglang/pull/15304) fixed MXFP4 accuracy with EP.
- [#18242](https://github.com/sgl-project/sglang/pull/18242) optimized R1 on MI300X.
- [#18451](https://github.com/sgl-project/sglang/pull/18451) selected AITER router GEMM for `<=256` experts.
- [#19843](https://github.com/sgl-project/sglang/pull/19843) kept correction bias BF16 in AITER FP8 routing to avoid a decode-time dtype conversion.
- [#19834](https://github.com/sgl-project/sglang/pull/19834) made KV FP8 and all-reduce fusion part of MI35x CI.

### 2. Single-node H200 optimization ladder

The H200 note covers a dense March-May 2025 performance push that was missing from the first version of this skill. The important point is that the shipped V3/R1 behavior is the result of several interacting tracks, not one PR.

FP8 Block GEMM moved from benchmarking to default DeepGEMM use:

- [#3893](https://github.com/sgl-project/sglang/pull/3893) added DeepGEMM and SGLang FP8 block-wise GEMM benchmarks.
- [#4165](https://github.com/sgl-project/sglang/pull/4165) integrated DeepGEMM into `sgl-kernel`.
- [#4199](https://github.com/sgl-project/sglang/pull/4199) made linear layers support DeepGEMM.
- [#4613](https://github.com/sgl-project/sglang/pull/4613), [#5263](https://github.com/sgl-project/sglang/pull/5263), [#5310](https://github.com/sgl-project/sglang/pull/5310), and [#5628](https://github.com/sgl-project/sglang/pull/5628) show that the default was tuned carefully: enable on Hopper, temporarily disable when needed, restrict to the safe architecture set, then re-enable with docs.
- [#5432](https://github.com/sgl-project/sglang/pull/5432) added a DeepGEMM `group_gemm_masked` BMM path and MLA FP8 quant kernels, but should be described as an explored path rather than the universal default.
- [#5473](https://github.com/sgl-project/sglang/pull/5473) moved per-token-group FP8 quantization from Triton to `sgl-kernel`.
- [#5549](https://github.com/sgl-project/sglang/pull/5549) added zero-scalar allocation reuse and removed a kernel from `per_tensor_quant_mla_fp8`.
- [#6890](https://github.com/sgl-project/sglang/pull/6890) later replaced Triton with DeepGEMM for `fused_qkv_a_proj_with_mqa`.
- [#7146](https://github.com/sgl-project/sglang/pull/7146), [#7150](https://github.com/sgl-project/sglang/pull/7150), [#7155](https://github.com/sgl-project/sglang/pull/7155), [#7156](https://github.com/sgl-project/sglang/pull/7156), and [#7172](https://github.com/sgl-project/sglang/pull/7172) migrated DeepSeek quantization to the new DeepGEMM input format.

Fused MoE was optimized around top-k, alignment, routed scaling, and shared experts:

- [#4530](https://github.com/sgl-project/sglang/pull/4530) added `sgl-kernel/csrc/moe/moe_fused_gate.cu`, tests, and benchmarks for the DeepSeek-style fused group gate.
- [#5086](https://github.com/sgl-project/sglang/pull/5086) reduced `moe_align_block_size_kernel` overhead, which matters for small batches.
- [#5371](https://github.com/sgl-project/sglang/pull/5371) applied the fused MoE gate in DeepSeek V3/R1.
- [#5571](https://github.com/sgl-project/sglang/pull/5571) enabled shared-expert fusion on SM90.
- [#5716](https://github.com/sgl-project/sglang/pull/5716) and [#5740](https://github.com/sgl-project/sglang/pull/5740) updated Triton fused-MoE configs for H20 and H200.
- [#6220](https://github.com/sgl-project/sglang/pull/6220) fused routed scaling into the top-k reduce kernel, and [#6970](https://github.com/sgl-project/sglang/pull/6970) later fused the same scaling directly in DeepSeek.

MLA backend and model-file hot-path changes were equally important:

- [#4472](https://github.com/sgl-project/sglang/pull/4472) added FlashMLA, [#4514](https://github.com/sgl-project/sglang/pull/4514) made it CUDA-graph capable, and [#6109](https://github.com/sgl-project/sglang/pull/6109) connected FlashMLA with MTP and FP8 KV cache.
- [#4831](https://github.com/sgl-project/sglang/pull/4831) added FA3 MLA and [#5210](https://github.com/sgl-project/sglang/pull/5210) made FA3 MLA the Hopper default.
- [#5390](https://github.com/sgl-project/sglang/pull/5390) added Cutlass MLA, and [#6034](https://github.com/sgl-project/sglang/pull/6034) documented the backend choices.
- [#5113](https://github.com/sgl-project/sglang/pull/5113) added `MHA_CHUNKED_KV` for chunked prefill, while [#5381](https://github.com/sgl-project/sglang/pull/5381) added the `merge_state_v2` CUDA kernel used in that family of paths.
- [#5385](https://github.com/sgl-project/sglang/pull/5385) applied DeepSeek CUDA RoPE.
- [#5578](https://github.com/sgl-project/sglang/pull/5578) removed an extra `forward_absorb` copy.
- [#5619](https://github.com/sgl-project/sglang/pull/5619) fused `q_a_proj` and `kv_a_proj_with_mqa`.
- [#5748](https://github.com/sgl-project/sglang/pull/5748) fused MLA KV-cache writes and removed K concat overhead.
- [#5977](https://github.com/sgl-project/sglang/pull/5977) overlapped q/k norm on two streams.

MTP interactions also belong to the H200 story. [#3582](https://github.com/sgl-project/sglang/pull/3582) introduced NextN/EAGLE for V3/R1, [#4218](https://github.com/sgl-project/sglang/pull/4218) connected NextN with FlashInfer MLA, [#5707](https://github.com/sgl-project/sglang/pull/5707) fixed MTP with shared-expert fusion for R1, [#5793](https://github.com/sgl-project/sglang/pull/5793) auto-set the draft path, [#5952](https://github.com/sgl-project/sglang/pull/5952) updated MTP API tests/docs, and [#6081](https://github.com/sgl-project/sglang/pull/6081) added MTP plus DP-attention support.

Closed/exploratory note: [#6151](https://github.com/sgl-project/sglang/pull/6151) is a closed hybrid-attention experiment. It should be mentioned when auditing history, but not counted as current mainline V3/R1 support.

### 3. MLA and weight loading

Current main keeps most DeepSeek V3/R1 attention in `deepseek_common/attention_forward_methods/`. The important current split is:

- `attention_backend_handler.py` decides MHA/MLA subtype and respects PCG/deterministic mode.
- `forward_mla.py` handles MLA absorb and backend-specific q/k/v preparation.
- `deepseek_weight_loader.py` post-processes `kv_b_proj` because MLA absorption needs `w_kc` and `w_vc`-style components and because FP8 block scales may need dequant or requant.

[#16649](https://github.com/sgl-project/sglang/pull/16649) split the loader into `DeepseekV2WeightLoaderMixin`, which is why later V3, V3.1, V3.2, and NextN fixes all land in the shared mixin.

[#17744](https://github.com/sgl-project/sglang/pull/17744) is a practical loader fix: it defers materializing `dict(weights)`, avoiding OOM on large DeepSeek checkpoints.

[#23195](https://github.com/sgl-project/sglang/pull/23195) is still open and important for AWQ/compressed-tensors: code that assumes `.weight` exists on attention projection modules can break quantized layers.

### 4. Shared experts and MoE routing

Current `DeepseekV2MoE` computes `num_fused_shared_experts` from config and server args. When active, the loader remaps:

```text
mlp.shared_experts -> mlp.experts.256
```

The MoE layer then exposes a fused expert layout. Under ordinary fusion, the number of experts is `n_routed_experts + n_shared_experts`. Under DeepEP fusion, the layout becomes `256 + EP_size` local slots and TopK handles interleaving.

Current main keeps fusion conservative:

- disabled for TBO/SBO
- disabled by default under DeepEP unless `--enforce-shared-experts-fusion`
- disabled when architecture, expert counts, or capability do not match
- disabled for W4AFP8 because routed and shared experts can use different quant methods

[#15347](https://github.com/sgl-project/sglang/pull/15347) moved routing toward `fused_topk_deepseek`. [#17707](https://github.com/sgl-project/sglang/pull/17707) added a Blackwell router benchmark. [#22933](https://github.com/sgl-project/sglang/pull/22933) later expanded the CPU shared-expert interface when scaling factor is absent, which is CPU parity cleanup rather than H200 GPU throughput. [#21531](https://github.com/sgl-project/sglang/pull/21531) is the open JIT migration for `dsv3_router_gemm`.

### 5. MTP and NextN

DeepSeek V3/R1 MTP is implemented through EAGLE and `DeepseekV3ForCausalLMNextN`.

Important current details:

- only one NextN layer is supported
- target model and draft layer can have different quant handling
- shared head and embed weights are reused from the target model
- AMD R1 MXFP4 has a naming workaround for `model.layers.61*`
- `SGLANG_DEEPEP_BF16_DISPATCH` can be toggled around NextN execution when needed

[#7376](https://github.com/sgl-project/sglang/pull/7376) fixed R1 FP4 MTP. [#13548](https://github.com/sgl-project/sglang/pull/13548) fixed V3 MTP on B200. [#18607](https://github.com/sgl-project/sglang/pull/18607) fixed AMD TP4 V3 MTP accuracy. [#19425](https://github.com/sgl-project/sglang/pull/19425) fixed R1 MXFP4 NextN loading shape.

The newer spec line changes runtime assumptions even for the older V3/R1 model class. [#21599](https://github.com/sgl-project/sglang/pull/21599) adds adaptive `speculative_num_steps` for EAGLE top-k=1 by threading runtime speculative state and params through server args, EAGLE workers, and runner code. [#22128](https://github.com/sgl-project/sglang/pull/22128) lets piecewise CUDA graph coexist with speculative decoding by updating `model_runner.py`, `piecewise_cuda_graph_runner.py`, and the related server-argument gate. Open [#23336](https://github.com/sgl-project/sglang/pull/23336) carries the adaptive-spec idea into spec v2 through `scheduler_output_processor_mixin.py`, `managers/utils.py`, `eagle_worker_v2.py`, and `multi_layer_eagle_worker_v2.py`.

### 6. R1 W4AFP8 and DeepEP

[#7762](https://github.com/sgl-project/sglang/pull/7762) introduced the first large W4AFP8 stack:

- `W4AFp8Config`
- `W4AFp8MoEMethod`
- Cutlass W4A8 grouped MoE kernels
- packed int4 expert weights stored as int8
- expert input-scale mapping
- EP expert-map handling

[#8118](https://github.com/sgl-project/sglang/pull/8118) added TP mode. [#8247](https://github.com/sgl-project/sglang/pull/8247) added normal DeepEP support, including `apply_deepep_normal`. [#8464](https://github.com/sgl-project/sglang/pull/8464) added low-latency DeepEP support.

The low-latency FP8 communication line must be read with its revert history:

- [#14162](https://github.com/sgl-project/sglang/pull/14162) introduced fused MoE sum/all-reduce and FP8 communication behavior.
- [#21719](https://github.com/sgl-project/sglang/pull/21719) reverted it.
- [#22316](https://github.com/sgl-project/sglang/pull/22316) relanded it.

When investigating this area, inspect current main together with #14162.

### 7. Quantized and backend-specific support

The V3/R1 quantized landscape is broad:

- W4AFP8 / W4A8: [#7762](https://github.com/sgl-project/sglang/pull/7762), [#8118](https://github.com/sgl-project/sglang/pull/8118), [#8247](https://github.com/sgl-project/sglang/pull/8247), [#8464](https://github.com/sgl-project/sglang/pull/8464), [#10027](https://github.com/sgl-project/sglang/pull/10027), [#12921](https://github.com/sgl-project/sglang/pull/12921)
- FP4/NVFP4: [#11512](https://github.com/sgl-project/sglang/pull/11512), [#11708](https://github.com/sgl-project/sglang/pull/11708), [#12778](https://github.com/sgl-project/sglang/pull/12778)
- MXFP4: [#15304](https://github.com/sgl-project/sglang/pull/15304), [#19425](https://github.com/sgl-project/sglang/pull/19425), [#21529](https://github.com/sgl-project/sglang/pull/21529) open
- MXFP8: [#21280](https://github.com/sgl-project/sglang/pull/21280)
- LoRA with DeepSeek MLA: [#22323](https://github.com/sgl-project/sglang/pull/22323), with [#22268](https://github.com/sgl-project/sglang/pull/22268) still open for adapter bypass in `prepare_qkv_latent`

### 8. Reasoning radix-cache behavior

DeepSeek V3/R1 thinking and reasoning-parser paths now have a cache-specific branch to audit. Closed [#22950](https://github.com/sgl-project/sglang/pull/22950) tried parser-gated two-phase stripping for reasoning radix caches by touching model config, scheduler, radix caches, and `reasoning_parser.py`; it is history, not current support. Merged [#23315](https://github.com/sgl-project/sglang/pull/23315) is the current path: it adds an opt-in server argument and teaches `schedule_batch.py` plus `mem_cache/common.py` to strip thinking tokens from radix-cache entries so they do not become reusable prefix tokens in later requests.

When reproducing cache-hit or prefix-reuse issues, record the reasoning parser, the opt-in flag state, and whether the prompt contains `<think>` / `</think>` tokens. A model-side parser fix and a radix-cache stripping fix are different layers.

### 9. Tests that define current main

Current main has model-backed lanes for:

- V3 basic and MTP on 8-GPU H200
- AMD V3 basic, MTP, KV FP8, and R1 MXFP4
- R1 FP8 TRTLLM backend
- V3 FP4, W4A8, CuteDSL, Cutlass, MLA, INT8 MLA
- deterministic inference
- LoRA logprob regression
- router/top-k kernel coverage

Treat those tests as part of the support contract. A code path that is not represented by these lanes should be documented as a gap or an open PR to track.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG DeepSeek V3 / R1 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2024-12-26 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | [Feature, Hardware] Enable DeepseekV3 on AMD GPUs | attention/backend, MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD DeepSeek_V3 FP8 Numerical fix | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-05 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | Feature/docs deepseek usage and add multi-node | misc |  |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | refine deepseek_v3 launch server doc | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 | model wrapper, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py` |
| 2025-02-26 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | add deepgemm and sglang fp8 block-wise gemm benchmark | quantization, kernel, tests/benchmarks | `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md` |
| 2025-03-05 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | add INT8 example into dsv3 README | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-03-07 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGemm integrate to sgl-kernel | kernel, tests/benchmarks | `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh` |
| 2025-03-08 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | linear support deepgemm | quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py` |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | Support nextn for flashinfer mla attention backend | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | Support FlashMLA backend | attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-03-17 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | Support FlashMLA backend cuda graph | attention/backend | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2025-03-18 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | Add deepseek style fused moe group gate selection kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py` |
| 2025-03-20 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | Set deepgemm to the default value in the hopper architecture. | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` |
| 2025-03-20 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | [ROCm] Enable MTP (NextN) on AMD GPU | attention/backend, kernel, tests/benchmarks | `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py` |
| 2025-03-27 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | [Feature] Support FA3 backend for MLA | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-05 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | reduce moe_align_block_size_kernel small batch mode overhead | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| 2025-04-07 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | Support MHA with chunked prefix cache for DeepSeek chunked prefill | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2025-04-09 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | feat: use fa3 mla by default on hopper | attention/backend, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py` |
| 2025-04-11 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | [Fix] Turn off DeepGEMM by default | quantization, kernel, docs/config | `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | fix: use deepgemm only on hopper | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | apply fused moe gate in ds v3/r1 | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-04-14 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | kernel: support slightly faster merge_state_v2 cuda kernel | attention/backend, kernel, tests/benchmarks | `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py` |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | Apply deepseek cuda rope | misc | `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-04-14 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Add Cutlass MLA attention backend | attention/backend, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-04-15 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | [perf] introduce deep gemm group_gemm_masked as bmm | model wrapper, quantization, kernel, tests/benchmarks | `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-16 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | Remove one kernel in per_tensor_quant_mla_fp8 | model wrapper, quantization, kernel | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | enable DeepSeek V3 shared_experts_fusion in sm90 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-20 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | Remove extra copy in deepseek forward absorb | model wrapper, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-04-22 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | Fuse q_a_proj and kv_a_proj for DeepSeek models | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | Turn on DeepGemm By Default and Update Doc | quantization, docs/config | `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py` |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1 | model wrapper | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-24 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions. | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| 2025-04-25 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | Fuse MLA set kv cache kernel | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py` |
| 2025-04-27 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | Auto set draft model path for MTP | model wrapper, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | Update ci test and doc for MTP api change | attention/backend, tests/benchmarks, docs/config | `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md` |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | Overlap qk norm with two streams | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-05 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | Update doc for MLA attention backends | docs/config | `docs/references/deepseek.md`, `docs/backend/server_arguments.md` |
| 2025-05-07 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | feat: mtp support dp-attention | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks | `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` |
| 2025-05-08 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | [Feat] Support FlashMLA backend with MTP and FP8 KV cache | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-05-09 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | [Feat] optimize Qwen3 on H20 by hybrid Attention Backend | kernel, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py` |
| 2025-05-12 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | Fuse routed scaling factor in topk_reduce kernel | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa | quantization | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | Fuse routed scaling factor in deepseek | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | Support new DeepGEMM format in per token group quant | quantization, kernel, tests/benchmarks | `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h` |
| 2025-06-13 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | Refactor DeepGEMM integration | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | Support new DeepGEMM format in per token group quant (part 2: srt) | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py` |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | Re-quantize DeepSeek model weights to support DeepGEMM new input format | model wrapper, quantization | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py` |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | Support new DeepGEMM | model wrapper, MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-20 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | Fix MTP with Deepseek R1 Fp4 | model wrapper, MoE/router, kernel | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-07-04 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` |
| 2025-07-17 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` |
| 2025-07-22 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | [1/N]Support DeepSeek-R1 w4a8 normal deepep | MoE/router, quantization, tests/benchmarks | `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-07-28 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | [2/N]Support DeepSeek-R1 w4a8 low latency deepep | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` |
| 2025-09-04 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile | attention/backend, multimodal/processor | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py` |
| 2025-10-12 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | Update DeepSeek-R1-FP4 default config on blackwell | misc | `python/sglang/srt/server_args.py` |
| 2025-10-16 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | Support running FP4 Deepseek on SM120. | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py` |
| 2025-10-23 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU | model wrapper | `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-24 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | [doc] add example of using w4fp8 for Deepseek | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | Update dsv3 quantization auto setting for sm100 | misc | `python/sglang/srt/server_args.py` |
| 2025-11-09 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | [perf]optimize w4afp8 kernel on deepseek-v3-0324 | MoE/router, quantization, kernel | `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | [Fix] Fix DeepSeek V3 MTP on B200 | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-11-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | Fix dsv3 dp accuracy issue when using bf16-kv | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | Fix the accuracy issue when running mxfp4 dsv3 model and enable ep | MoE/router, quantization | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py` |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | MoE/router, kernel, tests/benchmarks | `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2025-12-20 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | Support piecewise cuda graph for dsv3 fp4 | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-07 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | [Refactor] Split out deepseek v2 weight loader function into mixin | model wrapper | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py` |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | tests/benchmarks | `python/sglang/test/run_eval.py` |
| 2026-01-25 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | Add dsv3 router gemm benchmark on blackwell | model wrapper, MoE/router, kernel, tests/benchmarks | `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-26 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization | model wrapper | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-02-04 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | [ROCm] Optimize Deepseek R1 on MI300X | model wrapper, attention/backend, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-08 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256. | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-09 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | [Intel GPU] Enable DeepSeek R1 inference on XPU | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py` |
| 2026-02-11 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile` |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-02-26 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4 | model wrapper, quantization | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-03-04 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek) | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-04 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-18 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | Fix gpu-fault issue when run deepseek-r1 and enable dp | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-03-24 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | [RL] Support mxfp8 DeepSeek V3 | attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2026-03-27 | [#21529](https://github.com/sgl-project/sglang/pull/21529) | open | Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-03-27 | [#21531](https://github.com/sgl-project/sglang/pull/21531) | open | [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py` |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | kernel, scheduler/runtime, tests/benchmarks, docs/config | `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication" | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | kernel, scheduler/runtime, tests/benchmarks | `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-04-07 | [#22268](https://github.com/sgl-project/sglang/pull/22268) | open | [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-08 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-08 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | [Lora] Lora quat info re-factor and support deepseekv3 mla lora | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py` |
| 2026-04-16 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | [CPU] expand the interface of shared_expert without scaling factor | MoE/router, quantization, kernel, tests/benchmarks | `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp` |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | scheduler/runtime, tests/benchmarks, docs/config | `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors | model wrapper, attention/backend, tests/benchmarks | `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention | model wrapper, attention/backend, MoE/router, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | multimodal/processor, scheduler/runtime | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |

## Diff Cards

### PR #2601 - [Feature, Hardware] Enable DeepseekV3 on AMD GPUs

- Link: https://github.com/sgl-project/sglang/pull/2601
- Status/date: `merged`, created 2024-12-26, merged 2025-01-03; author `BruceXcluding`.
- Diff scope read: `2` files, `+9/-5`; areas: attention/backend, MoE/router, kernel; keywords: triton, attention, config, expert, fp8, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +5/-5 (10 lines); hunks: def invoke_fused_moe_kernel(; def get_default_config(; symbols: invoke_fused_moe_kernel, get_default_config, get_default_config, get_default_config
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +4/-0 (4 lines); hunks: def _decode_grouped_att_m_fwd(; symbols: _decode_grouped_att_m_fwd
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`; keywords observed in patches: triton, attention, config, expert, fp8, moe. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #2667 - AMD DeepSeek_V3 FP8 Numerical fix

- Link: https://github.com/sgl-project/sglang/pull/2667
- Status/date: `merged`, created 2024-12-30, merged 2024-12-30; author `HaiShaw`.
- Diff scope read: `1` files, `+34/-7`; areas: model wrapper; keywords: attention, config, flash, fp8, kv, lora, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-7 (41 lines); hunks: from sglang.srt.layers.quantization.fp8_utils import (; from sglang.srt.managers.schedule_batch import global_server_args_dict; symbols: forward_absorb, forward_absorb, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, config, flash, fp8, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #3314 - Feature/docs deepseek usage and add multi-node

- Link: https://github.com/sgl-project/sglang/pull/3314
- Status/date: `merged`, created 2025-02-05, merged 2025-02-07; author `lycanlancelot`.
- Diff scope read: `0` files, `+0/-0`; areas: misc; keywords: n/a.
- Code diff details:
  - No patch file list returned.
- Optimization/support interpretation: The concrete diff surface is no returned patch files; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises no returned patch files; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #3522 - refine deepseek_v3 launch server doc

- Link: https://github.com/sgl-project/sglang/pull/3522
- Status/date: `merged`, created 2025-02-12, merged 2025-02-12; author `BBuf`.
- Diff scope read: `1` files, `+7/-0`; areas: tests/benchmarks; keywords: attention, benchmark, config, doc, fp8, kv, mla.
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +7/-0 (7 lines); hunks: python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-r; python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 -
- Optimization/support interpretation: The concrete diff surface is `benchmark/deepseek_v3/README.md`; keywords observed in patches: attention, benchmark, config, doc, fp8, kv. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/deepseek_v3/README.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #3582 - Support NextN (MTP) speculative decoding for DeepSeek-V3/R1

- Link: https://github.com/sgl-project/sglang/pull/3582
- Status/date: `merged`, created 2025-02-14, merged 2025-02-14; author `ispobock`.
- Diff scope read: `7` files, `+437/-7`; areas: model wrapper, scheduler/runtime, docs/config; keywords: spec, config, eagle, kv, attention, cuda, expert, mla, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` added +295/-0 (295 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN
  - `scripts/export_deepseek_nextn.py` added +113/-0 (113 lines); hunks: +"""; symbols: get_nexn_layer_id, update_and_save_config, copy_non_safetensors_files, export_nextn_layer_parameters
  - `python/sglang/srt/speculative/spec_info.py` modified +11/-1 (12 lines); hunks: class SpeculativeAlgorithm(IntEnum):; symbols: SpeculativeAlgorithm, is_none, is_eagle, is_nextn
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunks: def __post_init__(self):; def add_cli_args(parser: argparse.ArgumentParser):; symbols: __post_init__, add_cli_args
  - `python/sglang/srt/speculative/eagle_worker.py` modified +7/-2 (9 lines); hunks: fast_topk,; def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py`; keywords observed in patches: spec, config, eagle, kv, attention, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #3893 - add deepgemm and sglang fp8 block-wise gemm benchmark

- Link: https://github.com/sgl-project/sglang/pull/3893
- Status/date: `merged`, created 2025-02-26, merged 2025-03-02; author `BBuf`.
- Diff scope read: `2` files, `+320/-0`; areas: quantization, kernel, tests/benchmarks; keywords: benchmark, fp8, triton, config, cuda, quant, spec, test.
- Code diff details:
  - `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py` added +314/-0 (314 lines); hunks: +import itertools; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, fp8_gemm_deepgemm, fp8_gemm_sglang
  - `benchmark/kernels/deepseek/README.md` added +6/-0 (6 lines); hunks: +## DeepSeek kernels benchmark
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md`; keywords observed in patches: benchmark, fp8, triton, config, cuda, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4079 - add INT8 example into dsv3 README

- Link: https://github.com/sgl-project/sglang/pull/4079
- Status/date: `merged`, created 2025-03-05, merged 2025-03-13; author `laixinn`.
- Diff scope read: `1` files, `+16/-2`; areas: tests/benchmarks; keywords: awq, benchmark, quant.
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +16/-2 (18 lines); hunks: AWQ does not support BF16, so add the `--dtype half` flag if AWQ is used for qua
- Optimization/support interpretation: The concrete diff surface is `benchmark/deepseek_v3/README.md`; keywords observed in patches: awq, benchmark, quant. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/deepseek_v3/README.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4165 - DeepGemm integrate to sgl-kernel

- Link: https://github.com/sgl-project/sglang/pull/4165
- Status/date: `merged`, created 2025-03-07, merged 2025-03-10; author `laixinn`.
- Diff scope read: `6` files, `+324/-5`; areas: kernel, tests/benchmarks; keywords: cuda, flash, cache, doc, fp8, test.
- Code diff details:
  - `sgl-kernel/tests/test_deep_gemm.py` added +263/-0 (263 lines); hunks: +import os; symbols: per_token_cast_to_fp8, per_block_cast_to_fp8, construct, construct_grouped
  - `sgl-kernel/setup.py` modified +52/-1 (53 lines); hunks: # ==============================================================================; def _get_version():; symbols: _get_version, _get_version, CustomBuildPy, run
  - `sgl-kernel/build.sh` modified +4/-3 (7 lines); hunks: else; docker run --rm \
  - `.gitmodules` modified +3/-0 (3 lines); hunks: [submodule "sgl-kernel/3rdparty/flashinfer"]
  - `sgl-kernel/pyproject.toml` modified +1/-1 (2 lines); hunks: [build-system]
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh`; keywords observed in patches: cuda, flash, cache, doc, fp8, test. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4199 - linear support deepgemm

- Link: https://github.com/sgl-project/sglang/pull/4199
- Status/date: `merged`, created 2025-03-08, merged 2025-03-11; author `sleepcoo`.
- Diff scope read: `3` files, `+76/-44`; areas: quantization, kernel, tests/benchmarks; keywords: fp8, quant, cuda, test, config, triton.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +36/-28 (64 lines); hunks: _is_cuda = torch.cuda.is_available() and torch.version.cuda; def grid(META):; symbols: _per_token_group_quant_fp8, grid
  - `python/sglang/test/test_block_fp8.py` modified +39/-15 (54 lines); hunks: import itertools; w8a8_block_fp8_matmul,; symbols: native_per_token_group_quant_fp8, native_w8a8_block_fp8_matmul, TestW8A8BlockFP8Matmul, setUpClass
  - `test/srt/test_fp8_kernel.py` modified +1/-1 (2 lines); hunks: def setUpClass(cls):; symbols: setUpClass, _make_A
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py`; keywords observed in patches: fp8, quant, cuda, test, config, triton. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4218 - Support nextn for flashinfer mla attention backend

- Link: https://github.com/sgl-project/sglang/pull/4218
- Status/date: `merged`, created 2025-03-09, merged 2025-03-09; author `Fridge003`.
- Diff scope read: `5` files, `+393/-58`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config; keywords: flash, mla, eagle, spec, topk, cache, attention, cuda, kv, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +317/-57 (374 lines); hunks: from dataclasses import dataclass; from sglang.srt.layers.dp_attention import get_attention_tp_size; symbols: FlashInferMLAAttnBackend, __init__, __init__, init_forward_metadata
  - `test/srt/test_mla_flashinfer.py` modified +63/-0 (63 lines); hunks: import unittest; def test_gsm8k(self):; symbols: test_gsm8k, TestFlashinferMLAMTP, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunks: def init_attention_backend(self):; symbols: init_attention_backend
  - `docs/references/deepseek.md` modified +1/-1 (2 lines); hunks: Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/be
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunks: def no_absorb() -> bool:; symbols: no_absorb
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py`; keywords observed in patches: flash, mla, eagle, spec, topk, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4472 - Support FlashMLA backend

- Link: https://github.com/sgl-project/sglang/pull/4472
- Status/date: `merged`, created 2025-03-16, merged 2025-03-16; author `sleepcoo`.
- Diff scope read: `6` files, `+209/-1`; areas: attention/backend, scheduler/runtime, tests/benchmarks; keywords: flash, mla, cache, spec, attention, triton, cuda, kv, config, eagle.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` added +128/-0 (128 lines); hunks: +from __future__ import annotations; symbols: FlashMLABackend, __init__, forward_decode
  - `python/sglang/srt/layers/attention/utils.py` modified +54/-0 (54 lines); hunks: def create_flashinfer_kv_indices_triton(; def create_flashinfer_kv_indices_triton(; symbols: create_flashinfer_kv_indices_triton, create_flashinfer_kv_indices_triton, create_flashmla_kv_indices_triton
  - `python/sglang/srt/model_executor/model_runner.py` modified +8/-0 (8 lines); hunks: def __init__(; def model_specific_adjustment(self):; symbols: __init__, model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: class ServerArgs:; def __post_init__(self):; symbols: ServerArgs:, __post_init__, add_cli_args
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunks: "speculative_accept_threshold_single": ServerArgs.speculative_accept_threshold_single,; def merge_batch(self, other: "ScheduleBatch"):; symbols: merge_batch, get_model_worker_batch
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: flash, mla, cache, spec, attention, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4514 - Support FlashMLA backend cuda graph

- Link: https://github.com/sgl-project/sglang/pull/4514
- Status/date: `merged`, created 2025-03-17, merged 2025-03-19; author `sleepcoo`.
- Diff scope read: `3` files, `+188/-32`; areas: attention/backend; keywords: flash, mla, attention, cuda, kv, triton, cache, config, eagle, lora.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +184/-30 (214 lines); hunks: from __future__ import annotations; from sglang.srt.layers.radix_attention import RadixAttention; symbols: FlashMLADecodeMetadata:, __init__, FlashMLABackend, __init__
  - `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunks: def __post_init__(self):; symbols: __post_init__
  - `python/sglang/srt/layers/attention/utils.py` modified +0/-1 (1 lines); hunks: def create_flashmla_kv_indices_triton(; symbols: create_flashmla_kv_indices_triton
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py`; keywords observed in patches: flash, mla, attention, cuda, kv, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4530 - Add deepseek style fused moe group gate selection kernel

- Link: https://github.com/sgl-project/sglang/pull/4530
- Status/date: `merged`, created 2025-03-18, merged 2025-03-29; author `qingquansong`.
- Diff scope read: `9` files, `+616/-1`; areas: MoE/router, kernel, tests/benchmarks; keywords: moe, topk, expert, cuda, quant, spec, config, fp8, test, benchmark.
- Code diff details:
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` added +447/-0 (447 lines); hunks: +#include <ATen/cuda/CUDAContext.h>; symbols: versions:, int, int, int
  - `sgl-kernel/benchmark/bench_moe_fused_gate.py` added +74/-0 (74 lines); hunks: +import itertools; symbols: biased_grouped_topk_org, biased_grouped_topk_org_kernel, benchmark
  - `sgl-kernel/tests/test_moe_fused_gate.py` added +72/-0 (72 lines); hunks: +import pytest; symbols: test_moe_fused_gate_combined
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +12/-0 (12 lines); hunks: def topk_softmax(; symbols: topk_softmax, moe_fused_gate
  - `sgl-kernel/csrc/torch_extension.cc` modified +5/-0 (5 lines); hunks: TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py`; keywords observed in patches: moe, topk, expert, cuda, quant, spec. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4613 - Set deepgemm to the default value in the hopper architecture.

- Link: https://github.com/sgl-project/sglang/pull/4613
- Status/date: `merged`, created 2025-03-20, merged 2025-03-21; author `sleepcoo`.
- Diff scope read: `2` files, `+16/-3`; areas: quantization, kernel; keywords: cuda, fp8, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-3 (12 lines); hunks: direct_register_custom_op,; import deep_gemm # `pip install "sgl-kernel>=0.0.4.post3"`; symbols: grid
  - `python/sglang/srt/utils.py` modified +7/-0 (7 lines); hunks: def get_amdgpu_memory_capacity():; symbols: get_amdgpu_memory_capacity, get_device_sm, get_nvgpu_memory_capacity
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; keywords observed in patches: cuda, fp8, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4631 - [ROCm] Enable MTP (NextN) on AMD GPU

- Link: https://github.com/sgl-project/sglang/pull/4631
- Status/date: `merged`, created 2025-03-20, merged 2025-03-24; author `alexsun07`.
- Diff scope read: `7` files, `+43/-4`; areas: attention/backend, kernel, tests/benchmarks; keywords: cuda, spec, eagle, topk, cache, expert, kv, mla, moe, test.
- Code diff details:
  - `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h` added +20/-0 (20 lines); hunks: +#include <torch/library.h>
  - `sgl-kernel/csrc/torch_extension_rocm.cc` modified +12/-0 (12 lines); hunks: TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  - `python/sglang/srt/speculative/build_eagle_tree.py` modified +2/-2 (4 lines); hunks: import torch
  - `python/sglang/srt/speculative/eagle_utils.py` modified +3/-1 (4 lines); hunks: from sglang.srt.mem_cache.memory_pool import TokenToKVPoolAllocator; tree_speculative_sampling_target_only,
  - `sgl-kernel/csrc/speculative/eagle_utils.cu` modified +4/-0 (4 lines); hunks: #include <ATen/ATen.h>
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py`; keywords observed in patches: cuda, spec, eagle, topk, cache, expert. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4831 - [Feature] Support FA3 backend for MLA

- Link: https://github.com/sgl-project/sglang/pull/4831
- Status/date: `merged`, created 2025-03-27, merged 2025-03-29; author `Fridge003`.
- Diff scope read: `3` files, `+180/-74`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, flash, mla, triton, cache, config, cuda, fp8, kv, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +171/-73 (244 lines); hunks: import torch; def __init__(; symbols: __init__, init_forward_metadata, forward_extend, forward_extend
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunks: def model_specific_adjustment(self):; def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunks: def __init__(; def no_absorb(self, forward_batch: ForwardBatch) -> bool:; symbols: __init__, no_absorb, no_absorb
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, flash, mla, triton, cache, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5086 - reduce moe_align_block_size_kernel small batch mode overhead

- Link: https://github.com/sgl-project/sglang/pull/5086
- Status/date: `merged`, created 2025-04-05, merged 2025-04-10; author `BBuf`.
- Diff scope read: `4` files, `+143/-56`; areas: MoE/router, kernel, tests/benchmarks; keywords: expert, moe, topk, triton, test, benchmark, config, quant.
- Code diff details:
  - `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +111/-44 (155 lines); hunks: __global__ void moe_align_block_size_kernel(; __global__ void moe_align_block_size_kernel(; symbols: void, void, void
  - `sgl-kernel/benchmark/bench_moe_align_block_size.py` modified +31/-10 (41 lines); hunks: def calculate_diff(num_tokens, num_experts=256, block_size=128, topk=8):; def benchmark(num_tokens, num_experts, topk, provider):; symbols: calculate_diff, benchmark, sgl_moe_align_block_size_with_empty, benchmark
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-1 (2 lines); hunks: def moe_align_block_size(; symbols: moe_align_block_size
  - `sgl-kernel/tests/test_moe_align.py` modified +0/-1 (1 lines); hunks: def moe_align_block_size_triton(; symbols: moe_align_block_size_triton, test_moe_align_block_size_compare_implementations
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; keywords observed in patches: expert, moe, topk, triton, test, benchmark. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5113 - Support MHA with chunked prefix cache for DeepSeek chunked prefill

- Link: https://github.com/sgl-project/sglang/pull/5113
- Status/date: `merged`, created 2025-04-07, merged 2025-04-16; author `Fridge003`.
- Diff scope read: `10` files, `+734/-46`; areas: model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, mla, attention, cuda, flash, kv, expert, moe, triton, deepep.
- Code diff details:
  - `python/sglang/test/attention/test_prefix_chunk_info.py` added +224/-0 (224 lines); hunks: +import unittest; symbols: MockForwardBatch, __init__, get_max_chunk_capacity, MockReqToTokenPool:
  - `python/sglang/srt/models/deepseek_v2.py` modified +174/-9 (183 lines); hunks: import logging; _is_cuda = is_cuda(); symbols: AttnForwardMethod, DeepseekV2MLP, __init__, __init__
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +181/-0 (181 lines); hunks: class ForwardBatch:; def _compute_mrope_positions(; symbols: ForwardBatch:, _compute_mrope_positions, get_max_chunk_capacity, set_prefix_chunk_idx
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +80/-34 (114 lines); hunks: from sglang.srt.layers.radix_attention import RadixAttention; def forward_extend(; symbols: forward_extend, forward_decode
  - `test/srt/test_fa3.py` modified +53/-2 (55 lines); hunks: from sglang.srt.utils import get_device_sm, kill_process_tree; """; symbols: test_gsm8k, TestFlashAttention3MLASpeculativeDecode, get_server_args, test_gsm8k
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; keywords observed in patches: cache, mla, attention, cuda, flash, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5210 - feat: use fa3 mla by default on hopper

- Link: https://github.com/sgl-project/sglang/pull/5210
- Status/date: `merged`, created 2025-04-09, merged 2025-04-12; author `zhyncs`.
- Diff scope read: `3` files, `+42/-11`; areas: attention/backend, scheduler/runtime; keywords: cuda, attention, cache, flash, fp8, kv, mla, spec, topk, config.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunks: is_cuda,; def model_specific_adjustment(self):; symbols: model_specific_adjustment, model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +12/-7 (19 lines); hunks: def init_forward_metadata(self, forward_batch: ForwardBatch):; def forward_extend(; symbols: init_forward_metadata, forward_extend, forward_decode, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/utils.py` modified +9/-0 (9 lines); hunks: def fast_topk(values, topk, dim):; symbols: fast_topk, is_hopper_with_cuda_12_3
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py`; keywords observed in patches: cuda, attention, cache, flash, fp8, kv. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5263 - [Fix] Turn off DeepGEMM by default

- Link: https://github.com/sgl-project/sglang/pull/5263
- Status/date: `merged`, created 2025-04-11, merged 2025-04-15; author `Fridge003`.
- Diff scope read: `2` files, `+6/-2`; areas: quantization, kernel, docs/config; keywords: fp8, quant, attention, doc, eagle, spec.
- Code diff details:
  - `docs/references/deepseek.md` modified +3/-1 (4 lines); hunks: With data parallelism attention enabled, we have achieved up to **1.9x** decoding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +3/-1 (4 lines); hunks: from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_quant_fp8
- Optimization/support interpretation: The concrete diff surface is `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; keywords observed in patches: fp8, quant, attention, doc, eagle, spec. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5310 - fix: use deepgemm only on hopper

- Link: https://github.com/sgl-project/sglang/pull/5310
- Status/date: `merged`, created 2025-04-12, merged 2025-04-12; author `zhyncs`.
- Diff scope read: `1` files, `+1/-1`; areas: quantization, kernel; keywords: fp8, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +1/-1 (2 lines); hunks: from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_quant_fp8
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`; keywords observed in patches: fp8, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5371 - apply fused moe gate in ds v3/r1

- Link: https://github.com/sgl-project/sglang/pull/5371
- Status/date: `merged`, created 2025-04-14, merged 2025-04-14; author `BBuf`.
- Diff scope read: `1` files, `+37/-16`; areas: MoE/router; keywords: cuda, expert, moe, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +37/-16 (53 lines); hunks: # limitations under the License.; _is_cuda = is_cuda(); symbols: biased_grouped_topk_impl, is_power_of_two, biased_grouped_topk, biased_grouped_topk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: cuda, expert, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5381 - kernel: support slightly faster merge_state_v2 cuda kernel

- Link: https://github.com/sgl-project/sglang/pull/5381
- Status/date: `merged`, created 2025-04-14, merged 2025-04-15; author `DefTruth`.
- Diff scope read: `7` files, `+638/-4`; areas: attention/backend, kernel, tests/benchmarks; keywords: attention, cuda, mla, cache, kv, triton, fp8, test.
- Code diff details:
  - `sgl-kernel/tests/test_merge_state_v2.py` added +396/-0 (396 lines); hunks: +from typing import Optional; symbols: merge_state_kernel, merge_state_triton, merge_state_torch, generate_markdown_table
  - `sgl-kernel/csrc/attention/merge_attn_states.cu` added +201/-0 (201 lines); hunks: +#include <ATen/cuda/CUDAContext.h>; symbols: void, uint
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +35/-4 (39 lines); hunks: -from typing import Tuple; def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):; symbols: lightning_attention_decode, merge_state, merge_state_v2, cutlass_mla_decode
  - `sgl-kernel/csrc/common_extension.cc` modified +2/-0 (2 lines); hunks: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-0 (2 lines); hunks: void lightning_attention_decode(
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py`; keywords observed in patches: attention, cuda, mla, cache, kv, triton. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5385 - Apply deepseek cuda rope

- Link: https://github.com/sgl-project/sglang/pull/5385
- Status/date: `merged`, created 2025-04-14, merged 2025-04-14; author `ispobock`.
- Diff scope read: `1` files, `+12/-1`; areas: misc; keywords: cache, cuda.
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +12/-1 (13 lines); hunks: def _compute_cos_sin_cache(self) -> torch.Tensor:; symbols: _compute_cos_sin_cache, forward, forward_hip, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: cache, cuda. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5390 - Add Cutlass MLA attention backend

- Link: https://github.com/sgl-project/sglang/pull/5390
- Status/date: `merged`, created 2025-04-14, merged 2025-04-28; author `trevor-m`.
- Diff scope read: `7` files, `+305/-3`; areas: attention/backend, kernel, scheduler/runtime, docs/config; keywords: attention, mla, flash, triton, kv, spec, cache, cuda, config, doc.
- Code diff details:
  - `python/sglang/srt/layers/attention/cutlass_mla_backend.py` added +278/-0 (278 lines); hunks: +from __future__ import annotations; symbols: CutlassMLADecodeMetadata:, __init__, CutlassMLABackend, __init__
  - `python/sglang/srt/server_args.py` modified +14/-1 (15 lines); hunks: def __post_init__(self):; def add_cli_args(parser: argparse.ArgumentParser):; symbols: __post_init__, add_cli_args
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunks: def model_specific_adjustment(self):; def init_attention_backend(self):; symbols: model_specific_adjustment, init_attention_backend
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +3/-0 (3 lines); hunks: def cutlass_mla_decode(; def cutlass_mla_decode(; symbols: cutlass_mla_decode, cutlass_mla_decode, cutlass_mla_get_workspace_size
  - `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunks: Please consult the documentation below to learn more about the parameters you ma
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`; keywords observed in patches: attention, mla, flash, triton, kv, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5432 - [perf] introduce deep gemm group_gemm_masked as bmm

- Link: https://github.com/sgl-project/sglang/pull/5432
- Status/date: `merged`, created 2025-04-15, merged 2025-04-20; author `Alcanderian`.
- Diff scope read: `3` files, `+361/-20`; areas: model wrapper, quantization, kernel, tests/benchmarks; keywords: cuda, fp8, mla, quant, triton, moe, awq, config, expert, flash.
- Code diff details:
  - `python/sglang/test/test_block_fp8.py` modified +167/-0 (167 lines); hunks: from sglang.srt.layers.activation import SiluAndMul; def test_per_tensor_quant_mla_fp8(self):; symbols: test_per_tensor_quant_mla_fp8, TestPerTokenGroupQuantMlaDeepGemmMaskedFP8, setUpClass, _per_token_group_quant_mla_deep_gemm_masked_fp8
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +108/-4 (112 lines); hunks: fp8_min = -fp8_max; ); symbols: per_tensor_quant_mla_fp8, _per_token_group_quant_mla_deep_gemm_masked_fp8, per_tensor_quant_mla_deep_gemm_masked_fp8, scaled_fp8_quant
  - `python/sglang/srt/models/deepseek_v2.py` modified +86/-16 (102 lines); hunks: from sglang.srt.layers.moe.fused_moe_triton import FusedMoE; _is_cuda = is_cuda(); symbols: __init__, forward_absorb, forward_absorb, post_load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: cuda, fp8, mla, quant, triton, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5473 - use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel

- Link: https://github.com/sgl-project/sglang/pull/5473
- Status/date: `merged`, created 2025-04-16, merged 2025-04-18; author `strgrb`.
- Diff scope read: `2` files, `+25/-6`; areas: quantization, kernel; keywords: fp8, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +24/-5 (29 lines); hunks: def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-1 (2 lines); hunks: def apply_w8a8_block_fp8_linear(; symbols: apply_w8a8_block_fp8_linear
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: fp8, quant. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5549 - Remove one kernel in per_tensor_quant_mla_fp8

- Link: https://github.com/sgl-project/sglang/pull/5549
- Status/date: `merged`, created 2025-04-19, merged 2025-04-19; author `fzyzcjy`.
- Diff scope read: `4` files, `+62/-18`; areas: model wrapper, quantization, kernel; keywords: cuda, fp8, mla, quant, attention, config, deepep, expert, kv, spec.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +32/-9 (41 lines); hunks: from sglang.srt.managers.schedule_batch import global_server_args_dict; class AttnForwardMethod(IntEnum):; symbols: AttnForwardMethod, forward, forward, forward_normal
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-7 (16 lines); hunks: ):; def _per_tensor_quant_mla_fp8_stage2(; symbols: deep_gemm_fp8_fp8_bf16_nt, _per_tensor_quant_mla_fp8_stage2, per_tensor_quant_mla_fp8, per_tensor_quant_mla_fp8
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture, BumpAllocator:, __init__, allocate
  - `python/sglang/srt/models/deepseek_nextn.py` modified +8/-2 (10 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; keywords observed in patches: cuda, fp8, mla, quant, attention, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5571 - enable DeepSeek V3 shared_experts_fusion in sm90

- Link: https://github.com/sgl-project/sglang/pull/5571
- Status/date: `merged`, created 2025-04-20, merged 2025-04-20; author `BBuf`.
- Diff scope read: `1` files, `+12/-0`; areas: model wrapper; keywords: config, cuda, deepep, expert, fp8, kv, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-0 (12 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, cuda, deepep, expert, fp8, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5578 - Remove extra copy in deepseek forward absorb

- Link: https://github.com/sgl-project/sglang/pull/5578
- Status/date: `merged`, created 2025-04-20, merged 2025-04-22; author `ispobock`.
- Diff scope read: `3` files, `+18/-21`; areas: model wrapper, tests/benchmarks; keywords: cache, doc, kv, lora, mla, test.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunks: def forward_absorb(; def forward_absorb(; symbols: forward_absorb, forward_absorb
  - `.github/workflows/pr-test-amd.yml` modified +7/-7 (14 lines); hunks: jobs:; jobs:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +2/-1 (3 lines); hunks: def forward_native(; def forward_native(; symbols: forward_native, forward_native, Llama3RotaryEmbedding
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: cache, doc, kv, lora, mla, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5619 - Fuse q_a_proj and kv_a_proj for DeepSeek models

- Link: https://github.com/sgl-project/sglang/pull/5619
- Status/date: `merged`, created 2025-04-22, merged 2025-04-23; author `Fridge003`.
- Diff scope read: `1` files, `+78/-25`; areas: model wrapper; keywords: attention, cache, config, expert, kv, lora, mla, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +78/-25 (103 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, forward_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, config, expert, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5628 - Turn on DeepGemm By Default and Update Doc

- Link: https://github.com/sgl-project/sglang/pull/5628
- Status/date: `merged`, created 2025-04-22, merged 2025-04-22; author `Fridge003`.
- Diff scope read: `2` files, `+9/-3`; areas: quantization, docs/config; keywords: quant, attention, doc, eagle, fp8, spec.
- Code diff details:
  - `docs/references/deepseek.md` modified +8/-2 (10 lines); hunks: With data parallelism attention enabled, we have achieved up to **1.9x** decoding
  - `python/sglang/srt/layers/quantization/deep_gemm.py` modified +1/-1 (2 lines); hunks: sm_version = get_device_sm()
- Optimization/support interpretation: The concrete diff surface is `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py`; keywords observed in patches: quant, attention, doc, eagle, fp8, spec. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5707 - [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1

- Link: https://github.com/sgl-project/sglang/pull/5707
- Status/date: `merged`, created 2025-04-24, merged 2025-04-24; author `guoyuhong`.
- Diff scope read: `2` files, `+68/-15`; areas: model wrapper; keywords: config, expert, fp8, kv, moe, processor, quant, attention, awq, cuda.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +50/-1 (51 lines); hunks: # ==============================================================================; from vllm._custom_ops import awq_dequantize; symbols: DeepseekModelNextN, __init__, __init__, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-14 (32 lines); hunks: def __init__(; def __init__(; symbols: __init__, determine_n_share_experts_fusion, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, expert, fp8, kv, moe, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5716 - perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling

- Link: https://github.com/sgl-project/sglang/pull/5716
- Status/date: `merged`, created 2025-04-24, merged 2025-04-27; author `saltyfish66`.
- Diff scope read: `1` files, `+27/-27`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +27/-27 (54 lines); hunks: "BLOCK_SIZE_K": 128,; },
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5740 - update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions.

- Link: https://github.com/sgl-project/sglang/pull/5740
- Status/date: `merged`, created 2025-04-25, merged 2025-04-25; author `BBuf`.
- Diff scope read: `2` files, `+45/-42`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, moe, triton, benchmark, fp8.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +41/-41 (82 lines); hunks: {; "BLOCK_SIZE_M": 64,
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +4/-1 (5 lines); hunks: def get_moe_configs(; symbols: get_moe_configs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; keywords observed in patches: config, moe, triton, benchmark, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5748 - Fuse MLA set kv cache kernel

- Link: https://github.com/sgl-project/sglang/pull/5748
- Status/date: `merged`, created 2025-04-25, merged 2025-04-27; author `ispobock`.
- Diff scope read: `4` files, `+100/-9`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, kv, cache, mla, flash, lora, triton.
- Code diff details:
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +87/-0 (87 lines); hunks: import numpy as np; def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):; symbols: copy_two_array, set_mla_kv_buffer_kernel, set_mla_kv_buffer_triton, MLATokenToKVPool
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +6/-4 (10 lines); hunks: def forward_extend(; def forward_extend(; symbols: forward_extend, forward_extend, forward_decode, forward_decode
  - `python/sglang/srt/layers/radix_attention.py` modified +5/-2 (7 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-3 (5 lines); hunks: def forward_absorb(; symbols: forward_absorb
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`; keywords observed in patches: attention, kv, cache, mla, flash, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5793 - Auto set draft model path for MTP

- Link: https://github.com/sgl-project/sglang/pull/5793
- Status/date: `merged`, created 2025-04-27, merged 2025-04-29; author `ispobock`.
- Diff scope read: `6` files, `+115/-287`; areas: model wrapper, scheduler/runtime, docs/config; keywords: config, kv, cache, quant, spec, awq, cuda, expert, lora, moe.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-257 (258 lines); hunks: def forward(; symbols: forward, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-17 (91 lines); hunks: def forward(; def post_load_weights(self):; symbols: forward, post_load_weights, post_load_weights, post_load_weights
  - `python/sglang/srt/server_args.py` modified +21/-11 (32 lines); hunks: import tempfile; def __post_init__(self):; symbols: __post_init__, __post_init__, __call__, auto_choose_speculative_params
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-2 (13 lines); hunks: def profile_max_num_token(self, total_gpu_memory: int):; def init_memory_pool(; symbols: profile_max_num_token, init_memory_pool
  - `python/sglang/srt/configs/model_config.py` modified +7/-0 (7 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, kv, cache, quant, spec, awq. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5952 - Update ci test and doc for MTP api change

- Link: https://github.com/sgl-project/sglang/pull/5952
- Status/date: `merged`, created 2025-05-01, merged 2025-05-01; author `ispobock`.
- Diff scope read: `6` files, `+66/-14`; areas: attention/backend, tests/benchmarks, docs/config; keywords: spec, eagle, topk, mla, test, flash, kv, attention, cache, config.
- Code diff details:
  - `test/srt/test_mla_deepseek_v3.py` modified +57/-0 (57 lines); hunks: def test_gsm8k(self):; symbols: test_gsm8k, TestDeepseekV3MTP, setUpClass, tearDownClass
  - `python/sglang/srt/server_args.py` modified +7/-4 (11 lines); hunks: def __post_init__(self):; symbols: __post_init__
  - `docs/references/deepseek.md` modified +2/-4 (6 lines); hunks: The precompilation process typically takes around 10 minutes to complete.
  - `test/srt/test_full_deepseek_v3.py` modified +0/-2 (2 lines); hunks: def setUpClass(cls):; symbols: setUpClass
  - `test/srt/test_mla_flashinfer.py` modified +0/-2 (2 lines); hunks: def setUpClass(cls):; symbols: setUpClass
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md`; keywords observed in patches: spec, eagle, topk, mla, test, flash. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #5977 - Overlap qk norm with two streams

- Link: https://github.com/sgl-project/sglang/pull/5977
- Status/date: `merged`, created 2025-05-02, merged 2025-05-02; author `ispobock`.
- Diff scope read: `1` files, `+26/-6`; areas: model wrapper; keywords: attention, cache, config, cuda, kv, lora, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-6 (32 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, forward_absorb, forward_absorb
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, cache, config, cuda, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6034 - Update doc for MLA attention backends

- Link: https://github.com/sgl-project/sglang/pull/6034
- Status/date: `merged`, created 2025-05-05, merged 2025-05-08; author `Fridge003`.
- Diff scope read: `2` files, `+3/-3`; areas: docs/config; keywords: attention, cache, doc, flash, kv, mla, spec, triton, config, cuda.
- Code diff details:
  - `docs/references/deepseek.md` modified +2/-2 (4 lines); hunks: Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/be; Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--specula
  - `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `docs/references/deepseek.md`, `docs/backend/server_arguments.md`; keywords observed in patches: attention, cache, doc, flash, kv, mla. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/references/deepseek.md`, `docs/backend/server_arguments.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6081 - feat: mtp support dp-attention

- Link: https://github.com/sgl-project/sglang/pull/6081
- Status/date: `merged`, created 2025-05-07, merged 2025-06-17; author `u4lr451`.
- Diff scope read: `22` files, `+636/-146`; areas: model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks; keywords: spec, attention, eagle, cuda, config, cache, kv, topk, mla, flash.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker.py` modified +125/-39 (164 lines); hunks: import torch; def draft_tp_context(tp_group: GroupCoordinator):; symbols: draft_tp_context, __init__, forward_batch_speculative_generation, check_forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +103/-11 (114 lines); hunks: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +97/-12 (109 lines); hunks: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/speculative/eagle_utils.py` modified +74/-4 (78 lines); hunks: from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode; class EagleDraftInput:; symbols: EagleDraftInput:, prepare_for_extend, prepare_for_extend, create_idle_input
  - `test/srt/test_dp_attention.py` modified +72/-0 (72 lines); hunks: import unittest; def test_mgsm_en(self):; symbols: test_mgsm_en, TestDPAttentionDP2TP2DeepseekV3MTP, setUpClass, tearDownClass
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`; keywords observed in patches: spec, attention, eagle, cuda, config, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6109 - [Feat] Support FlashMLA backend with MTP and FP8 KV cache

- Link: https://github.com/sgl-project/sglang/pull/6109
- Status/date: `merged`, created 2025-05-08, merged 2025-05-15; author `quinnrong94`.
- Diff scope read: `8` files, `+444/-87`; areas: attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: attention, flash, mla, cache, cuda, eagle, kv, spec, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +340/-78 (418 lines); hunks: """; # FlashMLA only supports pagesize=64; symbols: __init__, FlashMLABackend, __init__, __init__
  - `test/srt/test_flashmla.py` modified +68/-0 (68 lines); hunks: import unittest; DEFAULT_MODEL_NAME_FOR_TEST_MLA,; symbols: test_latency, TestFlashMLAMTP, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +13/-0 (13 lines); hunks: def init_attention_backend(self):; symbols: init_attention_backend
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +8/-4 (12 lines); hunks: def forward_extend(; def forward_extend(; symbols: forward_extend, forward_extend, forward_decode, __init__
  - `docs/backend/attention_backend.md` modified +7/-1 (8 lines); hunks: \| **FA3** \| ✅ \| ✅ \| ✅ \| ✅ \| ✅ \|; python3 -m sglang.launch_server --tp 8 --model deepseek-ai/D
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py`; keywords observed in patches: attention, flash, mla, cache, cuda, eagle. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6151 - [Feat] optimize Qwen3 on H20 by hybrid Attention Backend

- Link: https://github.com/sgl-project/sglang/pull/6151
- Status/date: `closed`, created 2025-05-09, closed 2025-05-18; author `TianQiLin666666`.
- Diff scope read: `3` files, `+39/-9`; areas: kernel, scheduler/runtime; keywords: attention, flash, kv, spec, cache, config, cuda, lora, mla, processor.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +24/-4 (28 lines); hunks: def __init__(; def init_attention_backend(self):; symbols: __init__, init_attention_backend, init_double_sparsity_channel_config, apply_torch_tp
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-5 (14 lines); hunks: def __init__(self, model_runner: ModelRunner):; def capture_one_batch_size(self, bs: int, forward: Callable):; symbols: __init__, capture_one_batch_size, capture_one_batch_size, replay_prepare
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: attention, flash, kv, spec, cache, config. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6220 - Fuse routed scaling factor in topk_reduce kernel

- Link: https://github.com/sgl-project/sglang/pull/6220
- Status/date: `merged`, created 2025-05-12, merged 2025-06-07; author `BBuf`.
- Diff scope read: `10` files, `+331/-9`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks; keywords: moe, quant, config, expert, router, triton, benchmark, cuda, topk, cache.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py` added +199/-0 (199 lines); hunks: +import torch; symbols: _moe_sum_reduce_kernel, moe_sum_reduce, compute_sum_scaled_baseline, compute_sum_scaled_compiled
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +124/-8 (132 lines); hunks: def inplace_fused_experts(; def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunks: def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: def forward_cuda(; symbols: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: moe, quant, config, expert, router, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6890 - Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa

- Link: https://github.com/sgl-project/sglang/pull/6890
- Status/date: `merged`, created 2025-06-05, merged 2025-06-05; author `fzyzcjy`.
- Diff scope read: `1` files, `+2/-2`; areas: quantization; keywords: fp8, quant, triton.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2 (4 lines); hunks: def deepgemm_w8a8_block_fp8_linear_with_fallback(; symbols: deepgemm_w8a8_block_fp8_linear_with_fallback
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: fp8, quant, triton. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #6970 - Fuse routed scaling factor in deepseek

- Link: https://github.com/sgl-project/sglang/pull/6970
- Status/date: `merged`, created 2025-06-08, merged 2025-06-08; author `BBuf`.
- Diff scope read: `10` files, `+338/-15`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks; keywords: moe, quant, cuda, config, expert, router, triton, benchmark, topk, cache.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py` added +199/-0 (199 lines); hunks: +import torch; symbols: _moe_sum_reduce_kernel, moe_sum_reduce, compute_sum_scaled_baseline, compute_sum_scaled_compiled
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +130/-14 (144 lines); hunks: def inplace_fused_experts(; def inplace_fused_experts(; symbols: inplace_fused_experts, inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunks: def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:; symbols: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunks: def forward_cuda(; symbols: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: moe, quant, cuda, config, expert, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7146 - Support new DeepGEMM format in per token group quant

- Link: https://github.com/sgl-project/sglang/pull/7146
- Status/date: `merged`, created 2025-06-13, merged 2025-06-13; author `fzyzcjy`.
- Diff scope read: `5` files, `+92/-44`; areas: quantization, kernel, tests/benchmarks; keywords: fp8, quant, cuda, test.
- Code diff details:
  - `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +83/-40 (123 lines); hunks: __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {; __global__ void per_token_group_quant_8bit_kernel(; symbols: void, void, void
  - `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +4/-1 (5 lines); hunks: def sglang_per_token_group_quant_8bit(; symbols: sglang_per_token_group_quant_8bit
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunks: void sgl_per_token_group_quant_fp8(
  - `sgl-kernel/python/sgl_kernel/gemm.py` modified +2/-1 (3 lines); hunks: def sgl_per_token_group_quant_fp8(; symbols: sgl_per_token_group_quant_fp8
  - `sgl-kernel/csrc/common_extension.cc` modified +1/-1 (2 lines); hunks: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h`; keywords observed in patches: fp8, quant, cuda, test. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7150 - Refactor DeepGEMM integration

- Link: https://github.com/sgl-project/sglang/pull/7150
- Status/date: `merged`, created 2025-06-13, merged 2025-06-14; author `fzyzcjy`.
- Diff scope read: `12` files, `+207/-147`; areas: model wrapper, MoE/router, quantization, kernel, scheduler/runtime, docs/config; keywords: quant, config, fp8, deepep, expert, moe, topk, cuda, triton, attention.
- Code diff details:
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py` renamed +22/-76 (98 lines); hunks: from enum import IntEnum, auto; def get_enable_jit_deepgemm():; symbols: get_enable_jit_deepgemm, get_enable_jit_deepgemm, DeepGemmKernelHelper:, _compile_warning_1
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` added +95/-0 (95 lines); hunks: +import logging; symbols: grouped_gemm_nt_f8f8bf16_masked, grouped_gemm_nt_f8f8bf16_contig, gemm_nt_f8f8bf16, update_deep_gemm_config
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +35/-32 (67 lines); hunks: import logging; from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported; symbols: create_weights, __init__, forward, forward_deepgemm_contiguous
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` added +26/-0 (26 lines); hunks: +import logging; symbols: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +6/-10 (16 lines); hunks: import triton; sgl_per_token_quant_fp8,; symbols: is_fp8_fnuz, deep_gemm_fp8_fp8_bf16_nt, deep_gemm_fp8_fp8_bf16_nt, deep_gemm_fp8_fp8_bf16_nt_fake
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: quant, config, fp8, deepep, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7155 - Support new DeepGEMM format in per token group quant (part 2: srt)

- Link: https://github.com/sgl-project/sglang/pull/7155
- Status/date: `merged`, created 2025-06-13, merged 2025-06-13; author `fzyzcjy`.
- Diff scope read: `3` files, `+19/-4`; areas: quantization, kernel; keywords: config, cuda, flash, fp8, quant, test.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +17/-2 (19 lines); hunks: def sglang_per_token_group_quant_fp8(; def sglang_per_token_group_quant_fp8(; symbols: sglang_per_token_group_quant_fp8, sglang_per_token_group_quant_fp8
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunks: runtime_common = [
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`; keywords observed in patches: config, cuda, flash, fp8, quant, test. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7156 - Re-quantize DeepSeek model weights to support DeepGEMM new input format

- Link: https://github.com/sgl-project/sglang/pull/7156
- Status/date: `merged`, created 2025-06-13, merged 2025-06-13; author `fzyzcjy`.
- Diff scope read: `3` files, `+125/-0`; areas: model wrapper, quantization; keywords: fp8, quant, config, deepep, expert, kv, moe.
- Code diff details:
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +61/-0 (61 lines); hunks: import torch; def block_quant_dequant(; symbols: block_quant_dequant, requant_weight_ue8m0_inplace, _requant_weight_ue8m0, _transform_scale
  - `python/sglang/srt/models/deepseek_v2.py` modified +56/-0 (56 lines); hunks: block_quant_to_tensor_quant,; def post_load_weights(self, is_nextn=False, weight_names=None):; symbols: post_load_weights, _weight_requant_ue8m0, load_weights
  - `python/sglang/math_utils.py` added +8/-0 (8 lines); hunks: +# COPIED FROM DeepGEMM; symbols: align, ceil_div
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py`; keywords observed in patches: fp8, quant, config, deepep, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7172 - Support new DeepGEMM

- Link: https://github.com/sgl-project/sglang/pull/7172
- Status/date: `merged`, created 2025-06-14, merged 2025-06-14; author `fzyzcjy`.
- Diff scope read: `8` files, `+59/-19`; areas: model wrapper, MoE/router, quantization, kernel, docs/config; keywords: fp8, quant, moe, expert, topk, triton, config, deepep, processor.
- Code diff details:
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` modified +18/-8 (26 lines); hunks: if ENABLE_JIT_DEEPGEMM:; symbols: grouped_gemm_nt_f8f8bf16_masked
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +20/-3 (23 lines); hunks: def prepare_block_fp8_matmul_inputs(; symbols: prepare_block_fp8_matmul_inputs
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunks: def forward_deepgemm_masked(; symbols: forward_deepgemm_masked
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` modified +7/-1 (8 lines); hunks: def _compute_enable_deep_gemm():; symbols: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-4 (5 lines); hunks: import triton; def fused_moe_kernel(; symbols: fused_moe_kernel, ceil_div, moe_align_block_size_stage1
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: fp8, quant, moe, expert, topk, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7376 - Fix MTP with Deepseek R1 Fp4

- Link: https://github.com/sgl-project/sglang/pull/7376
- Status/date: `merged`, created 2025-06-20, merged 2025-06-24; author `pyc96`.
- Diff scope read: `3` files, `+20/-1`; areas: model wrapper, MoE/router, kernel; keywords: config, quant, kv, moe, awq, cache, expert, flash, fp4, triton.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; symbols: load_weights, load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +6/-0 (6 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: config, quant, kv, moe, awq, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #7762 - feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode

- Link: https://github.com/sgl-project/sglang/pull/7762
- Status/date: `merged`, created 2025-07-04, merged 2025-07-07; author `yangsijia-celina`.
- Diff scope read: `10` files, `+1006/-9`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, quant, expert, fp8, config, topk, triton, router, cuda, flash.
- Code diff details:
  - `python/sglang/test/test_cutlass_w4a8_moe.py` added +281/-0 (281 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: pack_int4_values_to_int8, pack_interleave, test_cutlass_w4a8_moe, cutlass_moe
  - `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunks: +import logging; symbols: W4AFp8Config, for, __init__, get_name
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunks: ); moe_ep_deepgemm_preprocess,; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunks: def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):; def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):; symbols: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_moe_ep_preproess, run_cutlass_moe_ep_preproess
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`; keywords observed in patches: moe, quant, expert, fp8, config, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8118 - [feat] Support tp mode for DeepSeek-R1-W4AFP8

- Link: https://github.com/sgl-project/sglang/pull/8118
- Status/date: `merged`, created 2025-07-17, merged 2025-09-02; author `chenxijun1029`.
- Diff scope read: `11` files, `+291/-120`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: expert, moe, fp8, config, quant, spec, topk, cuda, test, triton.
- Code diff details:
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +206/-60 (266 lines); hunks: #include <cudaTypedefs.h>; void dispatch_w4a8_moe_mm_sm90(; symbols: Sched, SM90W4A8Config, JOIN_STRUCT_NAME, JOIN_STRUCT_NAME_CO
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunks: from __future__ import annotations; def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales
  - `python/sglang/test/test_cutlass_w4a8_moe.py` modified +24/-9 (33 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Ten; symbols: pack_int4_values_to_int8, pack_interleave, pack_interleave, pack_interleave
  - `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py` modified +10/-4 (14 lines); hunks: def pack_interleave(num_experts, ref_weight, ref_scale):; def test_int4_fp8_grouped_gemm_single_expert(batch_size):; symbols: pack_interleave, test_int4_fp8_grouped_gemm_single_expert, test_int4_fp8_grouped_gemm_multi_experts
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` modified +7/-6 (13 lines); hunks: using MmaType = cutlass::float_e4m3_t; // FP8 e4m3 type; static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;; symbols: int, cutlass_3x_w4a8_group_gemm, int, int
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`; keywords observed in patches: expert, moe, fp8, config, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8247 - [1/N]Support DeepSeek-R1 w4a8 normal deepep

- Link: https://github.com/sgl-project/sglang/pull/8247
- Status/date: `merged`, created 2025-07-22, merged 2025-10-15; author `ayrnb`.
- Diff scope read: `7` files, `+334/-7`; areas: MoE/router, quantization, tests/benchmarks; keywords: deepep, moe, fp8, quant, config, expert, flash, fp4, topk, attention.
- Code diff details:
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +196/-0 (196 lines); hunks: # SPDX-License-Identifier: Apache-2.0; ); symbols: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal
  - `test/srt/quant/test_w4a8_deepseek_v3.py` modified +55/-0 (55 lines); hunks: def test_gsm8k(; symbols: test_gsm8k, TestDeepseekV3W4Afp8DeepepNormal, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +47/-1 (48 lines); hunks: from __future__ import annotations; if TYPE_CHECKING:; symbols: apply, apply_deepep_normal
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +21/-2 (23 lines); hunks: CUTEDSL_MOE_NVFP4_DISPATCH,; def __init__(; symbols: __init__, __init__, moe_impl, forward_flashinfer_cutedsl
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +10/-4 (14 lines); hunks: DispatchOutput,; def dispatch_a(; symbols: dispatch_a, _dispatch_core, _dispatch_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; keywords observed in patches: deepep, moe, fp8, quant, config, expert. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8464 - [2/N]Support DeepSeek-R1 w4a8 low latency deepep

- Link: https://github.com/sgl-project/sglang/pull/8464
- Status/date: `merged`, created 2025-07-28, merged 2025-10-25; author `ayrnb`.
- Diff scope read: `8` files, `+531/-9`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: fp8, moe, quant, deepep, expert, topk, attention, config, cuda, test.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +194/-0 (194 lines); hunks: def zero_experts_compute_triton(; symbols: zero_experts_compute_triton, compute_problem_sizes_w4a8_kernel, compute_problem_sizes_w4a8, deepep_ll_get_cutlass_w4a8_moe_mm_data
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +138/-0 (138 lines); hunks: ); def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` modified +72/-6 (78 lines); hunks: __global__ void int4_fp8_get_group_gemm_starts(; __global__ void int4_fp8_get_group_gemm_starts(; symbols: void, void, void
  - `test/srt/quant/test_w4a8_deepseek_v3.py` modified +69/-0 (69 lines); hunks: +import os; def test_gsm8k(; symbols: test_gsm8k, TestDeepseekV3W4Afp8DeepepAutoMtp, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +36/-0 (36 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE; def apply(; symbols: apply, apply_deepep_ll, apply_deepep_normal
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh`; keywords observed in patches: fp8, moe, quant, deepep, expert, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10027 - [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels

- Link: https://github.com/sgl-project/sglang/pull/10027
- Status/date: `merged`, created 2025-09-04, merged 2025-11-24; author `yuhyao`.
- Diff scope read: `3` files, `+253/-77`; areas: MoE/router, quantization, kernel; keywords: deepep, moe, quant, config, expert, fp8, topk, triton, cuda, processor.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +227/-54 (281 lines); hunks: import triton.language as tl; def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):; symbols: _get_launch_config_1d, get_num_blocks, _get_launch_config_2d, get_num_blocks
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +21/-17 (38 lines); hunks: silu_and_mul,; def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe, cutlass_w4a8_moe, cutlass_w4a8_moe, cutlass_w4a8_moe
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +5/-6 (11 lines); hunks: def process_weights_after_loading(self, layer: Module) -> None:; def apply(; symbols: process_weights_after_loading, apply, apply_deepep_ll
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; keywords observed in patches: deepep, moe, quant, config, expert, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10361 - Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile

- Link: https://github.com/sgl-project/sglang/pull/10361
- Status/date: `merged`, created 2025-09-12, merged 2025-09-12; author `kkHuang-amd`.
- Diff scope read: `2` files, `+39/-5`; areas: attention/backend, multimodal/processor; keywords: attention, processor.
- Code diff details:
  - `python/sglang/srt/layers/dp_attention.py` modified +24/-0 (24 lines); hunks: def get_local_dp_buffer_len(cls) -> int:; def get_dp_global_num_tokens() -> List[int]:; symbols: get_local_dp_buffer_len, get_dp_global_num_tokens, get_dp_hidden_size, get_dp_dtype
  - `python/sglang/srt/layers/logits_processor.py` modified +15/-5 (20 lines); hunks: get_attention_dp_rank,; def compute_dp_attention_metadata(self):; symbols: compute_dp_attention_metadata, _get_logits
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`; keywords observed in patches: attention, processor. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11512 - Update DeepSeek-R1-FP4 default config on blackwell

- Link: https://github.com/sgl-project/sglang/pull/11512
- Status/date: `merged`, created 2025-10-12, merged 2025-10-13; author `Qiaolin-Yu`.
- Diff scope read: `1` files, `+26/-1`; areas: misc; keywords: attention, config, cuda, flash, fp4, kv, mla, moe, quant, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +26/-1 (27 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: attention, config, cuda, flash, fp4, kv. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11708 - Support running FP4 Deepseek on SM120.

- Link: https://github.com/sgl-project/sglang/pull/11708
- Status/date: `merged`, created 2025-10-16, merged 2025-10-28; author `weireweire`.
- Diff scope read: `9` files, `+33/-35`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks; keywords: cuda, flash, kv, attention, cache, moe, spec, config, fp4, fp8.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +8/-7 (15 lines); hunks: from sglang.srt.layers.quantization.fp8_utils import (; ); symbols: apply, ModelOptNvFp4FusedMoEMethod, __init__, apply
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-10 (11 lines); hunks: enable_fused_set_kv_buffer,
  - `python/sglang/srt/utils/common.py` modified +10/-1 (11 lines); hunks: def device_context(device: torch.device):; symbols: device_context, is_blackwell, is_blackwell_supported
  - `python/sglang/srt/server_args.py` modified +4/-3 (7 lines); hunks: get_device,; def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_attention_backend_compatibility
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-5 (6 lines); hunks: get_int_env_var,; else:; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py`; keywords observed in patches: cuda, flash, kv, attention, cache, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12000 - [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU

- Link: https://github.com/sgl-project/sglang/pull/12000
- Status/date: `merged`, created 2025-10-23, merged 2025-10-24; author `zminglei`.
- Diff scope read: `3` files, `+64/-5`; areas: model wrapper; keywords: attention, flash, spec, triton, cache, config, cuda, kv, mla.
- Code diff details:
  - `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +31/-2 (33 lines); hunks: def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype \| None =; def enable_batch_invariant_mode():; symbols: mean_batch_invariant, bmm_batch_invariant, is_batch_invariant_mode_enabled, enable_batch_invariant_mode
  - `python/sglang/srt/server_args.py` modified +24/-2 (26 lines); hunks: def _handle_deterministic_inference(self):; def _handle_deterministic_inference(self):; symbols: _handle_deterministic_inference, _handle_deterministic_inference
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunks: def handle_attention_flashinfer(attn, forward_batch):; def handle_attention_nsa(attn, forward_batch):; symbols: handle_attention_flashinfer, handle_attention_fa3, handle_attention_flashmla, handle_attention_nsa
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: attention, flash, spec, triton, cache, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12057 - [doc] add example of using w4fp8 for Deepseek

- Link: https://github.com/sgl-project/sglang/pull/12057
- Status/date: `merged`, created 2025-10-24, merged 2025-10-27; author `Kevin-XiongC`.
- Diff scope read: `1` files, `+15/-0`; areas: tests/benchmarks; keywords: benchmark, config, expert, fp8, moe, quant.
- Code diff details:
  - `benchmark/deepseek_v3/README.md` modified +15/-0 (15 lines); hunks: edit your `config.json` and remove the `quantization_config` block. For example:
- Optimization/support interpretation: The concrete diff surface is `benchmark/deepseek_v3/README.md`; keywords observed in patches: benchmark, config, expert, fp8, moe, quant. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/deepseek_v3/README.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12778 - Update dsv3 quantization auto setting for sm100

- Link: https://github.com/sgl-project/sglang/pull/12778
- Status/date: `merged`, created 2025-11-06, merged 2025-11-06; author `ispobock`.
- Diff scope read: `1` files, `+22/-9`; areas: misc; keywords: config, flash, fp4, fp8, kv, moe, quant, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +22/-9 (31 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`; keywords observed in patches: config, flash, fp4, fp8, kv, moe. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12921 - [perf]optimize w4afp8 kernel on deepseek-v3-0324

- Link: https://github.com/sgl-project/sglang/pull/12921
- Status/date: `merged`, created 2025-11-09, merged 2025-12-18; author `Bruce-x-1997`.
- Diff scope read: `3` files, `+160/-264`; areas: MoE/router, quantization, kernel; keywords: moe, topk, expert, config, cuda, fp8, quant.
- Code diff details:
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +64/-236 (300 lines); hunks: inline void invoke_gemm(; void dispatch_w4a8_moe_mm_sm90(; symbols: parameters, parameter
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` modified +96/-27 (123 lines); hunks: #include <cudaTypedefs.h>; __global__ void compute_problem_sizes_w4a8(; symbols: uint64_t, void, void, void
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +0/-1 (1 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`; keywords observed in patches: moe, topk, expert, config, cuda, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13548 - [Fix] Fix DeepSeek V3 MTP on B200

- Link: https://github.com/sgl-project/sglang/pull/13548
- Status/date: `merged`, created 2025-11-19, merged 2025-11-19; author `Fridge003`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: config, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`; keywords observed in patches: config, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14162 - DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- Link: https://github.com/sgl-project/sglang/pull/14162
- Status/date: `merged`, created 2025-11-30, merged 2026-03-30; author `xieminghe1`.
- Diff scope read: `5` files, `+94/-12`; areas: MoE/router, quantization, kernel; keywords: fp8, moe, quant, deepep, config, topk, triton, expert, fp4.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: def forward_cutlass_w4afp8(; def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: def _dispatch_core(; symbols: _dispatch_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: fp8, moe, quant, deepep, config, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14897 - Fix dsv3 dp accuracy issue when using bf16-kv

- Link: https://github.com/sgl-project/sglang/pull/14897
- Status/date: `merged`, created 2025-12-11, merged 2025-12-11; author `Duyi-Wang`.
- Diff scope read: `1` files, `+8/-2`; areas: attention/backend; keywords: attention, cache, fp8, kv, mla.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +8/-2 (10 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`; keywords observed in patches: attention, cache, fp8, kv, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15304 - Fix the accuracy issue when running mxfp4 dsv3 model and enable ep

- Link: https://github.com/sgl-project/sglang/pull/15304
- Status/date: `merged`, created 2025-12-17, merged 2025-12-17; author `kkHuang-amd`.
- Diff scope read: `2` files, `+2/-0`; areas: MoE/router, quantization; keywords: expert, quant, fp4, moe.
- Code diff details:
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +1/-0 (1 lines); hunks: def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunks: def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py`; keywords observed in patches: expert, quant, fp4, moe. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15347 - Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

- Link: https://github.com/sgl-project/sglang/pull/15347
- Status/date: `merged`, created 2025-12-18, merged 2026-01-19; author `leejnau`.
- Diff scope read: `3` files, `+165/-12`; areas: MoE/router, kernel, tests/benchmarks; keywords: cuda, expert, moe, test, topk, config, flash, spec.
- Code diff details:
  - `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunks: +import pytest; symbols: test_fused_topk_deepseek
  - `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunks: if _is_cuda:; def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu, biased_grouped_topk_gpu
  - `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunks: def test_a_gsm8k(; def test_bs_1_speed(self):; symbols: test_a_gsm8k, test_bs_1_speed, test_bs_1_speed
- Optimization/support interpretation: The concrete diff surface is `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py`; keywords observed in patches: cuda, expert, moe, test, topk, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15531 - Support piecewise cuda graph for dsv3 fp4

- Link: https://github.com/sgl-project/sglang/pull/15531
- Status/date: `merged`, created 2025-12-20, merged 2025-12-21; author `ispobock`.
- Diff scope read: `7` files, `+148/-16`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks; keywords: flash, fp4, attention, cuda, quant, cache, mla, config, expert, moe.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +64/-10 (74 lines); hunks: StandardDispatcher,; def _quantize_hidden_states_fp4(self, hidden_states: torch.Tensor):; symbols: _quantize_hidden_states_fp4, forward, moe_forward_piecewise_cuda_graph_impl_fake, flashinfer_fp4_moe_forward_piecewise_cuda_graph_impl
  - `test/srt/test_deepseek_v3_fp4_4gpu.py` modified +67/-0 (67 lines); hunks: def test_bs_1_speed(self):; symbols: test_bs_1_speed, TestDeepseekV3FP4PiecewiseCudaGraph, setUpClass, tearDownClass
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunks: import concurrent.futures; def handle_attention_fa4(attn, forward_batch):; symbols: handle_attention_fa4, handle_attention_trtllm_mla, forward
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +3/-1 (4 lines); hunks: import triton; def init_forward_metadata(self, forward_batch: ForwardBatch):; symbols: init_forward_metadata
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +1/-2 (3 lines); hunks: ); def apply_weights(; symbols: apply_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: flash, fp4, attention, cuda, quant, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16649 - [Refactor] Split out deepseek v2 weight loader function into mixin

- Link: https://github.com/sgl-project/sglang/pull/16649
- Status/date: `merged`, created 2026-01-07, merged 2026-01-18; author `xyjixyjixyji`.
- Diff scope read: `4` files, `+721/-600`; areas: model wrapper; keywords: cuda, fp8, moe, awq, config, fp4, kv, quant, spec, triton.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` added +657/-0 (657 lines); hunks: +# Copyright 2026 SGLang Team; symbols: DeepseekV2WeightLoaderMixin:, do_load_weights, post_load_weights, _maybe_quant_weights_to_fp8_ue8m0
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-594 (603 lines); hunks: """Inference-only DeepseekV2 model."""; per_tensor_quant_mla_fp8,; symbols: enable_nextn_moe_bf16_cast_to_fp8, forward, DeepseekV2ForCausalLM, DeepseekV2ForCausalLM
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +53/-1 (54 lines); hunks: +# Copyright 2026 SGLang Team; _is_cpu = is_cpu(); symbols: awq_dequantize_func, enable_nextn_moe_bf16_cast_to_fp8
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-5 (7 lines); hunks: VocabParallelEmbedding,
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`; keywords observed in patches: cuda, fp8, moe, awq, config, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- Link: https://github.com/sgl-project/sglang/pull/17133
- Status/date: `merged`, created 2026-01-15, merged 2026-01-16; author `xu-yfei`.
- Diff scope read: `6` files, `+959/-217`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, triton, config, fp8, benchmark, cache, cuda, expert, quant, router.
- Code diff details:
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215 (552 lines); hunks: # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py; sort_config,; symbols: MoeInputs:, KernelWrapper:, __init__, cuda_graph_wrapper
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; keywords observed in patches: moe, triton, config, fp8, benchmark, cache. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- Link: https://github.com/sgl-project/sglang/pull/17178
- Status/date: `merged`, created 2026-01-16, merged 2026-01-16; author `hlu1`.
- Diff scope read: `1` files, `+3/-2`; areas: tests/benchmarks; keywords: spec, test.
- Code diff details:
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunks: def get_thinking_kwargs(args):; def run_eval(args):; symbols: get_thinking_kwargs, run_eval, run_eval
- Optimization/support interpretation: The concrete diff surface is `python/sglang/test/run_eval.py`; keywords observed in patches: spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/test/run_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17707 - Add dsv3 router gemm benchmark on blackwell

- Link: https://github.com/sgl-project/sglang/pull/17707
- Status/date: `merged`, created 2026-01-25, merged 2026-04-04; author `harrisonlimh`.
- Diff scope read: `2` files, `+284/-4`; areas: model wrapper, MoE/router, kernel, tests/benchmarks; keywords: cuda, flash, router, attention, benchmark, config, fp8, kv, mla, quant.
- Code diff details:
  - `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py` added +250/-0 (250 lines); hunks: +import argparse; symbols: create_benchmark_configs, dsv3_router_gemm_flashinfer, dsv3_router_gemm_sgl, check_accuracy
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-4 (38 lines); hunks: pass; def forward(; symbols: forward, DeepseekV32ForCausalLM, flashinfer_dsv3_router_gemm
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: cuda, flash, router, attention, benchmark, config. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17744 - Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization

- Link: https://github.com/sgl-project/sglang/pull/17744
- Status/date: `merged`, created 2026-01-26, merged 2026-01-31; author `hsuchifeng`.
- Diff scope read: `1` files, `+16/-12`; areas: model wrapper; keywords: config, fp8, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +16/-12 (28 lines); hunks: def _maybe_quant_weights_to_fp8_ue8m0(; def _maybe_quant_weights_to_fp8_ue8m0(; symbols: _maybe_quant_weights_to_fp8_ue8m0, _maybe_quant_weights_to_fp8_ue8m0
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; keywords observed in patches: config, fp8, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18242 - [ROCm] Optimize Deepseek R1 on MI300X

- Link: https://github.com/sgl-project/sglang/pull/18242
- Status/date: `merged`, created 2026-02-04, merged 2026-02-25; author `zhentaocc`.
- Diff scope read: `3` files, `+7/-2`; areas: model wrapper, attention/backend, quantization; keywords: cache, cuda, fp8, attention, kv, mla, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-2 (7 lines); hunks: get_dsv3_gemm_output_zero_allocator_size,; def forward_absorb_prepare(; symbols: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-0 (1 lines); hunks: def forward_decode(; symbols: forward_decode
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunks: aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; keywords observed in patches: cache, cuda, fp8, attention, kv, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18451 - [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256.

- Link: https://github.com/sgl-project/sglang/pull/18451
- Status/date: `merged`, created 2026-02-08, merged 2026-03-19; author `amd-mvarjoka`.
- Diff scope read: `1` files, `+5/-1`; areas: model wrapper; keywords: router.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18461 - [Intel GPU] Enable DeepSeek R1 inference on XPU

- Link: https://github.com/sgl-project/sglang/pull/18461
- Status/date: `merged`, created 2026-02-09, merged 2026-03-30; author `polisettyvarma`.
- Diff scope read: `6` files, `+46/-28`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks; keywords: cuda, moe, quant, benchmark, config, expert, fp8, spec, awq, flash.
- Code diff details:
  - `benchmark/kernels/quantization/tuning_block_wise_kernel.py` modified +22/-20 (42 lines); hunks: _w8a8_block_fp8_matmul_unrolledx4,; def benchmark_config(; symbols: benchmark_config, run, run, tune
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +9/-4 (13 lines); hunks: ServerArgs,; def run():; symbols: benchmark_config, run, BenchmarkWorker:, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/standard.py` modified +9/-3 (12 lines); hunks: get_moe_runner_backend,; def dispatch(; symbols: dispatch
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunks: _is_fp8_fnuz,; def post_load_weights(; symbols: post_load_weights
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0 (2 lines); hunks: is_hip,; _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py`; keywords observed in patches: cuda, moe, quant, benchmark, config, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18607 - [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp

- Link: https://github.com/sgl-project/sglang/pull/18607
- Status/date: `merged`, created 2026-02-11, merged 2026-02-12; author `1am9trash`.
- Diff scope read: `2` files, `+9/-5`; areas: attention/backend; keywords: attention, cache, doc, fp8, kv, mla, triton.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +7/-3 (10 lines); hunks: def __init__(; def make_mla_meta_data(; symbols: __init__, make_mla_meta_data
  - `docker/rocm.Dockerfile` modified +2/-2 (4 lines); hunks: ENV BUILD_TRITON="0"; ENV BUILD_TRITON="0"
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile`; keywords observed in patches: attention, cache, doc, fp8, kv, mla. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19122 - [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py

- Link: https://github.com/sgl-project/sglang/pull/19122
- Status/date: `merged`, created 2026-02-21, merged 2026-02-27; author `Fridge003`.
- Diff scope read: `9` files, `+906/-818`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: attention, mla, cache, kv, fp8, lora, quant, cuda, triton, config.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +22/-811 (833 lines); hunks: from __future__ import annotations; MaybeTboDeepEPDispatcher,; symbols: DeepseekV2MLP, __init__, op_output, DeepseekV2AttentionMLA
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0 (492 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLAForwardMixin:, init_mla_forward, forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0 (227 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLARocmForwardMixin:, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare, forward_absorb_fused_mla_rope_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0 (152 lines); hunks: +from __future__ import annotations; symbols: DeepseekMLACpuForwardMixin:, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare, forward_absorb_fused_mla_rope_cpu_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` modified +6/-0 (6 lines); hunks: from .forward_methods import AttnForwardMethod
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; keywords observed in patches: attention, mla, cache, kv, fp8, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19425 - [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4

- Link: https://github.com/sgl-project/sglang/pull/19425
- Status/date: `merged`, created 2026-02-26, merged 2026-02-27; author `billishyahao`.
- Diff scope read: `2` files, `+18/-2`; areas: model wrapper, quantization; keywords: config, kv, cache, cuda, fp4, fp8, moe, quant.
- Code diff details:
  - `python/sglang/srt/models/deepseek_nextn.py` modified +11/-0 (11 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; class DeepseekModelNextN(nn.Module):; symbols: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-2 (9 lines); hunks: def __init__(; def get_min_capability(cls) -> int:; symbols: __init__, get_min_capability, get_name, apply_weight_name_mapper
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; keywords observed in patches: config, kv, cache, cuda, fp4, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19834 - [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek)

- Link: https://github.com/sgl-project/sglang/pull/19834
- Status/date: `merged`, created 2026-03-04, merged 2026-03-05; author `yctseng0211`.
- Diff scope read: `13` files, `+1614/-177`; areas: quantization, tests/benchmarks; keywords: test, cache, fp4, config, fp8, kv, benchmark, quant, doc, attention.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` added +281/-0 (281 lines); hunks: +"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with KV Cache FP8 (8-GPU); symbols: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` added +280/-0 (280 lines); hunks: +"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with AIter AllReduce Fusion (8-GPU); symbols: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +153/-68 (221 lines); hunks: on:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +153/-68 (221 lines); hunks: on:; jobs:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` added +178/-0 (178 lines); hunks: +"""MI35x Nightly performance benchmark for DeepSeek-R1-MXFP4 model with KV Cache FP8.; symbols: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; keywords observed in patches: test, cache, fp4, config, fp8, kv. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19843 - [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3

- Link: https://github.com/sgl-project/sglang/pull/19843
- Status/date: `merged`, created 2026-03-04, merged 2026-03-06; author `inkcherry`.
- Diff scope read: `1` files, `+12/-7`; areas: model wrapper; keywords: config, expert, flash, fp4, fp8, moe, quant, topk.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-7 (19 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: config, expert, flash, fp4, fp8, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20841 - Fix gpu-fault issue when run deepseek-r1 and enable dp

- Link: https://github.com/sgl-project/sglang/pull/20841
- Status/date: `merged`, created 2026-03-18, merged 2026-03-19; author `kkHuang-amd`.
- Diff scope read: `1` files, `+1/-1`; areas: attention/backend; keywords: attention, cuda, spec.
- Code diff details:
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunks: def init_forward_metadata_replay_cuda_graph(; symbols: init_forward_metadata_replay_cuda_graph, get_cuda_graph_seq_len_fill_value, update_verify_buffers_to_fill_after_draft
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/aiter_backend.py`; keywords observed in patches: attention, cuda, spec. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/aiter_backend.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21280 - [RL] Support mxfp8 DeepSeek V3

- Link: https://github.com/sgl-project/sglang/pull/21280
- Status/date: `merged`, created 2026-03-24, merged 2026-04-04; author `zianglih`.
- Diff scope read: `3` files, `+105/-45`; areas: attention/backend, MoE/router, quantization, kernel, scheduler/runtime; keywords: fp8, moe, quant, config, cuda, flash, fp4, topk, cache, expert.
- Code diff details:
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +86/-38 (124 lines); hunks: else:; def align_fp8_moe_weights_for_flashinfer_trtllm(; symbols: align_fp8_moe_weights_for_flashinfer_trtllm, align_fp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-7 (19 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/layers/quantization/fp8.py` modified +7/-0 (7 lines); hunks: from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput; def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, apply_weight_name_mapper, Fp8LinearMethod
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; keywords observed in patches: fp8, moe, quant, config, cuda, flash. Impact reading: attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21529 - Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm

- Link: https://github.com/sgl-project/sglang/pull/21529
- Status/date: `open`, created 2026-03-27; author `JohnQinAMD`.
- Diff scope read: `10` files, `+308/-126`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks; keywords: quant, fp4, config, expert, moe, attention, cuda, triton, cache, kv.
- Code diff details:
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +85/-44 (129 lines); hunks: if _use_aiter:; def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):; symbols: __init__, create_weights, create_weights, process_weights_after_loading
  - `test/registered/amd/test_glm5_mxfp4.py` added +114/-0 (114 lines); hunks: +"""GLM-5 MXFP4 tests (4-GPU, MI35x); symbols: _GLM5MXFP4Base, for, setUpClass, tearDownClass
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +40/-62 (102 lines); hunks: -"""Kimi-K2.5-MXFP4 aiter MLA backend test (4-GPU, FP8 KV cache); register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x"); symbols: TestKimiK25QuarkMXFP4, TestKimiK25MXFP4, setUpClass, tearDownClass
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +28/-14 (42 lines); hunks: def __init__(; def _load_w13(; symbols: __init__, _load_w13, _load_w2, weight_loader
  - `python/sglang/srt/models/deepseek_v2.py` modified +20/-2 (22 lines); hunks: def forward(; def determine_num_fused_shared_experts(; symbols: forward, DeepseekV2ForCausalLM, __init__, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; keywords observed in patches: quant, fp4, config, expert, moe, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21531 - [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel

- Link: https://github.com/sgl-project/sglang/pull/21531
- Status/date: `open`, created 2026-03-27; author `meinie0826`.
- Diff scope read: `11` files, `+450/-39`; areas: model wrapper, MoE/router, kernel, tests/benchmarks; keywords: router, cuda, expert, test, attention, awq, benchmark, cache, config, fp8.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +184/-0 (184 lines); hunks: +/*; symbols: int, int, int, __launch_bounds__
  - `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +104/-0 (104 lines); hunks: +"""Benchmark for DeepSeek V3 router GEMM (JIT kernel vs torch.nn.functional.linear).; symbols: benchmark_bf16_output, benchmark_float32_output
  - `python/sglang/jit_kernel/dsv3_router_gemm.py` added +103/-0 (103 lines); hunks: +"""; symbols: _jit_dsv3_router_gemm_module, can_use_dsv3_router_gemm, dsv3_router_gemm
  - `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py` added +36/-0 (36 lines); hunks: +"""Tests for JIT dsv3_router_gemm kernel."""; symbols: _ref, test_dsv3_router_gemm
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-6 (19 lines); hunks: pass; def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py`; keywords observed in patches: router, cuda, expert, test, attention, awq. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- Link: https://github.com/sgl-project/sglang/pull/21599
- Status/date: `merged`, created 2026-03-28, merged 2026-04-20; author `alphabetc1`.
- Diff scope read: `13` files, `+1296/-33`; areas: kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, cuda, eagle, attention, topk, cache, kv, quant, test.
- Code diff details:
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunks: +"""Benchmark adaptive speculative decoding against static baselines.; symbols: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunks: +import unittest; symbols: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunks: +import json; symbols: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunks: import logging; alloc_token_slots,; symbols: __init__, __init__, init_cuda_graphs, apply_runtime_state
  - `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0 (156 lines); hunks: +# Adaptive Speculative Decoding
- Optimization/support interpretation: The concrete diff surface is `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; keywords observed in patches: spec, config, cuda, eagle, attention, topk. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21719 - Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication"

- Link: https://github.com/sgl-project/sglang/pull/21719
- Status/date: `merged`, created 2026-03-31, merged 2026-03-31; author `BBuf`.
- Diff scope read: `5` files, `+12/-94`; areas: MoE/router, quantization, kernel; keywords: fp8, moe, quant, deepep, config, topk, triton, expert, fp4.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +0/-73 (73 lines); hunks: def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +7/-15 (22 lines); hunks: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunks: def forward_cutlass_w4afp8(; def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-2 (3 lines); hunks: def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: def _dispatch_core(; symbols: _dispatch_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: fp8, moe, quant, deepep, config, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- Link: https://github.com/sgl-project/sglang/pull/22128
- Status/date: `merged`, created 2026-04-05, merged 2026-04-17; author `narutolhy`.
- Diff scope read: `4` files, `+272/-18`; areas: kernel, scheduler/runtime, tests/benchmarks; keywords: cuda, spec, quant, attention, cache, config, eagle, expert, fp8, lora.
- Code diff details:
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunks: +"""Test piecewise CUDA graph coexisting with speculative decoding.; symbols: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunks: def _handle_piecewise_cuda_graph(self):; symbols: _handle_piecewise_cuda_graph
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunks: def can_run(self, forward_batch: ForwardBatch):; symbols: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: def init_piecewise_cuda_graphs(self):; symbols: init_piecewise_cuda_graphs
- Optimization/support interpretation: The concrete diff surface is `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; keywords observed in patches: cuda, spec, quant, attention, cache, config. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22268 - [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3

- Link: https://github.com/sgl-project/sglang/pull/22268
- Status/date: `open`, created 2026-04-07; author `SuperMarioYL`.
- Diff scope read: `1` files, `+5/-0`; areas: model wrapper; keywords: kv, lora.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: def prepare_qkv_latent(; symbols: prepare_qkv_latent
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22316 - [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- Link: https://github.com/sgl-project/sglang/pull/22316
- Status/date: `merged`, created 2026-04-08, merged 2026-04-10; author `xieminghe1`.
- Diff scope read: `5` files, `+91/-12`; areas: MoE/router, quantization, kernel; keywords: fp8, moe, quant, deepep, config, topk, triton, expert, fp4.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunks: def silu_and_mul_masked_post_per_tensor_quant_fwd(; symbols: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunks: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; symbols: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3 (3 lines); hunks: def forward_cutlass_w4afp8_masked(; symbols: forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunks: def apply_deepep_ll(; symbols: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunks: def _dispatch_core(; symbols: _dispatch_core
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; keywords observed in patches: fp8, moe, quant, deepep, config, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22323 - [Lora] Lora quat info re-factor and support deepseekv3 mla lora

- Link: https://github.com/sgl-project/sglang/pull/22323
- Status/date: `merged`, created 2026-04-08, merged 2026-04-09; author `yushengsu-thu`.
- Diff scope read: `16` files, `+458/-80`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, config, lora, triton, expert, kv, quant, mla, attention, cache.
- Code diff details:
  - `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py` added +156/-0 (156 lines); hunks: +# Copyright 2023-2025 SGLang Team; symbols: kl_v2, get_prompt_logprobs, TestLoRADeepSeekV3BaseLogprobDiff, test_lora_deepseek_v3_base_logprob_accuracy
  - `python/sglang/srt/lora/layers.py` modified +91/-7 (98 lines); hunks: ColumnParallelLinear,; def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):; symbols: slice_lora_b_weights, ReplicatedLinearWithLoRA, __init__, set_lora_info
  - `python/sglang/srt/lora/utils.py` modified +41/-3 (44 lines); hunks: def get_hidden_dim(; def get_normalized_target_modules(; symbols: get_hidden_dim, get_normalized_target_modules, get_stacked_multiply, get_target_module_name
  - `python/sglang/srt/layers/quantization/fp8.py` modified +21/-20 (41 lines); hunks: def create_moe_runner(; def apply(; symbols: create_moe_runner, get_triton_quant_info, apply, apply
  - `python/sglang/srt/lora/lora.py` modified +29/-1 (30 lines); hunks: def _process_weight(self, name: str, loaded_weight: torch.Tensor):; def normalize_gate_up_proj(; symbols: _process_weight, _normalize_weights, normalize_qkv_proj, normalize_gate_up_proj
- Optimization/support interpretation: The concrete diff surface is `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py`; keywords observed in patches: moe, config, lora, triton, expert, kv. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22933 - [CPU] expand the interface of shared_expert without scaling factor

- Link: https://github.com/sgl-project/sglang/pull/22933
- Status/date: `merged`, created 2026-04-16, merged 2026-04-21; author `mingfeima`.
- Diff scope read: `9` files, `+313/-623`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: moe, topk, expert, kv, fp8, test, quant, awq, cache, scheduler.
- Code diff details:
  - `sgl-kernel/csrc/cpu/moe_int4.cpp` modified +10/-176 (186 lines); hunks: #include "common.h"; symbols: int, int, int, int
  - `sgl-kernel/csrc/cpu/moe.h` added +173/-0 (173 lines); hunks: +#pragma once; symbols: int, int, int, int
  - `sgl-kernel/csrc/cpu/moe.cpp` modified +25/-119 (144 lines); hunks: +#include "moe.h"; namespace {; symbols: int, int, int, int64_t
  - `sgl-kernel/csrc/cpu/moe_fp8.cpp` modified +6/-136 (142 lines); hunks: #include "common.h"; void shared_expert_fp8_kernel_impl(; symbols: int, int, int
  - `sgl-kernel/csrc/cpu/moe_int8.cpp` modified +6/-108 (114 lines); hunks: #include "common.h"; void shared_expert_int8_kernel_impl(; symbols: int, int, int
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp`; keywords observed in patches: moe, topk, expert, kv, fp8, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- Link: https://github.com/sgl-project/sglang/pull/22950
- Status/date: `closed`, created 2026-04-16, closed 2026-04-21; author `Wen-xuan-Xu`.
- Diff scope read: `11` files, `+597/-64`; areas: scheduler/runtime, tests/benchmarks, docs/config; keywords: cache, kv, cuda, eagle, test, attention, spec.
- Code diff details:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunks: +import unittest; symbols: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunks: from numpy import float64; MatchPrefixParams,; symbols: cache_finished_req, _skip_cache_unfinished_req, _skip_cache_unfinished_req
  - `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14 (41 lines); hunks: MatchPrefixParams,; def cache_finished_req(self, req: Req, is_insert: bool = True):; symbols: cache_finished_req, cache_unfinished_req, cache_unfinished_req, pretty_print
  - `python/sglang/srt/mem_cache/common.py` modified +22/-0 (22 lines); hunks: def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; symbols: alloc_for_decode, maybe_strip_thinking_tokens, release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; keywords observed in patches: cache, kv, cuda, eagle, test, attention. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23195 - [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors

- Link: https://github.com/sgl-project/sglang/pull/23195
- Status/date: `open`, created 2026-04-20; author `JasonLeviGoodison`.
- Diff scope read: `4` files, `+138/-14`; areas: model wrapper, attention/backend, tests/benchmarks; keywords: attention, kv, mla, cuda, fp8, awq, config, marlin, moe, quant.
- Code diff details:
  - `test/registered/unit/models/test_deepseek_v2_attention_mla.py` added +111/-0 (111 lines); hunks: +import unittest; symbols: TestDeepseekV2AttentionMLA, _make_attn, test_get_fused_qkv_a_proj_weight_returns_none_when_missing, test_can_use_min_latency_fused_a_gemm_preserves_bf16_path
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-9 (27 lines); hunks: class DeepseekV2AttentionMLA(; def __init__(; symbols: DeepseekV2AttentionMLA, _get_fused_qkv_a_proj_weight, _can_use_min_latency_fused_a_gemm, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +5/-4 (9 lines); hunks: def init_mla_fused_rope_cpu_forward(self: DeepseekV2AttentionMLA):; symbols: init_mla_fused_rope_cpu_forward
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +4/-1 (5 lines); hunks: def _dispatch_mla_subtype(attn, forward_batch):; symbols: _dispatch_mla_subtype
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`; keywords observed in patches: attention, kv, mla, cuda, fp8, awq. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23257 - Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention

- Link: https://github.com/sgl-project/sglang/pull/23257
- Status/date: `open`, created 2026-04-20; author `yhyang201`.
- Diff scope read: `2` files, `+5/-0`; areas: model wrapper, attention/backend, MoE/router, scheduler/runtime; keywords: flash, moe, attention, config, cuda, fp4.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunks: from sglang.srt.layers.moe import (; def forward_normal_dual_stream(; symbols: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` modified +2/-0 (2 lines); hunks: def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:; symbols: ensure_cutedsl_wrapper
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`; keywords observed in patches: flash, moe, attention, config, cuda, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- Link: https://github.com/sgl-project/sglang/pull/23315
- Status/date: `merged`, created 2026-04-21, merged 2026-04-21; author `hnyls2002`.
- Diff scope read: `4` files, `+72/-4`; areas: scheduler/runtime, tests/benchmarks; keywords: cache, kv, spec, cuda, scheduler, test.
- Code diff details:
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunks: from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType; def test_cache_finished_req_insert(self):; symbols: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunks: def output_ids_through_stop(self) -> List[int]:; def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; symbols: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunks: def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = Tr; symbols: release_kv_cache
- Optimization/support interpretation: The concrete diff surface is `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: cache, kv, spec, cuda, scheduler, test. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- Link: https://github.com/sgl-project/sglang/pull/23336
- Status/date: `open`, created 2026-04-21; author `alphabetc1`.
- Diff scope read: `6` files, `+193/-10`; areas: multimodal/processor, scheduler/runtime; keywords: spec, eagle, cuda, scheduler, attention, processor, config, kv, moe, topk.
- Code diff details:
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunks: from sglang.srt.managers.schedule_batch import ModelWorkerBatch; def __init__(; symbols: __init__, __init__, target_worker, forward_batch_generation
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunks: def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; symbols: prepare_for_decode, prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunks: def _resolve_spec_overlap_token_ids(; symbols: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunks: def adaptive_unsupported_reason(server_args: ServerArgs) -> str \| None:; symbols: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunks: class GenerationBatchResult:; symbols: GenerationBatchResult:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; keywords observed in patches: spec, eagle, cuda, scheduler, attention, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
