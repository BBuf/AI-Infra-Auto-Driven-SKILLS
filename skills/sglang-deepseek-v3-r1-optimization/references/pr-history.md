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
