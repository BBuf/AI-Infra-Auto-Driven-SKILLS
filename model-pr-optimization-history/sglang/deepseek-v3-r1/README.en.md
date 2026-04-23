# SGLang DeepSeek V3 / R1 Support and Optimization Timeline

This document is based on SGLang `origin/main` snapshot `929e00eea`, sgl-cookbook `origin/main` snapshot `8ec4d03`, and patch-level reading of DeepSeek V3/R1 merged, open, and reverted PRs. The scope only covers DeepSeek V3, V3-0324, R1, R1-0528, and their quantization, MTP, DeepEP, LoRA, and backend optimization tracks. DeepSeek V3.1 parser/template differences and DeepSeek V3.2 DSA/NSA sparse attention are documented separately.

Conclusion: as of `929e00eea`, the main DeepSeek V3/R1 runtime entry is still `DeepseekV3ForCausalLM` in `python/sglang/srt/models/deepseek_v2.py`, and the MTP draft-model entry is `DeepseekV3ForCausalLMNextN` in `python/sglang/srt/models/deepseek_nextn.py`. Main now has a full surface for MLA backend selection, FP8/FP4/W4AFP8/MXFP4/MXFP8/NVFP4 loading, shared-expert fusion, MTP, R1 W4A8 DeepEP, DP attention, LoRA, and multi-hardware validation. Additional runtime items include adaptive EAGLE, PCG plus speculative decoding, thinking-token radix-cache stripping, and spec-v2 adaptive speculative decoding. The main open items are JIT router GEMM, quantized DeepSeek MLA `.weight` access, ROCm MLA restoration, LoRA adapter bypass, CuteDSL EP plus DP-attention double reduce, MUSA, DCP, and spec-v2 adaptive speculative decoding.

## 1. Timeline Overview

| Created | PR | State | Track | Code Area | Effect |
| --- | ---: | --- | --- | --- | --- |
| 2024-12-26 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | AMD bring-up | Triton decode attention, fused MoE, `deepseek_v2.py` | Made DeepSeek V3 runnable on AMD paths. |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD FP8 | `deepseek_v2.py` | Fixed DeepSeek V3 FP8 accuracy on AMD. |
| 2025-02-05 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | docs | DeepSeek docs | Added DeepSeek usage and multi-node launch docs. |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | docs | DeepSeek V3 launch docs | Refined DeepSeek V3 launch parameters in docs. |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | MTP | `deepseek_nextn.py`, speculative decoding | Added NextN/EAGLE speculative decoding for DeepSeek V3/R1. |
| 2025-02-26 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | FP8 GEMM | benchmarks, DeepGEMM | Added DeepGEMM and SGLang FP8 block-wise GEMM benchmarks. |
| 2025-03-05 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | INT8 docs | DeepSeek docs | Added an INT8 launch example. |
| 2025-03-07 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGEMM | `sgl-kernel` | Integrated DeepGEMM into `sgl-kernel`. |
| 2025-03-08 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | DeepGEMM | Linear layers | Made Linear layers support DeepGEMM. |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | MTP/MLA | FlashInfer MLA | Added NextN support for the FlashInfer MLA backend. |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | FlashMLA | attention backend | Added the initial FlashMLA backend. |
| 2025-03-17 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | FlashMLA graph | `flashmla_backend.py`, server args | Added CUDA graph support for the FlashMLA backend. |
| 2025-03-18 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | fused MoE | `moe_fused_gate.cu`, tests, benchmarks | Added the DeepSeek-style fused group gate selection kernel. |
| 2025-03-20 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | DeepGEMM default | server defaults | Enabled DeepGEMM by default on Hopper. |
| 2025-03-20 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | ROCm MTP | NextN | Enabled MTP/NextN on AMD GPUs. |
| 2025-03-27 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | FA3 MLA | attention backend | Added FA3 backend support for MLA. |
| 2025-04-05 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | MoE align | `moe_align_kernel.cu`, fused MoE | Reduced `moe_align_block_size_kernel` small-batch overhead. |
| 2025-04-07 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | MHA chunked prefill | `flashattention_backend.py`, scheduler, `deepseek_v2.py` | Added `MHA_CHUNKED_KV` for DeepSeek chunked prefill. |
| 2025-04-09 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | FA3 default | server defaults | Used FA3 MLA by default on Hopper. |
| 2025-04-11 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | DeepGEMM guard | defaults | Temporarily turned off DeepGEMM by default. |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | DeepGEMM guard | defaults | Limited DeepGEMM usage to Hopper. |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | fused MoE | `deepseek_v2.py`, MoE gate | Applied the fused MoE gate in DeepSeek V3/R1. |
| 2025-04-14 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | MLA kernel | `merge_attn_states.cu` | Added the faster `merge_state_v2` CUDA merge-attention-state kernel. |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | RoPE | `rotary_embedding.py` | Applied DeepSeek CUDA RoPE. |
| 2025-04-14 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Cutlass MLA | `cutlass_mla_backend.py`, sgl-kernel attention | Added the Cutlass MLA attention backend. |
| 2025-04-15 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | DeepGEMM BMM | `fp8_kernel.py`, `deepseek_v2.py` | Introduced DeepGEMM `group_gemm_masked` as an MLA BMM exploration path. |
| 2025-04-16 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | FP8 quant | `fp8_kernel.py`, `fp8_utils.py` | Replaced the Triton kernel with `sglang_per_token_group_quant_fp8` from `sgl-kernel`. |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | MLA FP8 quant | `fp8_kernel.py`, `deepseek_v2.py` | Reused a zero-scalar allocator and removed one `per_tensor_quant_mla_fp8` kernel. |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | shared experts | SM90 shared experts | Enabled DeepSeek V3 shared-expert fusion on SM90. |
| 2025-04-20 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | MLA copy | `deepseek_v2.py`, RoPE | Removed an extra copy in DeepSeek `forward_absorb`. |
| 2025-04-22 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | MLA projection | `deepseek_v2.py`, loader | Fused `q_a_proj` and `kv_a_proj_with_mqa`. |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | DeepGEMM default | defaults, docs | Turned DeepGEMM back on by default and updated docs. |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | MTP/fusion | R1 MTP, shared experts | Fixed the R1 combination of MTP and shared-expert fusion. |
| 2025-04-24 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | MoE tuning | Triton fused-MoE config | Updated H20 DeepSeek/R1 FP8 W8A8 fused-MoE Triton configs. |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | MoE tuning | H200 Triton fused-MoE config | Updated H200 Triton 3.2 fused-MoE configs and warning behavior. |
| 2025-04-25 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | MLA KV cache | `flashattention_backend.py`, `memory_pool.py`, `deepseek_v2.py` | Fused MLA set-KV-cache and removed K concat overhead. |
| 2025-04-27 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | MTP ergonomics | server/spec args | Auto-set the MTP draft model path. |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | MTP API | CI, docs | Updated tests and docs for the MTP API change. |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | MLA streams | `deepseek_v2.py` | Overlapped q/k norm with two streams. |
| 2025-05-05 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | docs | MLA backend docs | Updated MLA attention backend documentation. |
| 2025-05-07 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | MTP/DP attention | MTP, DP attention | Added MTP support with DP attention. |
| 2025-05-08 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | FlashMLA/MTP | FlashMLA, FP8 KV | Added FlashMLA backend support with MTP and FP8 KV cache. |
| 2025-05-09 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | hybrid attention | model_runner, cuda graph, server args | Explored a hybrid attention backend; it did not become the V3/R1 main path. |
| 2025-05-12 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | fused MoE | top-k reduce, quant methods | Fused routed scaling factor into the top-k reduce kernel. |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | DeepGEMM/MLA | `fused_qkv_a_proj_with_mqa` | Replaced the Triton path with DeepGEMM for this fused projection. |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | routed scaling | DeepSeek MoE | Fused the routed scaling factor in DeepSeek. |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | DeepGEMM format | per-token-group quant | Supported the new DeepGEMM format in per-token-group quantization. |
| 2025-06-13 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | DeepGEMM refactor | DeepGEMM integration | Refactored DeepGEMM integration. |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | DeepGEMM format | SRT quant | Added SRT-side support for the new DeepGEMM quant format. |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | DeepGEMM format | DeepSeek weights | Re-quantized DeepSeek weights for the new DeepGEMM input format. |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | DeepGEMM | new DeepGEMM path | Completed support for the new DeepGEMM path. |
| 2025-06-20 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | MTP/FP4 | `deepseek_nextn.py`, speculative decoding | Fixed MTP with DeepSeek R1 FP4. |
| 2025-07-04 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | R1 W4AFP8 | `w4afp8.py`, `cutlass_w4a8_moe.py`, EP MoE | Added R1 W4AFP8 config, Cutlass W4A8 MoE, and EP-MoE paths. |
| 2025-07-17 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | R1 W4AFP8 TP | Cutlass grouped W4A8 MoE | Added TP mode for R1-W4AFP8. |
| 2025-07-22 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | R1 W4A8 DeepEP | `token_dispatcher/deepep.py`, W4A8 MoE | Added normal DeepEP for R1 W4A8/W4AFP8. |
| 2025-07-28 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | R1 W4A8 DeepEP LL | DeepEP low latency | Added low-latency DeepEP for R1 W4A8. |
| 2025-09-04 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | W4AFP8 perf | glue kernels | Optimized R1 W4AFP8 glue kernels. |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | DP/compile | DP plus torch compile | Fixed GPU fault with DeepSeek V3 DP plus torch-compile. |
| 2025-10-12 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | FP4 default | server defaults | Updated R1-FP4 default config on Blackwell. |
| 2025-10-16 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | FP4/SM120 | backend defaults | Enabled FP4 DeepSeek on SM120. |
| 2025-10-23 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | deterministic | DeepSeek attention | Added deterministic inference for single-GPU DeepSeek-architecture models. |
| 2025-10-24 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | docs | W4FP8 docs | Added a W4FP8 usage example. |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | Blackwell default | `server_args.py` | Updated DeepSeek V3 auto quantization on SM100. |
| 2025-11-09 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | W4AFP8 perf | W4A8 kernels | Optimized W4AFP8 kernels for DeepSeek-V3-0324. |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | MTP/B200 | NextN, speculative decoding | Fixed DeepSeek V3 MTP on B200. |
| 2025-11-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepEP LL | R1 W4A8 DeepEP | Made R1 W4A8 DeepEP low-latency dispatch use FP8 communication. |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | DP accuracy | BF16 KV | Fixed DeepSeek V3 DP accuracy with BF16 KV. |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | MXFP4 | AMD EP | Fixed MXFP4 DeepSeek V3 with EP accuracy. |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | router/top-k | `topk.py` | Replaced the generic `moe_fused_gate` hot path with `fused_topk_deepseek`. |
| 2025-12-20 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | PCG/FP4 | CUDA graph | Added piecewise CUDA graph support for DeepSeek V3 FP4. |
| 2026-01-07 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | loader refactor | `deepseek_common/deepseek_weight_loader.py` | Split DeepSeek V2/V3 weight loading into a mixin. |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | fused MoE configs | Added H20/H20-3E MoE configs for DeepSeek-family shapes. |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | eval/parser | eval choices | Removed `deepseek-r1` from thinking-mode choices because R1 parser behavior differs from V3-style thinking. |
| 2026-01-25 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | router bench | `dsv3_router_gemm` | Added a Blackwell router GEMM benchmark. |
| 2026-01-26 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | loader memory | weight loader | Deferred `dict(weights)` materialization to avoid large-checkpoint OOM. |
| 2026-02-03 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | ROCm perf | MI300X | Optimized DeepSeek R1 on MI300X. |
| 2026-02-08 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | AMD router | AITER router GEMM | Uses `aiter_dsv3_router_gemm` when the expert count is at most 256. |
| 2026-02-09 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | XPU | Intel GPU | Enabled R1 inference on Intel GPU. |
| 2026-02-11 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | AMD MTP | TP4 MTP | Fixed DeepSeek V3 MTP accuracy on AMD TP4. |
| 2026-02-22 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | `deepseek_common/attention_forward_methods/` | Moved DeepSeek MLA forward code into shared forward-method modules. |
| 2026-02-26 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | R1 MXFP4 | NextN loading | Fixed R1-0528-MXFP4 weight-loading shape mismatch. |
| 2026-03-04 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | AMD CI | MI35x lanes | Added DeepSeek KV FP8 and all-reduce fusion tests on MI35x. |
| 2026-03-04 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | AMD perf | AITER FP8 top-k | Kept correction bias in BF16 for AITER FP8 routing to avoid runtime dtype conversion. |
| 2026-03-18 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | DP bugfix | DeepSeek R1 DP | Fixed a GPU fault when DeepSeek R1 runs with DP. |
| 2026-03-24 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | MXFP8 | routed MoE | Added MXFP8 DeepSeek V3 routed-MoE support. |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | server args, EAGLE runtime, spec workers | Added adaptive `speculative_num_steps` for EAGLE top-k=1. |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | revert | DeepEP LL | Reverted `#14162`. |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | `model_runner.py`, PCG runner, server args | Allowed piecewise CUDA graph to run with speculative decoding. |
| 2026-04-08 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | reland | DeepEP LL | Relanded R1 W4A8 DeepEP low-latency FP8 communication. |
| 2026-04-08 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | LoRA | quant info, MLA LoRA | Refactored LoRA quant info and added DeepSeek V3 MLA LoRA support. |
| 2026-04-16 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | CPU shared expert | CPU MoE | Expanded the CPU shared-expert interface without scaling factor; this is CPU parity, not H200 throughput. |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | model config, scheduler, radix cache, reasoning parser | Explored parser-gated two-phase reasoning radix-cache stripping; it did not become current main. |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | quant bugfix | `DeepseekV2AttentionMLA` | Guards `.weight` access for AWQ/compressed-tensors layers. |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | MoE/DP | CuteDSL EP plus DP attention | Fixes double reduce in `DeepseekV2MoE` with CuteDSL EP plus DP attention. |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | `schedule_batch.py`, `mem_cache/common.py`, `server_args.py` | Added opt-in thinking-token stripping from radix cache. |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | scheduler output processor, EAGLE v2 workers | Extends adaptive speculative decoding to spec v2. |

## 2. Single-Node H200 Optimization Coverage

The single-node H200 optimization notes explicitly name `#4514`, `#4530`, `#5086`, `#5113`, `#5381`, `#5385`, `#5390`, `#5432`, `#5473`, `#5549`, `#5578`, `#5619`, `#5716`, `#5740`, `#5748`, `#5977`, `#6034`, `#6151`, and `#6220`. These PRs are included in the timeline and are marked according to their current-main status: default path, optional backend, exploratory path, or closed direction.

These PRs fall into four main tracks.

The first track is FP8 Block GEMM / DeepGEMM. `#3893` put DeepGEMM and SGLang FP8 block-wise GEMM on the same benchmark surface; `#4165` integrated DeepGEMM into `sgl-kernel`; and `#4199` made Linear layers support DeepGEMM. The sequence `#4613`, `#5263`, `#5310`, and `#5628` shows that the default was iterated carefully: enable on Hopper, temporarily disable when needed, restrict to the safe architecture set, then re-enable with documentation. `#5432`'s DeepGEMM `group_gemm_masked` BMM and MLA FP8 quant kernel is an exploratory path and should not be described as the current H200 default. `#5473` moved per-token-group FP8 quantization from Triton to `sgl-kernel`, while `#5549` reused a zero-scalar allocator and removed one kernel from `per_tensor_quant_mla_fp8`. Later, `#6890` and `#7146`, `#7150`, `#7155`, `#7156`, `#7172` moved the fused projection and DeepSeek weight quantization to the new DeepGEMM input format.

The second track is Fused MoE. `#4530` added `moe_fused_gate.cu`, bindings, benchmarks, and tests for DeepSeek biased grouped top-k / group gate selection; `#5086` reduced `moe_align_block_size_kernel` small-batch overhead; `#5371` connected the fused MoE gate to DeepSeek V3/R1; `#5571` enabled shared-expert fusion on SM90; `#5716` and `#5740` added H20/H200 fused-MoE Triton configs; `#6220` fused routed scaling factor into the top-k reduce kernel, and `#6970` later fused the same scaling directly in the DeepSeek path. When reading current main, check `topk.py`, `fused_moe_triton/fused_moe.py`, `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `moe_align_kernel.cu`, and `sgl_kernel_ops.h` together.

The third track is MLA / attention backend work. FlashMLA starts with `#4472`, gains CUDA graph support in `#4514`, and later gets MTP plus FP8 KV cache support in `#6109`. FA3 MLA starts with `#4831`, and `#5210` makes FA3 MLA the Hopper default. Cutlass MLA is `#5390`, while `#6034` documents the boundaries between FA3, FlashMLA, Cutlass MLA, and other backends. On the model-file hot path, `#5113` adds `MHA_CHUNKED_KV`, `#5381` adds the `merge_state_v2` CUDA kernel, `#5385` applies DeepSeek CUDA RoPE, `#5578` removes a `forward_absorb` copy, `#5619` fuses `q_a_proj` and `kv_a_proj_with_mqa`, `#5748` fuses MLA set-KV-cache, and `#5977` overlaps q/k norm with two streams.

The fourth track is MTP and backend interaction. `#3582` is the V3/R1 NextN/EAGLE starting point, `#4218` supports FlashInfer MLA plus NextN, `#5707` fixes R1 MTP plus shared-expert fusion, `#5793` auto-sets the draft model path, `#5952` updates tests and docs for the MTP API, `#6081` supports MTP plus DP attention, and `#6109` connects FlashMLA, MTP, and FP8 KV cache. `#6151` is a closed hybrid-attention backend exploration, so it should be recorded as history but not counted as current mainline support.

## 2.1 Update: MTP/PCG and Thinking Radix Cache

After refreshing SGLang and sgl-cookbook, SGLang main is still `929e00eea`, but sgl-cookbook moved to `8ec4d03`; there are no DeepSeek cookbook doc or model-entry changes from the previous cookbook snapshot to `8ec4d03`. The real additions in this pass are therefore SGLang runtime PRs.

`#21599` makes EAGLE top-k=1 `speculative_num_steps` adaptive, touching `server_args.py`, speculative runtime state/params, EAGLE workers, and runner code. It affects V3/R1 MTP performance tuning because the draft-step count should no longer be assumed static. `#22128` allows piecewise CUDA graph to coexist with speculative decoding through `model_runner.py`, `piecewise_cuda_graph_runner.py`, and the server-flag gate; PCG plus MTP should no longer be written off as categorically unsupported.

`#22950` is the closed early design for reasoning radix-cache stripping, spanning model config, scheduler, radix cache, and `reasoning_parser.py`; current main should be read from merged `#23315`. `#23315` adds an opt-in server argument and changes `schedule_batch.py` / `mem_cache/common.py` so thinking tokens can be stripped from radix-cache entries, preventing `<think>` / `</think>`-style reasoning tokens from becoming reusable prefix material. Open `#23336` carries adaptive spec into spec v2 via `scheduler_output_processor_mixin.py`, `managers/utils.py`, `eagle_worker_v2.py`, and `multi_layer_eagle_worker_v2.py`.

## 3. Current Main Code Shape

DeepSeek V3/R1 mainline support is not a brand-new model file. It reuses `deepseek_v2.py`, which evolved from the DeepSeek V2 implementation. `DeepseekV3ForCausalLM` inherits from `DeepseekV2ForCausalLM`, and the core layers include `DeepseekV2AttentionMLA`, `DeepseekV2MoE`, `DeepseekV2DecoderLayer`, and `DeepseekV2Model`. This is why many fixes named `deepseek_v2` also affect V3, R1, V3.1, and even V3.2.

The most important shared modules in current main are:

- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`: stacked qkv/gate_up loading, expert parameter mapping, `kv_b_proj` post-processing, W4AFP8 scale mapping, and DeepGEMM BMM weight transforms.
- `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`: forward-method selection by backend, deterministic mode, PCG, and MHA/MLA subtype.
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/`: MLA/MHA forward logic after `#19122`.
- `python/sglang/srt/models/deepseek_nextn.py`: MTP/NextN draft layer.
- `python/sglang/srt/parser/reasoning_parser.py`: separation between `deepseek-r1` and `deepseek-v3` reasoning parsers.
- `python/sglang/srt/function_call/deepseekv3_detector.py`: V3/R1 tool-call parser.
- `python/sglang/srt/managers/schedule_batch.py` and `python/sglang/srt/mem_cache/common.py`: current-main path for thinking-token radix-cache stripping.
- `python/sglang/srt/server_args.py`: the main entry point for DeepSeek-family default attention backend, KV-cache dtype, quantized backend, and DeepEP/DP-attention guards.

`server_args.py` now applies several DeepSeek V3/R1 automatic choices. On Blackwell SM100, if no MLA backend is specified, it defaults to `trtllm_mla`. Official FP8 and ModelOpt FP8/FP4 quantized checkpoints tend to select `flashinfer_trtllm` MoE runner when conditions allow. With piecewise CUDA graph enabled, V3/R1 records the “use MLA for prefill” path. ROCm has separate AITER all-reduce fusion and FP4/EAGLE backend defaults. Performance debugging must therefore include launch args, server-side automatic rewrites, and the model file.

## 4. MLA and Weight Loading: From Runnable to Backend-Aware

DeepSeek V3/R1 attention is primarily MLA. `DeepseekV2AttentionMLA` builds q/k/v latent projections from `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, and `v_head_dim`. In current main, `q_a_proj` and `kv_a_proj_with_mqa` can be fused into `fused_qkv_a_proj_with_mqa`, while `kv_b_proj` is post-processed after loading to expose components such as `w_kc` and `w_vc` that backends need.

After `#16649`, weight loading lives in `DeepseekV2WeightLoaderMixin`, which all later DeepSeek-family models reuse. Key details include:

- `gate_proj/up_proj` stack into `gate_up_proj`, while q/k/v names follow MLA-specific mapping.
- Expert parameters go through `make_expert_params_mapping` and W4AFP8 input-scale mapping.
- When shared-expert fusion is enabled, `mlp.shared_experts` can map to `mlp.experts.256`.
- `kv_b_proj` has post-load handling for AWQ, FP8 block scale, and DeepGEMM BMM.
- R1 MXFP4 / NextN checkpoints can use `model.layers.61*` naming and require special handling in `deepseek_nextn.py`.

`#17744` is a practical memory fix: it avoids eagerly materializing `dict(weights)` while loading large checkpoints. `#23195` remains open and warns that quantized layers may not expose `.weight`; if MLA initialization fails on AWQ or compressed-tensors checkpoints, inspect that direction before assuming missing weights.

After `#19122`, MLA forward code moved into `attention_forward_methods`. That makes backend switching cleaner, but it also creates compatibility-regression risk. Open `#22938` is still restoring MI300X DeepSeek MLA paths, and open `#21530` is still fixing ROCm fused MLA decode RoPE.

## 5. MoE Routing, Shared Experts, and Communication Boundaries

DeepSeek V3/R1 MoE has 256 routed experts plus shared experts. Current main's `DeepseekV2MoE` computes `num_fused_shared_experts` from config and server args. When fusion is enabled, the loader remaps `mlp.shared_experts` to `mlp.experts.256`, placing routed and shared experts on one fused-MoE compute surface.

Fusion is deliberately conservative:

- TBO/SBO disables fusion.
- DeepEP disables fusion by default unless `--enforce-shared-experts-fusion` is set.
- W4AFP8 disables fusion because routed and shared experts may use different quantization methods.
- Architecture, expert-count, or backend-capability mismatches also disable it.

Shared-expert fusion under DeepEP is more complex. Ordinary fusion is `256 + 1`; DeepEP fusion expands the local expert layout to `256 + EP_size`, and TopK must handle shared expert interleaving and mapping across EP ranks. Bugs here often show up as output correctness or double reduction, not as a slow single kernel. `#23257` is still open and targets exactly the overlap between MoE internal all-reduce and outer DP-attention reduce for CuteDSL EP plus DP attention.

The main routing change is `#15347`. DeepSeek biased grouped top-k no longer prefers the generic `moe_fused_gate`; when constraints match, it uses `fused_topk_deepseek`. `#17707` added a Blackwell router benchmark, and `#22933` expanded the CPU shared-expert interface when scaling factor is absent, which is CPU parity cleanup rather than H200 GPU throughput. Open `#21531` migrates `dsv3_router_gemm` from AOT sgl-kernel to JIT, which is important for future router maintainability and deployment.

## 6. MTP / NextN: The Draft Layer Is Its Own Runtime Surface

DeepSeek V3/R1 MTP uses EAGLE and `DeepseekV3ForCausalLMNextN`. It has a separate NextN layer, shared embed/head reuse, its own loading logic, and sometimes different quantization.

Current `deepseek_nextn.py` has these important constraints:

- Only one NextN layer is supported.
- The target model is `DeepseekV3ForCausalLM`, and the draft model is `DeepseekV3ForCausalLMNextN`.
- The draft layer may be BF16 or may use quantization handling that differs from the target.
- AMD R1 MXFP4 needs special naming and shape fixes.
- Some DeepEP BF16 dispatch environment variables may need to be toggled around NextN execution.

`#7376` fixed R1 FP4 MTP, `#13548` fixed V3 MTP on B200, `#18607` fixed V3 MTP accuracy on AMD TP4, and `#19425` fixed R1-0528-MXFP4 draft loading shape. In current validation, the H200 V3 MTP registered test expects GSM8K above `0.935`, average spec accept length above `2.8`, and batch-size-1 throughput above the non-MTP lane.

The newer spec line adds two constraints that should be recorded explicitly: `#21599` makes EAGLE top-k=1 draft steps adaptive, and `#22128` lets PCG coexist with speculative decoding. Open `#23336` continues this into the spec-v2 worker path. When writing a skill or debugging performance, record the target model, draft model, spec v1/v2, PCG, and DP attention together.

## 7. R1 W4AFP8 / W4A8 DeepEP: A Separate Quantized Optimization Ladder

R1 W4AFP8 should not be treated as ordinary FP8. `W4AFp8Config` from `#7762` detects mixed precision from the quant config, maps ordinary Linear layers to FP8 or unquantized methods, and maps MoE experts to W4A8. `cutlass_w4a8_moe.py` handles packed int4 expert weights, FP8 activations, input scales, and grouped MoE runners.

Several later PRs make the path complete:

- `#8118` adds TP mode for R1-W4AFP8.
- `#8247` adds normal DeepEP by letting DeepEP dispatch metadata enter W4A8 MoE `apply_deepep_normal`.
- `#8464` adds low-latency DeepEP.
- `#10027` and `#12921` optimize W4AFP8 glue kernels and DeepSeek-V3-0324 W4AFP8 performance.

The low-latency DeepEP FP8 communication track must be read with its revert history: `#14162` landed R1 W4A8 DeepEP LL FP8 communication, `#21719` reverted it, and `#22316` relanded it. Reading only `#14162` gives the wrong conclusion; the real current-main state is the post-`#22316` code.

## 8. Quantization, Platforms, and Parser Support

The current DeepSeek V3/R1 quantization surface is broad:

- Official V3 FP8: no need to manually pass `--quantization fp8`; the server can recognize the quantization config.
- FP4/NVFP4: `#11512`, `#11708`, and `#12778` make Blackwell/SM120 defaults and backend selection safer.
- W4AFP8/W4A8: centered on `w4afp8.py`, `cutlass_w4a8_moe.py`, and DeepEP normal/LL paths.
- MXFP4: `#15304` fixes AMD EP accuracy, `#19425` fixes R1-0528-MXFP4 draft loading, and open `#21529` continues ROCm Quark W4A4 work.
- MXFP8: `#21280` adds MXFP8 support for routed MoE.
- LoRA: `#22323` refactors LoRA quant info and supports DeepSeek V3 MLA LoRA; open `#22268` points to adapter bypass in `prepare_qkv_latent`.

Parser choices must also be separated. V3/R1 tool calling uses `--tool-call-parser deepseekv3`, V3-style thinking uses `--reasoning-parser deepseek-v3`, and R1 uses `--reasoning-parser deepseek-r1`. The R1 parser handles output without an opening `<think>` tag by forcing content into reasoning until `</think>`; this differs from the Qwen3-style parser used by V3/V3.1.

Radix cache should be checked separately from the thinking parser. `#23315`'s opt-in strip is cache-layer behavior: it decides whether thinking tokens can be reused as prefix-cache content; it is not a parser-format change in `deepseekv3_detector.py` or `reasoning_parser.py`. For multi-turn reasoning anomalies, record the parser, strip flag, and cache-hit state together.

## 9. Current Validation Surface and Open PRs

Current main has validation lanes for:

- `test/registered/8-gpu-models/test_deepseek_v3_basic.py`: H200 V3 base accuracy and performance, with GSM8K above `0.935`.
- `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`: H200 V3 MTP, average spec accept length, and throughput.
- `test/registered/amd/test_deepseek_v3_basic.py`, `test/registered/amd/test_deepseek_v3_mtp.py`, `test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`: AMD base, MTP, and R1 MXFP4.
- `test/registered/backends/test_deepseek_r1_fp8_trtllm_backend.py`: R1 FP8 TRTLLM backend.
- `test/registered/quant/test_deepseek_v3_fp4_4gpu.py`, `test/registered/quant/test_w4a8_deepseek_v3.py`: FP4 and W4A8.
- `test/registered/mla/test_mla_deepseek_v3.py`, `test/registered/mla/test_mla_int8_deepseek_v3.py`: MLA and INT8 MLA.
- `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`: LoRA logprob regression.
- `test/registered/kernels/test_fused_topk_deepseek.py`: DeepSeek fused top-k.

Open PRs to track:

- `#14194`: DCP for DeepSeek V2/V3.
- `#15315`, `#15380`: group GEMM for DeepSeek-R1-W4AFP8.
- `#18892`: DeepSeek V3 GEMM JIT.
- `#21526`: ROCm AITER router GEMM regression.
- `#21529`: ROCm DeepSeek-architecture MXFP4/Quark W4A4.
- `#21530`: ROCm fused MLA decode RoPE.
- `#21531`: `dsv3_router_gemm` JIT migration.
- `#22268`: DeepSeek MLA LoRA adapter bypass.
- `#22774`: MUSA backend.
- `#22938`: MI300X DeepSeek path restoration after the MLA refactor.
- `#23195`: quantized-layer `.weight` access guard.
- `#23257`: CuteDSL EP plus DP-attention double reduce.
- `#23336`: spec-v2 adaptive speculative decoding.
