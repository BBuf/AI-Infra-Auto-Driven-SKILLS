# Open PR Watch

Generated: `2026-07-28`.

This report is a triage aid for skill updates. Read the linked PR diffs
before changing benchmark, profiler, or model-history guidance.

## NVIDIA/TensorRT-LLM

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#16095](https://github.com/NVIDIA/TensorRT-LLM/pull/16095) | 2026-07-27 | `FP4` | [TRTLLM-14135][feat] Add Qwen-Image-Edit-2511 support |
| [#16514](https://github.com/NVIDIA/TensorRT-LLM/pull/16514) | 2026-07-27 | `MoE` | [https://nvbugs/6463829][fix] Fix fp8 MoE test |
| [#16558](https://github.com/NVIDIA/TensorRT-LLM/pull/16558) | 2026-07-27 | `DeepSeek V4`, `FP4`, `GLM-5`, `GLM-5.2`, `MLA`, `MoE`, `NVFP4` | [None][perf] Allocate DSA indexer k-cache only for layers that own an indexer |
| [#16561](https://github.com/NVIDIA/TensorRT-LLM/pull/16561) | 2026-07-27 | `FP4`, `NVFP4`, `Qwen3.6` | [None][feat] MTP one-model `advanced_sampling_mode`: skip redundant top-k / top-p filter kernels with additional config enum |
| [#16568](https://github.com/NVIDIA/TensorRT-LLM/pull/16568) | 2026-07-27 | `MoE`, `Qwen3.5` | [https://nvbugs/6442073][fix] Enforce a fixed max_seq_len for Qwen's AttentionOp caching purposes |
| [#16603](https://github.com/NVIDIA/TensorRT-LLM/pull/16603) | 2026-07-27 | `MLA`, `MoE` | [https://nvbugs/6379316][fix] Reject MNNVL on split NVLink topology |
| [#16629](https://github.com/NVIDIA/TensorRT-LLM/pull/16629) | 2026-07-27 | `MLA` | [https://nvbugs/6478707][fix] detect v2 inside remove_functionalize_inner, resolve each mutates_args via… |
| [#16683](https://github.com/NVIDIA/TensorRT-LLM/pull/16683) | 2026-07-27 | `FP4`, `NVFP4` | [TRTLLM-14557][test] Add single-device visual-gen feature accuracy regression tests |
| [#16759](https://github.com/NVIDIA/TensorRT-LLM/pull/16759) | 2026-07-27 | `DFlash`, `GDN` | [None][fix] SA spec dec: promote accepted hybrid recurrent states in-worker |
| [#16783](https://github.com/NVIDIA/TensorRT-LLM/pull/16783) | 2026-07-27 | `DeepSeek V4`, `FP4`, `NVFP4` | [None][perf] Preserve default V2 KV cache pool sizing |
| [#16788](https://github.com/NVIDIA/TensorRT-LLM/pull/16788) | 2026-07-27 | `FP4`, `NVFP4` | [https://nvbugs/6490028][fix] Bump only the cross-library `cublas_tolerance` from 1.05 to 1.10 in the test… |
| [#16810](https://github.com/NVIDIA/TensorRT-LLM/pull/16810) | 2026-07-27 | `MoE` | [None][feat] Support dense FP8 LoRA end to end |
| [#16852](https://github.com/NVIDIA/TensorRT-LLM/pull/16852) | 2026-07-27 | `MiniMax M3` | [None][perf] Fuse MiniMax-M3 prefill projections |
| [#16856](https://github.com/NVIDIA/TensorRT-LLM/pull/16856) | 2026-07-27 | `MiniMax M3` | [None][perf] Avoid Index-K cache materialization for MSA |
| [#16862](https://github.com/NVIDIA/TensorRT-LLM/pull/16862) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in TRTLLMGenFusedMoE |
| [#16864](https://github.com/NVIDIA/TensorRT-LLM/pull/16864) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in DeepGemmFusedMoE |
| [#16887](https://github.com/NVIDIA/TensorRT-LLM/pull/16887) | 2026-07-27 | `DeepSeek V4` | [https://nvbugs/6463967][fix] DeepSeek-V4 one-model MTP separate draft kv cache (TEP) |
| [#16905](https://github.com/NVIDIA/TensorRT-LLM/pull/16905) | 2026-07-27 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] Fused GEMM + SwiGLU-OAI |
| [#16906](https://github.com/NVIDIA/TensorRT-LLM/pull/16906) | 2026-07-27 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] Fuse QK-norm + RoPE and overlap qkv/idx_qk math |
| [#16908](https://github.com/NVIDIA/TensorRT-LLM/pull/16908) | 2026-07-27 | `MoE` | [[TRTLLM-13948][feat] Set DeepSeekV3 to use Python KV-cache transceiver V2 by default |

## lightseekorg/tokenspeed

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#555](https://github.com/lightseekorg/tokenspeed/pull/555) | 2026-07-14 | `FP4`, `Kimi K2.5`, `MLA`, `MoE`, `NVFP4` |  perf(scheduler): Split mixed prefill/decode forwards to preserve decode graphs |
| [#556](https://github.com/lightseekorg/tokenspeed/pull/556) | 2026-07-14 | `FP4`, `Kimi K2.5`, `MLA`, `MoE`, `NVFP4` | perf(multimodal): Prewarm Kimi vision encoder before readiness |
| [#563](https://github.com/lightseekorg/tokenspeed/pull/563) | 2026-07-25 | `DFlash`, `DeepSeek V4`, `FP4`, `MLA` | perf(deepseek-v4): accelerate indexer Q and packed FP8 quantization |
| [#580](https://github.com/lightseekorg/tokenspeed/pull/580) | 2026-07-21 | `DeepSeek V4`, `MoE` | feat(distributed): wire TensorRT-LLM all-reduce backend |
| [#608](https://github.com/lightseekorg/tokenspeed/pull/608) | 2026-07-15 | `Qwen3.5` | perf(multimodal): optimize Qwen3.5 M-RoPE preparation |
| [#616](https://github.com/lightseekorg/tokenspeed/pull/616) | 2026-07-25 | `MLA` | fix(trtllm-mla): make spec-decode CUDA graph capture causal |
| [#617](https://github.com/lightseekorg/tokenspeed/pull/617) | 2026-07-25 | `MLA` | feat(mla): support custom tree masks in decode |
| [#620](https://github.com/lightseekorg/tokenspeed/pull/620) | 2026-07-27 | `DeepSeek V4`, `FP4`, `MoE` | feat: add DeepSeek V4 L2 KV cache offload and perf optimize. |
| [#621](https://github.com/lightseekorg/tokenspeed/pull/621) | 2026-07-25 | `DeepSeek V4` | perf(deepseek-v4): widen sparse compress cache launch for large ratios |
| [#645](https://github.com/lightseekorg/tokenspeed/pull/645) | 2026-07-27 | `FP4`, `MoE` | feat(kernel): add SM120 FlashInfer MXFP4 MoE |
| [#648](https://github.com/lightseekorg/tokenspeed/pull/648) | 2026-07-13 | `DeepSeek V4`, `FP4`, `MLA`, `MoE` | feat(deepseek-v4): enable SM120 serving |
| [#651](https://github.com/lightseekorg/tokenspeed/pull/651) | 2026-07-27 | `FP4`, `Kimi K2.5` | feat(runtime): scheduler-driven full_refresh bit for page-table mirror |
| [#660](https://github.com/lightseekorg/tokenspeed/pull/660) | 2026-07-13 | `DeepSeek V4` | fix(pd): correct DeepSeek V4 layerwise cache handoff |
| [#694](https://github.com/lightseekorg/tokenspeed/pull/694) | 2026-07-22 | `Inkling` | feat(scheduler): live-tail allocation for sliding groups |
| [#714](https://github.com/lightseekorg/tokenspeed/pull/714) | 2026-07-18 | `DeepSeek V4`, `FP4` | perf(deepseek-v4): capture DSA projections and cache writes into the prefill graph |
| [#720](https://github.com/lightseekorg/tokenspeed/pull/720) | 2026-07-18 | `FP4`, `MoE`, `NVFP4`, `Qwen3.5` | [WIP][AMD] Implement MTP support for qwen3.5 MXFP4 |
| [#721](https://github.com/lightseekorg/tokenspeed/pull/721) | 2026-07-18 | `FP4`, `GDN`, `Qwen3.5` | [wip] refactor(kernel): migrate GDN Triton kernels to tensor descriptors |
| [#737](https://github.com/lightseekorg/tokenspeed/pull/737) | 2026-07-20 | `MoE` | feat(moe): R3 rollout routing replay — slot-indexed expert-routing pool (scaffold) |
| [#738](https://github.com/lightseekorg/tokenspeed/pull/738) | 2026-07-20 | `FP4`, `MoE`, `Qwen3.5` | feat(lora): LoRA adapter serving |
| [#745](https://github.com/lightseekorg/tokenspeed/pull/745) | 2026-07-23 | `DeepSeek V4` | refactor(runtime): make ForwardContext tensor-free (restore "Do not contain Tensor")' |
| [#750](https://github.com/lightseekorg/tokenspeed/pull/750) | 2026-07-27 | `DeepSeek V4` | [WIP] feat(deepseek): land V4 L1 KV cache on heterogeneous flat kv pools |
| [#758](https://github.com/lightseekorg/tokenspeed/pull/758) | 2026-07-22 | `MiniMax M3` | Support MiniMax M3 CPU KVStore |
| [#782](https://github.com/lightseekorg/tokenspeed/pull/782) | 2026-07-27 | `GLM-5`, `GLM-5.2` | perf(kernel): optimize gluon gfx950 DSA top-k selection kernel |
| [#793](https://github.com/lightseekorg/tokenspeed/pull/793) | 2026-07-24 | `MLA`, `MoE` | feat(kernel): support small-batch Gluon MLA decode |
| [#819](https://github.com/lightseekorg/tokenspeed/pull/819) | 2026-07-27 | `FP4`, `MiniMax M3`, `MoE`, `NVFP4` | perf(m3): avoid per-layer D2H sync in MSA prefill |
| [#822](https://github.com/lightseekorg/tokenspeed/pull/822) | 2026-07-27 | `Kimi K3` | feat(kimi-k3): integrate Kimi K3 support |

## sgl-project/sglang

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#18589](https://github.com/sgl-project/sglang/pull/18589) | 2026-07-27 | `MoE` | Make per‑token expert‑distribution recorder use asynchronous data syncing |
| [#20907](https://github.com/sgl-project/sglang/pull/20907) | 2026-07-27 | `MoE` | Expose Model Parallelism Information |
| [#27770](https://github.com/sgl-project/sglang/pull/27770) | 2026-07-27 | `DeepSeek V4` | [P/D disagg] Decode-side radix cache for SWA hybrid models (unified radix tree) |
| [#28354](https://github.com/sgl-project/sglang/pull/28354) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [FlashInfer v0.6.16] Support FlashInfer CuTe DSL NVFP4 MoE quantization |
| [#30393](https://github.com/sgl-project/sglang/pull/30393) | 2026-07-27 | `DeepSeek V4` | [HiCache] Add HiCache draft sidecar pool support for MTP/EAGLE |
| [#30825](https://github.com/sgl-project/sglang/pull/30825) | 2026-07-27 | `MLA` | [FullCG] Support chunked cached-prefix prefill |
| [#30967](https://github.com/sgl-project/sglang/pull/30967) | 2026-07-27 | `FP4`, `GDN`, `MoE`, `NVFP4`, `Qwen3.5` | [GDN] Add MTP cache mode for final-state recompute, with FlashInfer kernel integration and overlapped CUDA-graph state recovery |
| [#31057](https://github.com/sgl-project/sglang/pull/31057) | 2026-07-27 | `MLA` | feat(mem_cache): semantic KV cache reuse via a pluggable fuzzy-match radix backend |
| [#31099](https://github.com/sgl-project/sglang/pull/31099) | 2026-07-27 | `FP4`, `NVFP4` | Remove stale non-Standard-GQA gate for Nemotron BCG |
| [#31220](https://github.com/sgl-project/sglang/pull/31220) | 2026-07-27 | `FP4`, `GDN`, `MoE`, `NVFP4`, `Qwen3.5` | Qwen3.5-MoE: support modelopt_fp4 checkpoints that quantize attention (+ load baked FP8 KV scales) |
| [#31470](https://github.com/sgl-project/sglang/pull/31470) | 2026-07-27 | `DeepSeek V4`, `FP4`, `MoE`, `NVFP4` | Mega moe flashinfer |
| [#31522](https://github.com/sgl-project/sglang/pull/31522) | 2026-07-27 | `GDN`, `MoE`, `Qwen3.5` | [LoRA] Support GDN in_proj_ba adapters for Qwen3.5 |
| [#31768](https://github.com/sgl-project/sglang/pull/31768) | 2026-07-27 | `MoE` | [Model] Add LLaDA2.2 Block Routing MoE support |
| [#31946](https://github.com/sgl-project/sglang/pull/31946) | 2026-07-27 | `MLA` | [HiCache]handle TP-replicated hybrid cache backups per pool |
| [#32104](https://github.com/sgl-project/sglang/pull/32104) | 2026-07-27 | `Kimi K2.5` | [EPD][VLM] Fix Kimi-VL 2D encoder grids |
| [#32405](https://github.com/sgl-project/sglang/pull/32405) | 2026-07-27 | `FP4`, `MoE` | [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner |
| [#32413](https://github.com/sgl-project/sglang/pull/32413) | 2026-07-27 | `DeepSeek V4` | [PD] Handle unsupported decode KV retraction |
| [#32480](https://github.com/sgl-project/sglang/pull/32480) | 2026-07-27 | `FP4`, `Inkling`, `MoE`, `NVFP4` | [FA4] SM100 relative-bias decode kernel + paged dispatch |
| [#32541](https://github.com/sgl-project/sglang/pull/32541) | 2026-07-27 | `Kimi K3` | [Kimi] Support kimi-k3 |
| [#32544](https://github.com/sgl-project/sglang/pull/32544) | 2026-07-27 | `Kimi K3` | [NPU][Kimi] Support Kimi-K3 on NPU |
| [#32555](https://github.com/sgl-project/sglang/pull/32555) | 2026-07-27 | `GDN`, `KDA` | [Nemotron] Fix decode track-save reading the stale tail of the CUDA-graph track buffer |
| [#32556](https://github.com/sgl-project/sglang/pull/32556) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | Autotune flashinfer extend buckets at warmup |
| [#32557](https://github.com/sgl-project/sglang/pull/32557) | 2026-07-27 | `DeepSeek V4` | [Fix] Clear generic legacy CP alias in DeepSeek V4 hook (fixes crash with --enable-prefill-cp) |
| [#32561](https://github.com/sgl-project/sglang/pull/32561) | 2026-07-27 | `Inkling` | Fix: Prevent redundant text re-tokenization on text-only requests for multimodal models |
| [#32565](https://github.com/sgl-project/sglang/pull/32565) | 2026-07-27 | `MoE` | feat: expand weight loader v2 PR2 coverage |
| [#32567](https://github.com/sgl-project/sglang/pull/32567) | 2026-07-27 | `Kimi K3` | fix kimi-k3 reasoning parser on elided think close |
| [#32568](https://github.com/sgl-project/sglang/pull/32568) | 2026-07-27 | `FP4`, `KDA`, `Kimi K3`, `MLA`, `MoE`, `Qwen3.5` | [AMD] Add Kimi-K3 8-GPU MI35x nightly accuracy CI |

## vllm-project/vllm

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#42351](https://github.com/vllm-project/vllm/pull/42351) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | Create `test_kimi_k2_thinking_nvfp4.py` for accuracy check |
| [#42458](https://github.com/vllm-project/vllm/pull/42458) | 2026-07-27 | `FP4`, `NVFP4` | Kimi NVFP4 specialized model |
| [#43229](https://github.com/vllm-project/vllm/pull/43229) | 2026-07-27 | `FP4`, `NVFP4` | [CompressedTensors] FP4 Qutlass Integration |
| [#43952](https://github.com/vllm-project/vllm/pull/43952) | 2026-07-27 | `Kimi K2.5` | Kimi K2.5/2.6 LoRA adapter loading |
| [#44651](https://github.com/vllm-project/vllm/pull/44651) | 2026-07-27 | `FP4`, `MoE` | [ROCm][Bugfix] Add quantization compatibility guard for Fused Shared Expert in DeepSeek-V2/V3/Kimi-K2 |
| [#44941](https://github.com/vllm-project/vllm/pull/44941) | 2026-07-27 | `MoE` | [MoE Refactor] Rename FusedMoE to FusedMoEFactory |
| [#45642](https://github.com/vllm-project/vllm/pull/45642) | 2026-07-27 | `FP4`, `Kimi K2.5`, `NVFP4` | Add Kimi-K2.5 NVFP4 CuTe DSL decode kernels |
| [#45764](https://github.com/vllm-project/vllm/pull/45764) | 2026-07-27 | `MLA` | [BugFix] Add kimi_k2 to MLA allowlist for EAGLE3 draft model detection |
| [#46516](https://github.com/vllm-project/vllm/pull/46516) | 2026-07-27 | `FP4` | Enable gfx1250 ROCm architecture |
| [#46720](https://github.com/vllm-project/vllm/pull/46720) | 2026-07-27 | `DeepSeek V4`, `MLA`, `MoE` | [ROCm][DSV4] B-preshuffle the attention fp8 projections |
| [#46912](https://github.com/vllm-project/vllm/pull/46912) | 2026-07-27 | `GDN`, `KDA` | [Hybird][PrefixCache] Pre-copy-free align prefix cache |
| [#48250](https://github.com/vllm-project/vllm/pull/48250) | 2026-07-27 | `MLA` | Support MLA properly in the Transformers modeling backend |
| [#48280](https://github.com/vllm-project/vllm/pull/48280) | 2026-07-27 | `MoE`, `Qwen3.5`, `Qwen3.6` | [RFC] Heterogeneous rank-to-GPU mapping + Qwen3.5/3.6 GGUF enablement |
| [#48427](https://github.com/vllm-project/vllm/pull/48427) | 2026-07-27 | `MiniMax M3`, `MoE` | [ROCm][Quant] Requantize serialized MXFP8 linears to FP8 PTPC |
| [#49430](https://github.com/vllm-project/vllm/pull/49430) | 2026-07-27 | `KimiLinear` | [Bugfix] Fix pipeline parallelism for Kimi-Linear |
| [#49483](https://github.com/vllm-project/vllm/pull/49483) | 2026-07-27 | `FP4` | [compressed-tensors] update `find_matched_target` order to prioritize fused name matches over class match |
| [#49688](https://github.com/vllm-project/vllm/pull/49688) | 2026-07-27 | `GDN`, `Qwen3.5` | [Bugfix][CPU] Use C++ causal_conv1d kernels for GDN attention on non-AMX AVX-512BF16 CPUs |
| [#49894](https://github.com/vllm-project/vllm/pull/49894) | 2026-07-27 | `MoE` | [Misc] Add unit test for moe_fused_mul_sum Triton kernel |
| [#50000](https://github.com/vllm-project/vllm/pull/50000) | 2026-07-27 | `Kimi K3` | [New model] Kimi K3 |
| [#50019](https://github.com/vllm-project/vllm/pull/50019) | 2026-07-27 | `MoE` | Enable ModelOpt FP8 emulation on SM80 |
| [#50030](https://github.com/vllm-project/vllm/pull/50030) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [Quantization] Add per-token NVFP4 CuTe-DSL MoE backend |
| [#50040](https://github.com/vllm-project/vllm/pull/50040) | 2026-07-27 | `MoE` | [Perf] Use Triton moe backend for tensor fp8 quant scheme on Hopper |
