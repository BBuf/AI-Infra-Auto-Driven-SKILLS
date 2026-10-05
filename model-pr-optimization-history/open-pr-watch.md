# Open PR Watch

Generated: `2026-10-05`.

This report is a triage aid for skill updates. Read the linked PR diffs
before changing benchmark, profiler, or model-history guidance.

## NVIDIA/TensorRT-LLM

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#14579](https://github.com/NVIDIA/TensorRT-LLM/pull/14579) | 2026-10-04 | `MoE`, `Qwen3.5` | [#14561][fix] accept W4A16_AWQ scales in qwen3_5 split-qkv path |
| [#14586](https://github.com/NVIDIA/TensorRT-LLM/pull/14586) | 2026-10-04 | `FP4`, `MoE`, `NVFP4`, `fused` | [#14500][fix] avoid TypedStorage wrapper in CUTLASS MoE expert key on UMA GPUs |
| [#17306](https://github.com/NVIDIA/TensorRT-LLM/pull/17306) | 2026-10-04 | `FP4`, `MLA` | [None][fix] Support beam search with C++ KVCacheManagerV2 |
| [#18392](https://github.com/NVIDIA/TensorRT-LLM/pull/18392) | 2026-10-05 | `CUDA graph`, `DSpark`, `DeepSeek V4`, `MoE`, `fused`, `fusion`, `overlap` | [TRTLLM-14620][feat] Add confidence-guided dynamic verification for DeepSeek-V4 DSpark |
| [#18485](https://github.com/NVIDIA/TensorRT-LLM/pull/18485) | 2026-10-05 | `CUDA graph`, `MLA`, `fused` | [TRTLLM-14620][feat] Add ragged sparse-attention row contracts |
| [#18599](https://github.com/NVIDIA/TensorRT-LLM/pull/18599) | 2026-10-05 | `Kimi K3`, `MLA` | [https://nvbugs/6691772][fix] Update trtllm-gen fmha autotuner for 96 head MLA |
| [#19213](https://github.com/NVIDIA/TensorRT-LLM/pull/19213) | 2026-10-04 | `CUDA graph`, `DeepSeek V4`, `Qwen3.8` | [None][fix] Scope KVCM warmup capacity constraints to DeepSeek V4 |
| [#19418](https://github.com/NVIDIA/TensorRT-LLM/pull/19418) | 2026-10-04 | `overlap` | [DIS-2903][feat] Expose scheduled batch size without iteration statistics |
| [#19441](https://github.com/NVIDIA/TensorRT-LLM/pull/19441) | 2026-10-05 | `CUDA graph`, `DFlash`, `DSpark`, `DeepSeek V4`, `MLA` | [https://nvbugs/6797544][feat] Unified Dspark KVCache Disagg |
| [#19448](https://github.com/NVIDIA/TensorRT-LLM/pull/19448) | 2026-10-05 | `FP4`, `MoE`, `NVFP4`, `fused`, `fusion` | [#19362][fix] Name the act fusion in the skip reason |
| [#19494](https://github.com/NVIDIA/TensorRT-LLM/pull/19494) | 2026-10-04 | `FP4`, `Kimi K3` | [https://nvbugs/6791675][fix] Propagate client sys.path to remote MPI workers |
| [#19568](https://github.com/NVIDIA/TensorRT-LLM/pull/19568) | 2026-10-05 | `CUDA graph` | [#19527][fix] Account for beam width when sizing CUDA graph warmup KV requests |
| [#19613](https://github.com/NVIDIA/TensorRT-LLM/pull/19613) | 2026-10-05 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] Allow prefix-tokenization cache for MiniMax-M3 text-only prompts |
| [#19646](https://github.com/NVIDIA/TensorRT-LLM/pull/19646) | 2026-10-05 | `DeepSeek V4`, `FP4`, `MoE`, `fused`, `overlap` | [None][feat] Add DynamicEPLB kernels and runtime integration |
| [#19681](https://github.com/NVIDIA/TensorRT-LLM/pull/19681) | 2026-10-05 | `MiniMax M3` | [None][perf] Remove IndexK duplication with TEP for MinimaxM3 |
| [#19683](https://github.com/NVIDIA/TensorRT-LLM/pull/19683) | 2026-10-05 | `FP4`, `MiniMax M3`, `NVFP4` | [None][chore] Align the prefix-tokenization cache switch with main (enable_tokenization_cache) |
| [#19698](https://github.com/NVIDIA/TensorRT-LLM/pull/19698) | 2026-10-05 | `fused` | [None][fix] Initialize cu_kv_seqlens for packed-QKV context FMHA |
| [#19707](https://github.com/NVIDIA/TensorRT-LLM/pull/19707) | 2026-10-05 | `DSpark` | [TRTLLM-14620][feat] Add ragged DSpark verification semantics |
| [#19727](https://github.com/NVIDIA/TensorRT-LLM/pull/19727) | 2026-10-04 | `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `MoE`, `NVFP4` | [TRTLLM-16621][feat] Support DeepSeek-V4.1-Flash |
| [#19731](https://github.com/NVIDIA/TensorRT-LLM/pull/19731) | 2026-10-05 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] MiniMax-M3: native P128 draft KV view for the NVFP4 hybrid cache |
| [#19768](https://github.com/NVIDIA/TensorRT-LLM/pull/19768) | 2026-10-04 | `CUDA graph`, `FP4`, `MiniMax M3`, `MoE`, `NVFP4` | [None][perf] reuse MiniMax-M3 NVFP4 query-length buffer |
| [#19809](https://github.com/NVIDIA/TensorRT-LLM/pull/19809) | 2026-10-05 | `overlap` | [https://nvbugs/6838020][fix] Tolerate near-tie greedy flips in KV pool rebalance accuracy test |
| [#19814](https://github.com/NVIDIA/TensorRT-LLM/pull/19814) | 2026-10-04 | `DSpark`, `FP4`, `Kimi K3`, `MLA` | [None][feat] One-model speculative decoding: the worker produces the target logits |
| [#19816](https://github.com/NVIDIA/TensorRT-LLM/pull/19816) | 2026-10-04 | `CUDA graph`, `DFlash`, `DSpark`, `FP4`, `Kimi K3`, `MLA`, `overlap` | [None][perf] Cut host work between speculative decoding steps |
| [#19817](https://github.com/NVIDIA/TensorRT-LLM/pull/19817) | 2026-10-04 | `CUDA graph`, `DSpark`, `GDN`, `KDA`, `Kimi K3`, `fused` | [None][feat] KDA verify: optional per-token states in the V2 hybrid cache manager |
| [#19818](https://github.com/NVIDIA/TensorRT-LLM/pull/19818) | 2026-10-05 | `DSpark`, `fused` | [TRTLLM-14620][feat] Add DSpark confidence verification policy |
| [#19828](https://github.com/NVIDIA/TensorRT-LLM/pull/19828) | 2026-10-05 | `CUDA graph`, `fused`, `fusion` | [None][fix] MNNVL all-reduce: keep a grown workspace's predecessor alive for captured graphs |
| [#19834](https://github.com/NVIDIA/TensorRT-LLM/pull/19834) | 2026-10-04 | `CUDA graph`, `FP4`, `KDA`, `Kimi K3`, `MLA`, `MoE`, `fused` | [None][feat] modeling_v2 catalog: Kimi K3's generic-path entries and sm_100 cells |
| [#19841](https://github.com/NVIDIA/TensorRT-LLM/pull/19841) | 2026-10-04 | `CUDA graph`, `DFlash`, `DSpark`, `FP4`, `KDA`, `Kimi K3`, `MLA`, `MoE`, `NVFP4`, `PDL`, `fused`, `fusion` | [None][feat] Kimi K3 on modeling_v2: stateful catalog entries, MNNVL and collective decode kernels, drafter attention, MoE route B |
| [#19843](https://github.com/NVIDIA/TensorRT-LLM/pull/19843) | 2026-10-05 | `CUDA graph`, `overlap` | [None][perf] Experimental: compile Triton JIT variants ahead of the forward pass (Mamba2 SSD) |
| [#19845](https://github.com/NVIDIA/TensorRT-LLM/pull/19845) | 2026-10-04 | `MoE`, `fused`, `fusion` | [#19844][feat] Support Aleph Alpha Kolibri 1 model |
| [#19846](https://github.com/NVIDIA/TensorRT-LLM/pull/19846) | 2026-10-04 | `KDA`, `Kimi K3`, `MLA`, `fused` | [None][fix] KV cache manager V2: keep the KDA replay state across a suspend; refuse a pool rebalance with it |
| [#19848](https://github.com/NVIDIA/TensorRT-LLM/pull/19848) | 2026-10-04 | `KDA` | [DO_NOT_REVIEW][feat] Introduce the KDA MTP replay design |
| [#19850](https://github.com/NVIDIA/TensorRT-LLM/pull/19850) | 2026-10-04 | `Qwen3.8` | [None][feat] Add LlmArgs.lm_head_dtype for float32 LM head logits |
| [#19851](https://github.com/NVIDIA/TensorRT-LLM/pull/19851) | 2026-10-04 | `CUDA graph`, `FP4`, `NVFP4` | [https://nvbugs/6759067][fix] Draw the stage-2 video noise in the un-patchified 5-D latent shape and… |
| [#19854](https://github.com/NVIDIA/TensorRT-LLM/pull/19854) | 2026-10-04 | `KDA`, `Kimi K3` | [None][fix] kda_prefill: release evicted scratch with its cache entry |
| [#19855](https://github.com/NVIDIA/TensorRT-LLM/pull/19855) | 2026-10-05 | `FP4` | [None][fix] Bound V2 disaggregated KV transfer admission |

## flashinfer-ai/flashinfer

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#4876](https://github.com/flashinfer-ai/flashinfer/pull/4876) | 2026-10-05 | `CUDA graph`, `fused`, `fusion`, `overlap` | feat(comm): add a PCIe/RDMA Ulysses all-to-all backend |
| [#4926](https://github.com/flashinfer-ai/flashinfer/pull/4926) | 2026-10-03 | `FP4`, `MoE`, `NVFP4` | feat(moe): support independent static scaling for FP4 GEMM2 |
| [#5085](https://github.com/flashinfer-ai/flashinfer/pull/5085) | 2026-10-03 | `FP4`, `NVFP4` | fix: preserve FP8 KV tails in FA2 asymmetric prefill |
| [#5109](https://github.com/flashinfer-ai/flashinfer/pull/5109) | 2026-10-04 | `GDN`, `fused` | feat(gdn): add SM8x prefill for the gated delta rule, fused and chunk-parallel |
| [#5236](https://github.com/flashinfer-ai/flashinfer/pull/5236) | 2026-10-04 | `CUDA graph` | feat(attention): bound batch prefill/decode workspace over a shape range |
| [#5267](https://github.com/flashinfer-ai/flashinfer/pull/5267) | 2026-10-04 | `CUDA graph` | perf(mamba): enable native SM107 stochastic rounding |
| [#5272](https://github.com/flashinfer-ai/flashinfer/pull/5272) | 2026-10-04 | `FP4`, `NVFP4`, `overlap` | feat(attention): FA2 per-(token, head) FP8 KV scale |
| [#5335](https://github.com/flashinfer-ai/flashinfer/pull/5335) | 2026-10-03 | `GDN` | fix(gdn): fence TMEM loads before cg1 shared_acc releases |
| [#5478](https://github.com/flashinfer-ai/flashinfer/pull/5478) | 2026-10-03 | `CUDA graph`, `MLA` | fix(prims-ts): support CuTe DSL 4.7 and 4.8 |
| [#5526](https://github.com/flashinfer-ai/flashinfer/pull/5526) | 2026-10-04 | `CUDA graph`, `fused` | perf(cudnn): reduce attention execution and planning overhead |
| [#5556](https://github.com/flashinfer-ai/flashinfer/pull/5556) | 2026-10-03 | `CUDA graph`, `MiniMax M3` | feat(msa): add MiniMax-M3 speculative sparse decode for SM100/SM103 |
| [#5558](https://github.com/flashinfer-ai/flashinfer/pull/5558) | 2026-10-04 | `DeepSeek V4`, `FP4`, `MoE`, `NVFP4`, `PDL`, `fused` | Add clamped routed NVFP4 decode for DeepSeek-V4-Flash on B200 |
| [#5576](https://github.com/flashinfer-ai/flashinfer/pull/5576) | 2026-10-03 | `CUDA graph`, `MLA` | feat(mla): use device KV lengths with upper-bound FA3 plans |
| [#5672](https://github.com/flashinfer-ai/flashinfer/pull/5672) | 2026-10-05 | `CUDA graph`, `FP4`, `KDA`, `Kimi K3`, `MLA`, `NVFP4`, `PDL`, `fused`, `overlap` | perf(cake_fp8_projection): Kimi-K3 KDA/MLA projection GEMM (per-token FP8 activations x per-block FP8 weights, SM100/SM103) round 6: weight-stream L2 prefetch, 2-CTA weight multicast, 5-stage decode ring and GEMM prefetch |
| [#5753](https://github.com/flashinfer-ai/flashinfer/pull/5753) | 2026-10-03 | `CUDA graph`, `GDN`, `GLM-5`, `GLM-5.2`, `MLA`, `fused`, `overlap` | feat(cake_dense_projection_gemm): GLM-5.2 dense projection GEMMs (fwd/dgrad/wgrad, batched MLA, FP32 router) for SM100/SM107 |
| [#5758](https://github.com/flashinfer-ai/flashinfer/pull/5758) | 2026-10-03 | `FP4`, `NVFP4`, `fused`, `fusion` | feat(prims-ts): add FP8/NVFP4 GEMMs with SwiGLU and RoPE fusion |
| [#5862](https://github.com/flashinfer-ai/flashinfer/pull/5862) | 2026-10-05 | `CUDA graph`, `FP4`, `NVFP4`, `Qwen4`, `fused` | feat(qsa_ops): add Qwen4Exp quantized sparse attention, with the paged block-sparse route and the caller-buffer top-k it runs on |
| [#5913](https://github.com/flashinfer-ai/flashinfer/pull/5913) | 2026-10-05 | `overlap` | perf(cake_fmha): round-5 E4M3 head_dim 128 balanced DCP programs and the hd64 / hd256 uniform-batch fast path on SM100/SM103 |
| [#5977](https://github.com/flashinfer-ai/flashinfer/pull/5977) | 2026-10-03 | `CUDA graph`, `FP4`, `MoE` | fix(moe): skip unrouted (-1) expert ids in the SM12x fp8/mxfp8_mxfp4 grouped MoE route |
| [#6006](https://github.com/flashinfer-ai/flashinfer/pull/6006) | 2026-10-03 | `MoE` | fix(moe): keep CUTLASS MoE autotune cache keys rank-invariant |
| [#6012](https://github.com/flashinfer-ai/flashinfer/pull/6012) | 2026-10-03 | `FP4`, `NVFP4` | fix(attention): support packed FP4 KV in BatchAttentionWithAttentionSinkWrapper |
| [#6013](https://github.com/flashinfer-ai/flashinfer/pull/6013) | 2026-10-04 | `DFlash`, `FP4`, `MLA`, `NVFP4` | feat(page): add 4over6 block scales to the NVFP4 slot-mapping KV writer |
| [#6014](https://github.com/flashinfer-ai/flashinfer/pull/6014) | 2026-10-03 | `CUDA graph`, `overlap` | perf(prims-ts): tune the block-sparse decode kernel for SM103 |
| [#6015](https://github.com/flashinfer-ai/flashinfer/pull/6015) | 2026-10-04 | `CUDA graph`, `FP4`, `Kimi K3`, `MoE`, `NVFP4`, `fused` | perf(cake_fused_moe): Kimi-K3 NVFP4 SiTU experts SM100 + SM103 claim8 FC2 2 CTAs/SM |
| [#6018](https://github.com/flashinfer-ai/flashinfer/pull/6018) | 2026-10-05 | `CUDA graph`, `DeepSeek V4`, `FP4`, `MLA`, `NVFP4`, `fused` | feat(cake_dsv4): NVFP4 DeepSeek-V4 sparse-MLA decode on SM100/SM103 (backend="cake") |
| [#6021](https://github.com/flashinfer-ai/flashinfer/pull/6021) | 2026-10-04 | `CUDA graph`, `FP4`, `NVFP4`, `overlap` | feat(attention): attention sinks over an NVFP4 KV cache in prefill and decode |
| [#6022](https://github.com/flashinfer-ai/flashinfer/pull/6022) | 2026-10-03 | `CUDA graph`, `FP4`, `MoE`, `NVFP4`, `PDL`, `fused` | feat(moe): add TRTLLM Gen SwiGLU StepFun support |
| [#6024](https://github.com/flashinfer-ai/flashinfer/pull/6024) | 2026-10-04 | `FP4`, `NVFP4` | fix(attention): keep the NVFP4 scale-factor loads inside their tile |
| [#6028](https://github.com/flashinfer-ai/flashinfer/pull/6028) | 2026-10-03 | `FP4`, `MoE`, `fused` | Fix CuTe-DSL quantization at the minimum E8M0 scale |
| [#6032](https://github.com/flashinfer-ai/flashinfer/pull/6032) | 2026-10-04 | `CUDA graph`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `MoE` | perf: Add opt-in factorized search for ordinary MoE autotuning |
| [#6035](https://github.com/flashinfer-ai/flashinfer/pull/6035) | 2026-10-05 | `MoE` | Yanqinz/move frost out from experiment |
| [#6037](https://github.com/flashinfer-ai/flashinfer/pull/6037) | 2026-10-04 | `PDL` | [Build] Auto-init cutlass/cccl/spdlog submodules for SM120 JIT |
| [#6041](https://github.com/flashinfer-ai/flashinfer/pull/6041) | 2026-10-04 | `KDA` | docs: document newly added environment variables |
| [#6042](https://github.com/flashinfer-ai/flashinfer/pull/6042) | 2026-10-04 | `MoE`, `fused` | docs: document cutlass fused MoE backend parameter |
| [#6046](https://github.com/flashinfer-ai/flashinfer/pull/6046) | 2026-10-04 | `CUDA graph`, `FP4`, `MoE`, `NVFP4`, `PDL`, `fused`, `overlap` | feat: Cake backend for the StepFun SwiGLU fused MoE FC1 (Blackwell SM100/SM103) |
| [#6047](https://github.com/flashinfer-ai/flashinfer/pull/6047) | 2026-10-05 | `CUDA graph`, `FP4`, `MoE`, `NVFP4`, `fused` | feat(moe_ep): sync NVLinkOneSidedAlltoAll with TensorRT-LLM's one-sided kernels (CFT counted writes, 256 ranks) |
| [#6049](https://github.com/flashinfer-ai/flashinfer/pull/6049) | 2026-10-04 | `CUDA graph`, `MoE`, `fused` | feat(cake_grouped_fp8_gemm): per-call operands and -1 padding rows for the prepared fused grouped FP8 SiLU-quant launch (SM100 / SM103) |
| [#6050](https://github.com/flashinfer-ai/flashinfer/pull/6050) | 2026-10-04 | `GLM-5`, `Kimi K3`, `MLA` | feat: expose split_kv override for monolithic CuTeDSL MLA |
| [#6053](https://github.com/flashinfer-ai/flashinfer/pull/6053) | 2026-10-05 | `MoE`, `fused` | [MoK] Add experimental BF16 training with unequal source inputs |
| [#6054](https://github.com/flashinfer-ai/flashinfer/pull/6054) | 2026-10-05 | `CUDA graph` | feat(cake_all_gather_matmul): accept the engine's [N,K] weight view, any local M and any N % 256 == 0 |
| [#6055](https://github.com/flashinfer-ai/flashinfer/pull/6055) | 2026-10-05 | `CUDA graph`, `MiniMax-H3`, `fused`, `fusion` | feat(cake_minimax_h3): two-launch BF16 pre-attention stage consuming the engine-resident QKV weight (SM100a / SM103a) |
| [#6056](https://github.com/flashinfer-ai/flashinfer/pull/6056) | 2026-10-05 | `CUDA graph`, `fused` | feat(cake_dsa_indexer): round 2 generated programs -- host-dispatched scan pipeline levers and prefix-popcount rank finalize, bitwise identical (SM100 / SM103 / SM107) |

## lightseekorg/tokenspeed

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#1606](https://github.com/lightseekorg/tokenspeed/pull/1606) | 2026-10-02 | `CUDA graph`, `MoE`, `PDL` | perf(moe): pass unpacked routes to FlashInfer |
| [#1607](https://github.com/lightseekorg/tokenspeed/pull/1607) | 2026-10-02 | `CUDA graph`, `PDL` | perf(sampling): skip unused fallback-index reduction in target-only verification |
| [#1644](https://github.com/lightseekorg/tokenspeed/pull/1644) | 2026-09-30 | `CUDA graph`, `FP4`, `KDA`, `Kimi K3`, `MoE`, `NVFP4`, `PDL`, `overlap` | perf(moe): fuse FlashInfer NVFP4 routing-map padding initialization |
| [#1666](https://github.com/lightseekorg/tokenspeed/pull/1666) | 2026-09-28 | `DSpark`, `FP4`, `KDA`, `Kimi K3`, `MoE`, `NVFP4`, `fusion` | feat(kda): Add optional BF16 KDA state and FlashInfer decode/verify support |
| [#1682](https://github.com/lightseekorg/tokenspeed/pull/1682) | 2026-09-30 | `CUDA graph`, `PDL`, `fused`, `overlap` | fix(sampling): acquire PDL inputs before softmax reads |
| [#1687](https://github.com/lightseekorg/tokenspeed/pull/1687) | 2026-09-28 | `CUDA graph`, `DeepSeek V4`, `DeepSeek V4.1`, `MoE` | perf(engram): Fuse input preparation and n-gram hashing |
| [#1732](https://github.com/lightseekorg/tokenspeed/pull/1732) | 2026-10-03 | `KDA` | refactor(kernel): clean up fp8 quant and linear ops |
| [#1741](https://github.com/lightseekorg/tokenspeed/pull/1741) | 2026-10-01 | `MoE` | [wip]perf(moe): add persistent moe gluon kernel for cdna5 |
| [#1767](https://github.com/lightseekorg/tokenspeed/pull/1767) | 2026-09-28 | `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `MoE`, `fused`, `fusion`, `overlap` | perf(mhc): Optimize prefill with MegaMHC and decode with fused all-reduce |
| [#1770](https://github.com/lightseekorg/tokenspeed/pull/1770) | 2026-09-28 | `CUDA graph`, `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `MoE`, `PDL` | perf(moe): Optimize DeepSeek-V4/V4.1 routing and expert execution |
| [#1780](https://github.com/lightseekorg/tokenspeed/pull/1780) | 2026-09-29 | `GLM-5`, `GLM-5.3`, `MLA` | perf(dsa): compute selected-slot attention in absorbed-MLA tensor-core tiles |
| [#1783](https://github.com/lightseekorg/tokenspeed/pull/1783) | 2026-10-05 | `CUDA graph`, `FP4`, `GDN`, `NVFP4`, `PDL`, `Qwen3.8`, `Qwen3.8 Flash Next`, `Qwen4`, `fused` | feat(qwen4-exp): enable breakable prefill graphs for variable lengths |
| [#1830](https://github.com/lightseekorg/tokenspeed/pull/1830) | 2026-10-02 | `FP4`, `GDN`, `Inkling`, `Kimi K3`, `MiniMax M3`, `MoE`, `NVFP4`, `Qwen3.8`, `fused` | Fix Qwen3.8-2.4T loading, NVFP4 MoE gating and the Kimi K3 MoE tail on the sm103 arm runner |
| [#1834](https://github.com/lightseekorg/tokenspeed/pull/1834) | 2026-09-30 | `MoE`, `fused`, `overlap` | fix(scheduler): keep sliding-window decode live at KV exhaustion |
| [#1835](https://github.com/lightseekorg/tokenspeed/pull/1835) | 2026-09-30 | `CUDA graph`, `PDL`, `fused` | fix(sampling): renormalize small vocabularies in fused top-k/top-p |
| [#1836](https://github.com/lightseekorg/tokenspeed/pull/1836) | 2026-09-30 | `fused` | fix(sampling): keep the whole support at top_p >= 1 in fused top-k/top-p |
| [#1854](https://github.com/lightseekorg/tokenspeed/pull/1854) | 2026-09-30 | `CUDA graph`, `FP4`, `KDA`, `Kimi K3`, `MLA`, `NVFP4`, `fused`, `fusion` | [DO NOT MERGE] feat(kimi-k3): support TP sharding for attention projections within ADP |
| [#1857](https://github.com/lightseekorg/tokenspeed/pull/1857) | 2026-10-03 | `CUDA graph`, `DeepSeek V4`, `DeepSeek V4.1` | perf(dsv41): fuse block maxima and bound candidate selection |
| [#1860](https://github.com/lightseekorg/tokenspeed/pull/1860) | 2026-09-30 | `fused` | fix(scheduler): admit a remote prefill as the whole prompt |
| [#1861](https://github.com/lightseekorg/tokenspeed/pull/1861) | 2026-09-30 | `DeepSeek V4`, `fused` | fix(scheduler): pass a promotion boundary the chunk budget cannot reach |
| [#1874](https://github.com/lightseekorg/tokenspeed/pull/1874) | 2026-09-29 | `fused` | fix(rl): dispatch update_weights_from_disk in the scheduler |
| [#1878](https://github.com/lightseekorg/tokenspeed/pull/1878) | 2026-10-02 | `CUDA graph`, `FP4`, `GLM-5`, `GLM-5.2`, `LongCat`, `MiniMax M3`, `MoE`, `NVFP4`, `Qwen3.5`, `Qwen3.8`, `Qwen3.8 Flash Next`, `Qwen4`, `fused` | feat(moe): run TRT-LLM BF16 MoE at 64-aligned intermediate sizes |
| [#1880](https://github.com/lightseekorg/tokenspeed/pull/1880) | 2026-09-29 | `CUDA graph`, `fused`, `fusion` | feat(runtime): add Score API for decision-style typed output |
| [#1887](https://github.com/lightseekorg/tokenspeed/pull/1887) | 2026-09-30 | `GLM-5`, `GLM-5.3`, `MLA` | feat(dsa): support Hopper FP8 KV with FlashMLA |
| [#1889](https://github.com/lightseekorg/tokenspeed/pull/1889) | 2026-10-01 | `CUDA graph`, `MLA`, `Qwen3.8` | perf(attention): reuse one multi-CTA KV counter buffer in the trtllm MHA leaf |
| [#1890](https://github.com/lightseekorg/tokenspeed/pull/1890) | 2026-10-04 | `CUDA graph`, `PDL`, `fused`, `fusion` | feat(kernel): add an optional per-head bias to sigmoid_mul |
| [#1891](https://github.com/lightseekorg/tokenspeed/pull/1891) | 2026-10-02 | `CUDA graph`, `FP4`, `MoE`, `NVFP4`, `fused` | feat(moe): opt-in FP32 correction bias for TRT-LLM DeepSeekV3 routing |
| [#1895](https://github.com/lightseekorg/tokenspeed/pull/1895) | 2026-10-01 | `CUDA graph`, `fused` | feat(communication): [1/3] add projection TP collective primitives |
| [#1917](https://github.com/lightseekorg/tokenspeed/pull/1917) | 2026-10-01 | `MoE` | fix(moe): gather Triton BF16 MoE rows by pointer below sm_100 |
| [#1930](https://github.com/lightseekorg/tokenspeed/pull/1930) | 2026-10-04 | `CUDA graph`, `PDL`, `fused` | feat(layernorm): round the residual sum to BF16, scale residual inputs and enable PDL in the Triton RMSNorm |
| [#1931](https://github.com/lightseekorg/tokenspeed/pull/1931) | 2026-10-02 | `CUDA graph`, `DeepSeek V4`, `DeepSeek V4.1`, `LongCat`, `MiniMax M3` | perf(gemm): add FP32 row-CTA decode GEMV leaves for small-M projections |
| [#1938](https://github.com/lightseekorg/tokenspeed/pull/1938) | 2026-10-03 | `DeepSeek V4`, `DeepSeek V4.1`, `GLM-5`, `GLM-5.3`, `Kimi K3`, `MoE`, `fused` | feat(runtime): emulate rank 0 of a parallel layout on one GPU |
| [#1940](https://github.com/lightseekorg/tokenspeed/pull/1940) | 2026-10-05 | `CUDA graph`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `Kimi K2.5`, `NVFP4`, `fused` | perf(memory): reserve what startup keeps resident before the graph probe |
| [#1941](https://github.com/lightseekorg/tokenspeed/pull/1941) | 2026-10-03 | `fusion` | perf(amd): tune packed prefill with upstream gfx1250 controls |
| [#1950](https://github.com/lightseekorg/tokenspeed/pull/1950) | 2026-10-04 | `CUDA graph`, `LongCat`, `MoE`, `overlap` | feat(cache): add sparse KV offloading with fixed GPU hot pools and cross-layer prefetch |
| [#1972](https://github.com/lightseekorg/tokenspeed/pull/1972) | 2026-10-05 | `MoE` | fix(test): restore loading dtype on failure and update V4.1 runtime tests |
| [#1993](https://github.com/lightseekorg/tokenspeed/pull/1993) | 2026-10-05 | `KDA` | perf(runtime): capture Mamba2 layers in the prefill graph |

## sgl-project/sglang

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#27702](https://github.com/sgl-project/sglang/pull/27702) | 2026-10-05 | `FP4`, `fusion` | [diffusion][ROCm][Perf]: Enable BF16 attention and MXFP4 compilation on ROCm |
| [#28650](https://github.com/sgl-project/sglang/pull/28650) | 2026-10-05 | `fusion` | [diffusion][ROCm][Perf]: Set gfx942 AITER FMHA rounding mode to rtz instead of rtna |
| [#30719](https://github.com/sgl-project/sglang/pull/30719) | 2026-10-05 | `fusion` | [Diffusion][CPU] Adding AMX optimizations for CPU platform |
| [#32503](https://github.com/sgl-project/sglang/pull/32503) | 2026-10-05 | `DeepSeek V4`, `MLA`, `fused`, `fusion` | [Hicache] Enable HiCache L1<->L2 support on Intel XPU |
| [#32779](https://github.com/sgl-project/sglang/pull/32779) | 2026-10-05 | `CUDA graph`, `FP4`, `GLM-5`, `GLM-5.2`, `MLA`, `MoE`, `NVFP4`, `fused`, `overlap` | [SM120&90] Add CUDA fused Triton sparse-MLA prefill backend for DSA |
| [#33395](https://github.com/sgl-project/sglang/pull/33395) | 2026-10-05 | `CUDA graph`, `Qwen3.5`, `fused` | [Speculative] Seed rejection-sampling draft proposals for deterministic inference |
| [#33726](https://github.com/sgl-project/sglang/pull/33726) | 2026-10-05 | `CUDA graph`, `MoE`, `Qwen3.5` | fix(bcg): preserve Qwen3-VL DeepStack inputs during replay |
| [#33743](https://github.com/sgl-project/sglang/pull/33743) | 2026-10-05 | `MoE` | [MoE] Fix flashinfer TRT-LLM BF16 expert weight reload on refit |
| [#33855](https://github.com/sgl-project/sglang/pull/33855) | 2026-10-05 | `fusion` | [diffusion] attention: Allow disabling sequence masking in SP |
| [#34355](https://github.com/sgl-project/sglang/pull/34355) | 2026-10-05 | `MLA`, `Qwen3.5`, `fused` | [XPU] Support decode context parallelism (DCP) on Intel XPU |
| [#34502](https://github.com/sgl-project/sglang/pull/34502) | 2026-10-05 | `FP4`, `fused`, `fusion` | [ROCm] Fuse per-token activation quant into RMSNorm for per-channel quantized attention |
| [#35151](https://github.com/sgl-project/sglang/pull/35151) | 2026-10-05 | `FP4`, `fusion` | [diffusion][AMD] Add support for new quantized attention backends for GFX942/GFX950 |
| [#35807](https://github.com/sgl-project/sglang/pull/35807) | 2026-10-05 | `CUDA graph`, `FP4`, `Kimi K3`, `MLA`, `NVFP4` | [Kimi] trtllm_mla: serve varlen absorbed MLA under captured prefill CUDA graphs (fixes #32655) |
| [#36810](https://github.com/sgl-project/sglang/pull/36810) | 2026-10-05 | `overlap` | perf: fix overlap scheduling for NVIDIA Confidential Computing(CC) on Blackwell |
| [#38764](https://github.com/sgl-project/sglang/pull/38764) | 2026-10-05 | `CUDA graph`, `GLM-5`, `GLM-5.3`, `KDA`, `fused` | [AMD] [GLM5] Add opt-in PTPC FP8 KDA projections on gfx950 |
| [#40492](https://github.com/sgl-project/sglang/pull/40492) | 2026-10-05 | `DFlash`, `DSpark`, `Kimi K3`, `MLA`, `Qwen3.5`, `fused`, `fusion` | [unified-memory] Admit DFLASH and DSPARK with fused draft KV |
| [#40785](https://github.com/sgl-project/sglang/pull/40785) | 2026-10-05 | `CUDA graph`, `DFlash`, `fused` | [JIT] Fuse FP8 KV-cache quantization into the prefix-valid commit kernel |
| [#41006](https://github.com/sgl-project/sglang/pull/41006) | 2026-10-05 | `CUDA graph` | [docker] add runtime-efa target to build images with AWS EFA libs and mooncake-transfer-engine-efa |
| [#41008](https://github.com/sgl-project/sglang/pull/41008) | 2026-10-05 | `DFlash`, `DSpark` | [mem_cache] Size the unified-pool req_to_token row from the shared headroom helper |
| [#41134](https://github.com/sgl-project/sglang/pull/41134) | 2026-10-05 | `FP4`, `GDN`, `MoE`, `Qwen3.5`, `fused`, `fusion`, `overlap` | [AMD] Small-M W8A8 FP8 projection GEMM for Qwen3.5 AttnFP8 on gfx950 |
| [#41140](https://github.com/sgl-project/sglang/pull/41140) | 2026-10-05 | `Qwen3.6`, `Qwen3.8` | [PD] Consolidate sender and receiver implementations |
| [#41603](https://github.com/sgl-project/sglang/pull/41603) | 2026-10-05 | `CUDA graph`, `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `MLA`, `fused` | [DSV4.1] Add opt-in TRT-LLM sparse attention support |
| [#41906](https://github.com/sgl-project/sglang/pull/41906) | 2026-10-05 | `MiniMax-H3`, `fused`, `fusion` |  [Diffusion] MiniMax-H3 fuse the video VAE decoder's RMSNorm and QK RoPE |
| [#41962](https://github.com/sgl-project/sglang/pull/41962) | 2026-10-05 | `Kimi K3` | [Fix] Release buffered user text at stream end in InternLM, MiniCPM-5 and Hunyuan detectors |
| [#41982](https://github.com/sgl-project/sglang/pull/41982) | 2026-10-05 | `CUDA graph`, `FP4`, `MiniMax M3`, `MoE`, `Qwen3.5`, `fused` | [AMD] gfx950 small-batch MoE: expert-count gate for the small sort and FP8 block-scale small-M kernel |
| [#42008](https://github.com/sgl-project/sglang/pull/42008) | 2026-10-05 | `CUDA graph`, `fusion` | [AMD][Diffusion] Run residual_gate_add Triton kernels on ROCm without PTX |
| [#42030](https://github.com/sgl-project/sglang/pull/42030) | 2026-10-05 | `DeepSeek V4`, `GLM-5`, `GLM-5.3`, `MoE`, `Qwen3.5` | [Fix] GLM/Qwen3: keep a <tool_call> quoted in reasoning out of streamed tool calls |
| [#42057](https://github.com/sgl-project/sglang/pull/42057) | 2026-10-05 | `CUDA graph`, `DFlash`, `FP4`, `NVFP4`, `Qwen3.8`, `fused` | [Spec] LiLiCorr: named head MLPs and quantized head linears |
| [#42101](https://github.com/sgl-project/sglang/pull/42101) | 2026-10-05 | `CUDA graph`, `overlap` | [Perf] Optimize MiMo local audio attention on Hopper |
| [#42175](https://github.com/sgl-project/sglang/pull/42175) | 2026-10-05 | `CUDA graph`, `GDN`, `MoE`, `PDL`, `Qwen3.8`, `Qwen3.8 Flash Next`, `Qwen4`, `overlap` | [Performance] Optimize Qwen3.8-Flash-Next BF16 decode on Blackwell |
| [#42226](https://github.com/sgl-project/sglang/pull/42226) | 2026-10-05 | `fusion` | [Diffusion] Keep nightly regression baselines on a comparable methodology |
| [#42227](https://github.com/sgl-project/sglang/pull/42227) | 2026-10-05 | `fusion` | [Diffusion] Keep model compatibility details in cookbook recipes |
| [#42239](https://github.com/sgl-project/sglang/pull/42239) | 2026-10-05 | `CUDA graph`, `MoE`, `fusion` | [Perf] Accelerate DiffusionGemma with FA4 and CUDA graphs on Blackwell |
| [#42249](https://github.com/sgl-project/sglang/pull/42249) | 2026-10-05 | `fusion` | fix(diffusion): preserve lazy quantization exports and per-backend lookup |
| [#42254](https://github.com/sgl-project/sglang/pull/42254) | 2026-10-05 | `CUDA graph`, `DeepSeek V4`, `GLM-5`, `GLM-5.3`, `Qwen3.5`, `overlap` | Foundry Adapter |
| [#42285](https://github.com/sgl-project/sglang/pull/42285) | 2026-10-05 | `Inkling`, `MLA`, `MoE`, `fusion` | [Refactor][TCPCG] Relocate shared graph tensor and DSA head-gate helpers (1/9) |
| [#42406](https://github.com/sgl-project/sglang/pull/42406) | 2026-10-05 | `CUDA graph`, `FP4`, `GDN`, `MoE`, `NVFP4`, `Qwen3.5`, `fused` | [cake_kernels] opt-in model routes: Qwen3.5 GDN prefill/decode, FP8 grouped MoE, NVFP4 warp-decode MoE (SGLANG_CAKE_ROUTES) + Qwen3.5 e2e report |
| [#42489](https://github.com/sgl-project/sglang/pull/42489) | 2026-10-05 | `CUDA graph`, `DFlash`, `GLM-5`, `GLM-5.3`, `KDA`, `MoE` | [Perf] Improve GLM-5.3-Flash DFlash decoding on B300 |
| [#42505](https://github.com/sgl-project/sglang/pull/42505) | 2026-10-05 | `MoE`, `fusion` | [diffusion] CI: relax the load latency check and extend the 2-GPU retry deadline |
| [#42531](https://github.com/sgl-project/sglang/pull/42531) | 2026-10-05 | `FP4`, `MiniMax-H3`, `NVFP4`, `fused`, `fusion`, `overlap` | [cake_kernels] minimax_h3 route: pass the engine's native operands to the fused stages |
| [#42541](https://github.com/sgl-project/sglang/pull/42541) | 2026-10-05 | `CUDA graph`, `fused` | Revert "[AMD][ROCm] Keep cos_sin_cache fp32 on HIP for fused QSA indexer kernel (#41282)" |
| [#42542](https://github.com/sgl-project/sglang/pull/42542) | 2026-10-05 | `overlap` | [mlx] Clamp decode-KV flush to the owned prefix at release |

## vllm-project/vllm

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#51339](https://github.com/vllm-project/vllm/pull/51339) | 2026-10-05 | `GLM-5`, `GLM-5.2` | [Model Loader][Perf] Auto-prefetch VirtioFS checkpoints |
| [#51406](https://github.com/vllm-project/vllm/pull/51406) | 2026-10-05 | `MoE`, `Qwen3.5`, `fused`, `fusion` | [ROCm] Enable fused QK-norm+RoPE+gate Triton kernel for Qwen3-Next/Qwen3.5 |
| [#52244](https://github.com/vllm-project/vllm/pull/52244) | 2026-10-05 | `GDN`, `Qwen3.5` | [Bugfix][V1] Restore hybrid GDN prefix-cache hits under MTP spec decoding |
| [#54771](https://github.com/vllm-project/vllm/pull/54771) | 2026-10-05 | `CUDA graph` | [Performance][Pooling] Bulk-submit offline requests to avoid microbatch fragmentation |
| [#55191](https://github.com/vllm-project/vllm/pull/55191) | 2026-10-05 | `overlap` | [Bugfix][KV Offload] Keep direct caches topology-specific |
| [#55199](https://github.com/vllm-project/vllm/pull/55199) | 2026-10-05 | `Inkling`, `overlap` | [Bugfix][Parser] Preserve prose after tool calls |
| [#55203](https://github.com/vllm-project/vllm/pull/55203) | 2026-10-05 | `overlap` | [Bugfix][Multimodal] Fix per-video cache option reuse |
| [#55227](https://github.com/vllm-project/vllm/pull/55227) | 2026-10-05 | `overlap` | [Bugfix][Core] Skip cascade prefixes during deferred KV frees |
| [#55228](https://github.com/vllm-project/vllm/pull/55228) | 2026-10-05 | `MoE`, `overlap` | [Bugfix][EPLB] Reject unsafe dynamic weight updates |
| [#57733](https://github.com/vllm-project/vllm/pull/57733) | 2026-10-05 | `DFlash`, `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `Kimi K3`, `MoE`, `fused`, `fusion`, `overlap` | [Model Loader] Run model-level post-load finalization from the loader hook |
| [#57926](https://github.com/vllm-project/vllm/pull/57926) | 2026-10-05 | `CUDA graph`, `FP4` | [Bugfix][Core][MRV2] Profile the sampler with the params the warm-up actually uses |
| [#58013](https://github.com/vllm-project/vllm/pull/58013) | 2026-10-05 | `CUDA graph`, `Qwen3.8` | [Perf][Attention] FA2: split mixed prefill/decode batches into two calls |
| [#59263](https://github.com/vllm-project/vllm/pull/59263) | 2026-10-05 | `Kimi K3` | [Bugfix][Frontend] Forward the reasoning wiring to the engine for batch chat completions |
| [#59278](https://github.com/vllm-project/vllm/pull/59278) | 2026-10-05 | `Kimi K2.5`, `Kimi K3`, `fused` | [Model] Extend device-side mm normalization to Kimi K2.5 / K3 |
| [#59443](https://github.com/vllm-project/vllm/pull/59443) | 2026-10-05 | `FP4`, `MoE`, `NVFP4`, `Qwen3.8`, `Qwen3.8 Flash Next`, `Qwen4` | [Bugfix][Qwen4Exp] Load PLE tables unquantized under Quark checkpoints |
| [#59533](https://github.com/vllm-project/vllm/pull/59533) | 2026-10-05 | `CUDA graph`, `FP4`, `NVFP4`, `Qwen4`, `fusion` | [Perf][Qwen4Exp] Merge QSA QKVG and indexer QK projections |
| [#59591](https://github.com/vllm-project/vllm/pull/59591) | 2026-10-05 | `Kimi K3`, `MLA`, `MoE`, `fused` | [ROCm][Perf] Kimi-K3 Store only the current rank's shards in latent MoE up-proj  |
| [#59625](https://github.com/vllm-project/vllm/pull/59625) | 2026-10-05 | `DeepSeek V4`, `Qwen3.8`, `fused` | [KV Connector] Support sleep mode with MooncakeConnector over RDMA |
| [#59674](https://github.com/vllm-project/vllm/pull/59674) | 2026-10-05 | `MoE` | [Bugfix][Kernel] Restore Qwen2 key-bias scores for int4 KV cache |
| [#59684](https://github.com/vllm-project/vllm/pull/59684) | 2026-10-05 | `CUDA graph`, `MoE`, `fused` | [ROCm][Bugfix] Warm elastic EP target groups at commit on ROCm |
| [#59804](https://github.com/vllm-project/vllm/pull/59804) | 2026-10-05 | `fused` | [Bugfix][Sampler] Keep tokens top-p must keep in Triton top-k/top-p search |
| [#59902](https://github.com/vllm-project/vllm/pull/59902) | 2026-10-05 | `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `GLM-5`, `GLM-5.3`, `MLA`, `NVFP4` | [Bugfix][KV Connector] Mooncake: transfer GLM-5.3-Flash's kpool indexer pages and tail like NIXL |
| [#59916](https://github.com/vllm-project/vllm/pull/59916) | 2026-10-05 | `CUDA graph`, `DeepSeek V4`, `FP4`, `GLM-5`, `GLM-5.3`, `MLA`, `NVFP4`, `fused` | [Feature][DCP] --dcp-gather: gather-based DCP for MLA attention without native DCP (+ DSv4) |
| [#59927](https://github.com/vllm-project/vllm/pull/59927) | 2026-10-05 | `DeepSeek V4`, `FP4`, `MoE`, `fused` | [Model][DeepSeek-V4] Make MegaMoE shared-expert finalize independent of linear post-load order |
| [#59931](https://github.com/vllm-project/vllm/pull/59931) | 2026-10-05 | `MoE`, `fused` | [Bugfix] Preserve fp16 in legacy fused MoE LoRA |
| [#59942](https://github.com/vllm-project/vllm/pull/59942) | 2026-10-05 | `fusion`, `overlap` | [Bugfix][Core] Retain encoder cache references for repeated multimodal inputs |
| [#59953](https://github.com/vllm-project/vllm/pull/59953) | 2026-10-05 | `MLA` | [Bugfix][NIXL] Count pull completion notifications per transfer |
| [#59973](https://github.com/vllm-project/vllm/pull/59973) | 2026-10-05 | `CUDA graph`, `DFlash`, `DSpark`, `FP4`, `MoE`, `NVFP4`, `Qwen3.5`, `Qwen3.8`, `Qwen3.8 Flash Next`, `Qwen4` | [MRV2][Spec Decode] Opt-in quantized draft lm_head for drafters that share the target head |
| [#60002](https://github.com/vllm-project/vllm/pull/60002) | 2026-10-05 | `overlap` | [Feature][Rust Frontend] Return token offsets from render endpoints |
| [#60006](https://github.com/vllm-project/vllm/pull/60006) | 2026-10-05 | `FP4`, `NVFP4` | [Docs] Rewrite the preload (`ipc_cache`) guide and add Docker/Kubernetes usage |
| [#60009](https://github.com/vllm-project/vllm/pull/60009) | 2026-10-05 | `CUDA graph`, `DFlash`, `DSpark`, `DeepSeek V4`, `DeepSeek V4.1`, `FP4`, `Inkling`, `Kimi K3`, `LongCat`, `MLA`, `MoE`, `NVFP4`, `fused`, `overlap` | [Model][Core] Register every persistent device tensor as a parameter or buffer |
| [#60011](https://github.com/vllm-project/vllm/pull/60011) | 2026-10-05 | `Kimi K3`, `MiniMax M3`, `fused`, `fusion` | [Distributed] Make --disable-custom-all-reduce fall back to NCCL |
| [#60012](https://github.com/vllm-project/vllm/pull/60012) | 2026-10-05 | `GLM-5`, `GLM-5.2`, `GLM-5.3` | [ROCm][Perf] gfx950 device length aware top-k split policy for k=2048 |
