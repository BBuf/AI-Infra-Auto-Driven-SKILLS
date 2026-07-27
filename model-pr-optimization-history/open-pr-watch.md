# Open PR Watch

Generated: `2026-07-27`.

This report is a triage aid for skill updates. Read the linked PR diffs
before changing benchmark, profiler, or model-history guidance.

## NVIDIA/TensorRT-LLM

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#12733](https://github.com/NVIDIA/TensorRT-LLM/pull/12733) | 2026-07-27 | `DeepSeek V4`, `MLA`, `MiniMax M3` | [None][refactor] Unify sparse attention framework with clean backend interfaces |
| [#15343](https://github.com/NVIDIA/TensorRT-LLM/pull/15343) | 2026-07-27 | `MLA` | [https://nvbugs/6287561][fix] Add `get_sm_version() < 90` check at the top of `run_MTP()` in… |
| [#15550](https://github.com/NVIDIA/TensorRT-LLM/pull/15550) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [None][fix] Enable INT8 weight-only (W8A16) MoE for non-gated activations |
| [#15976](https://github.com/NVIDIA/TensorRT-LLM/pull/15976) | 2026-07-27 | `Qwen3.5` | [None][feat] Support MiniCPM-V 4.6 (image + video) on the PyTorch bac… |
| [#16190](https://github.com/NVIDIA/TensorRT-LLM/pull/16190) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [None][feat] Update CuTeDSL MegaMoE kernels |
| [#16224](https://github.com/NVIDIA/TensorRT-LLM/pull/16224) | 2026-07-27 | `DeepSeek V4`, `MLA` | [None][feat] Enable DeepSeek-V4 and DSA (DeepSeek-V3.2/GLM) serving on SM120 via FlashInfer sparse-MLA |
| [#16439](https://github.com/NVIDIA/TensorRT-LLM/pull/16439) | 2026-07-27 | `FP4`, `NVFP4` | [None][fix] Add mutex to avoid potentially concurrent modifications to `std::unordered_map` |
| [#16457](https://github.com/NVIDIA/TensorRT-LLM/pull/16457) | 2026-07-27 | `DeepSeek V4` | [None][perf] GVR top-K decode: enable R0 histogram-ladder admission by default |
| [#16502](https://github.com/NVIDIA/TensorRT-LLM/pull/16502) | 2026-07-27 | `FP4`, `NVFP4` | [None][feat] LTX-2 two-stage: dual-topology parallel Stage 2 (cfg folds into ulysses) |
| [#16598](https://github.com/NVIDIA/TensorRT-LLM/pull/16598) | 2026-07-27 | `Qwen3.5` | [TRTLLM-11875][feat] BREAKING: MambaCacheManager based on KVCacheManagerV2 & agentic prefix caching |
| [#16673](https://github.com/NVIDIA/TensorRT-LLM/pull/16673) | 2026-07-27 | `FP4`, `MoE` | [None][chore] update DeepGEMM to 2.6.1 |
| [#16683](https://github.com/NVIDIA/TensorRT-LLM/pull/16683) | 2026-07-27 | `FP4`, `NVFP4` | [TRTLLM-14557][test] Add single-device visual-gen feature accuracy regression tests |
| [#16768](https://github.com/NVIDIA/TensorRT-LLM/pull/16768) | 2026-07-27 | `GDN` | [TRTLLM-14345][feat] Improve the GDN Replay Kernel Under Low Latency |
| [#16782](https://github.com/NVIDIA/TensorRT-LLM/pull/16782) | 2026-07-27 | `FP4`, `NVFP4` | [TRTLLM-14541][fix] VisualGen: deterministic autotuner tactics across runs and ranks |
| [#16833](https://github.com/NVIDIA/TensorRT-LLM/pull/16833) | 2026-07-27 | `FP4`, `GLM-5`, `MoE`, `NVFP4`, `Qwen3.5` | [None][fix] Fix nemotron-h quant and loading config |
| [#16852](https://github.com/NVIDIA/TensorRT-LLM/pull/16852) | 2026-07-27 | `MiniMax M3` | [None][perf] Fuse MiniMax-M3 prefill projections |
| [#16861](https://github.com/NVIDIA/TensorRT-LLM/pull/16861) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in CutlassFusedMoE |
| [#16862](https://github.com/NVIDIA/TensorRT-LLM/pull/16862) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in TRTLLMGenFusedMoE |
| [#16863](https://github.com/NVIDIA/TensorRT-LLM/pull/16863) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in CuteDslFusedMoE |
| [#16864](https://github.com/NVIDIA/TensorRT-LLM/pull/16864) | 2026-07-27 | `MoE` | [TRTLLM-14609][chore] Remove legacy MoE path in DeepGemmFusedMoE |
| [#16865](https://github.com/NVIDIA/TensorRT-LLM/pull/16865) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [TRTLLM-14609][chore] Remove legacy MoE path in DenseGEMMFusedMoE |
| [#16866](https://github.com/NVIDIA/TensorRT-LLM/pull/16866) | 2026-07-27 | `Qwen3.5` | [https://nvbugs/6240584][fix] Qwen3ToolParser: bare-JSON fallback for reasoning-preceded tool calls |
| [#16890](https://github.com/NVIDIA/TensorRT-LLM/pull/16890) | 2026-07-27 | `DeepSeek V4` | [https://nvbugs/6450333][test] Unwaive DeepSeek V4 Flash auto dtype test |
| [#16900](https://github.com/NVIDIA/TensorRT-LLM/pull/16900) | 2026-07-27 | `FP4` | [None][feat] Add per-rank perf time events capture (TRTLLM_PERF_TIME_EVENTS_PATH) |

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
| [#782](https://github.com/lightseekorg/tokenspeed/pull/782) | 2026-07-23 | `GLM-5`, `GLM-5.2` | perf(kernel): optimize gluon gfx950 DSA top-k selection kernel |
| [#793](https://github.com/lightseekorg/tokenspeed/pull/793) | 2026-07-24 | `MLA`, `MoE` | feat(kernel): support small-batch Gluon MLA decode |
| [#819](https://github.com/lightseekorg/tokenspeed/pull/819) | 2026-07-27 | `FP4`, `MiniMax M3`, `MoE`, `NVFP4` | perf(m3): avoid per-layer D2H sync in MSA prefill |
| [#822](https://github.com/lightseekorg/tokenspeed/pull/822) | 2026-07-27 | `Kimi K3` | feat(kimi-k3): integrate Kimi K3 support |

## sgl-project/sglang

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#18589](https://github.com/sgl-project/sglang/pull/18589) | 2026-07-27 | `MoE` | Make per‑token expert‑distribution recorder use asynchronous data syncing |
| [#28655](https://github.com/sgl-project/sglang/pull/28655) | 2026-07-27 | `FP4`, `GDN`, `Qwen3.5` | [AMD][AITER-Wait] GDN linear out proj fusion |
| [#29173](https://github.com/sgl-project/sglang/pull/29173) | 2026-07-27 | `GLM-5`, `GLM-5.2` | feat: Session-reference-aware Unified Radix Cache for agentic multi-turn workloads |
| [#29723](https://github.com/sgl-project/sglang/pull/29723) | 2026-07-27 | `FP4`, `GDN`, `Qwen3.5` | [AMD] Add fused all-reduce RMSNorm per-token FP8/MXFP4 quant |
| [#30024](https://github.com/sgl-project/sglang/pull/30024) | 2026-07-27 | `MLA` | perf(sgl-kernel): default block_quota=16 for MLA page_first KV gather… |
| [#30715](https://github.com/sgl-project/sglang/pull/30715) | 2026-07-27 | `FP4`, `GLM-5`, `GLM-5.2` | [AMD] [GLM5] Fuse DSA indexer query Hadamard + FP8 quant into one Triton kernel (gfx950) |
| [#30778](https://github.com/sgl-project/sglang/pull/30778) | 2026-07-27 | `FP4`, `MoE` | [AMD] Enable MXFP4 GPT-OSS on RDNA via triton_kernels |
| [#30808](https://github.com/sgl-project/sglang/pull/30808) | 2026-07-27 | `FP4`, `GLM-5`, `GLM-5.2`, `MLA`, `MoE` | [AMD] [GLM5] Enable dense-MHA short-context prefill fallback on gfx950 |
| [#30825](https://github.com/sgl-project/sglang/pull/30825) | 2026-07-27 | `MLA` | [FullCG] Support chunked cached-prefix prefill |
| [#31099](https://github.com/sgl-project/sglang/pull/31099) | 2026-07-27 | `FP4`, `NVFP4` | Remove stale non-Standard-GQA gate for Nemotron BCG |
| [#31137](https://github.com/sgl-project/sglang/pull/31137) | 2026-07-27 | `MLA`, `MoE`, `Qwen3.5`, `Qwen3.6` | [AMD] sgl-kernel: enable gfx1151 (RDNA3.5 / Strix Halo) for single-GPU |
| [#31477](https://github.com/sgl-project/sglang/pull/31477) | 2026-07-27 | `GLM-5`, `GLM-5.2` | [Spec][PD] Enable fused TopK for GLM-5.2 MTP IndexShare |
| [#31959](https://github.com/sgl-project/sglang/pull/31959) | 2026-07-27 | `Qwen3.5` | sm120 support for trtllm mha prefill |
| [#32022](https://github.com/sgl-project/sglang/pull/32022) | 2026-07-27 | `MoE`, `Qwen3.5` | fix(qwen3.5): restrict MoE weights to local PP layers |
| [#32041](https://github.com/sgl-project/sglang/pull/32041) | 2026-07-27 | `FP4` | [diffusion][npu][quant] Add FA MXFP8 quantization support for Wan2.2 Diffusion on Ascend NPU |
| [#32046](https://github.com/sgl-project/sglang/pull/32046) | 2026-07-27 | `FP4`, `MoE`, `Qwen3.5` | [AMD]Qwen3.5 integration gfx950 fmha fp8 hd256 |
| [#32405](https://github.com/sgl-project/sglang/pull/32405) | 2026-07-27 | `FP4`, `MoE` | [MoE Refactor] Migrate SM100 trtllm-gen mxfp4 MoE onto MoeRunner |
| [#32442](https://github.com/sgl-project/sglang/pull/32442) | 2026-07-27 | `MoE` | [MLX] Fix test_batched_decode_matches_solo failing past EOS |
| [#32443](https://github.com/sgl-project/sglang/pull/32443) | 2026-07-27 | `GDN`, `Qwen3.5`, `Qwen3.6` | [Qwen3.5] Fuse gated RMSNorm and FP8 quantization |
| [#32508](https://github.com/sgl-project/sglang/pull/32508) | 2026-07-27 | `DFlash` | fix: account for DFlash draft KV in HybridSWAPoolConfigurator cell_size |
| [#32516](https://github.com/sgl-project/sglang/pull/32516) | 2026-07-27 | `MoE` | [AMD] Enable AITER CK bpreshuffle w8a8 block GEMM on gfx942 (MI300X), opt-in |
| [#32539](https://github.com/sgl-project/sglang/pull/32539) | 2026-07-27 | `DeepSeek V4` | fix(parsers): tools are leaking in the reasoning content (deepseek) |
| [#32541](https://github.com/sgl-project/sglang/pull/32541) | 2026-07-27 | `Kimi K3` | [Kimi] Support kimi-k3 |
| [#32543](https://github.com/sgl-project/sglang/pull/32543) | 2026-07-27 | `MoE` | [AMD] sgl-kernel: enable RDNA (wave32) build with multi-arch warp size |
| [#32544](https://github.com/sgl-project/sglang/pull/32544) | 2026-07-27 | `Kimi K3` | [NPU][Kimi] Support Kimi-K3 on NPU |
| [#32545](https://github.com/sgl-project/sglang/pull/32545) | 2026-07-27 | `Kimi K3` | [Kimi-K3] Clone branch source in Docker images |
| [#32546](https://github.com/sgl-project/sglang/pull/32546) | 2026-07-27 | `Kimi K3` | docs(cookbook): fix broken PD disaggregation anchor on Kimi-K3 |

## vllm-project/vllm

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#41834](https://github.com/vllm-project/vllm/pull/41834) | 2026-07-27 | `DeepSeek V4`, `FP4`, `GLM-5`, `MLA`, `MoE`, `NVFP4` | [New Model][Nvidia] Add SM12x support for DeepSeek V4 Flash with essential fixes |
| [#43229](https://github.com/vllm-project/vllm/pull/43229) | 2026-07-27 | `FP4`, `NVFP4` | [CompressedTensors] FP4 Qutlass Integration |
| [#44941](https://github.com/vllm-project/vllm/pull/44941) | 2026-07-27 | `MoE` | [MoE Refactor] Rename FusedMoE to FusedMoEFactory |
| [#45535](https://github.com/vllm-project/vllm/pull/45535) | 2026-07-27 | `FP4`, `MoE`, `NVFP4` | [Model][Quant] compressed-tensors WNA16 input embeddings + tied embedding (lm_head) support |
| [#45545](https://github.com/vllm-project/vllm/pull/45545) | 2026-07-27 | `FP4`, `Kimi K2.5`, `NVFP4` | Add Kimi video chunk splitting |
| [#46701](https://github.com/vllm-project/vllm/pull/46701) | 2026-07-27 | `MoE` | [Core][V1] Support trace_decode_token_ids for deterministic decode replay |
| [#46720](https://github.com/vllm-project/vllm/pull/46720) | 2026-07-27 | `DeepSeek V4`, `MLA`, `MoE` | [ROCm][DSV4] B-preshuffle the attention fp8 projections |
| [#47355](https://github.com/vllm-project/vllm/pull/47355) | 2026-07-27 | `MLA` | [Attention] Overlap sparse MLA indexer with native CUDA streams |
| [#48962](https://github.com/vllm-project/vllm/pull/48962) | 2026-07-27 | `MLA`, `MoE` | [Do not merge!] [Build] Migrate vendored DeepGEMM from pybind to TORCH_LIBRARY (abi3) |
| [#49436](https://github.com/vllm-project/vllm/pull/49436) | 2026-07-27 | `FP4`, `NVFP4`, `Qwen3.5` | [Perf][Hybrid] 3D-grid tiling of the state-copy Triton kernels |
| [#49483](https://github.com/vllm-project/vllm/pull/49483) | 2026-07-27 | `FP4` | [compressed-tensors] update `find_matched_target` order to prioritize fused name matches over class match |
| [#49629](https://github.com/vllm-project/vllm/pull/49629) | 2026-07-27 | `GLM-5`, `GLM-5.2`, `MLA` | HiSparse: host-resident sparse-MLA decode hot-buffering + GLM-5.2 indexCache opts |
| [#49688](https://github.com/vllm-project/vllm/pull/49688) | 2026-07-27 | `GDN`, `Qwen3.5` | [CPU] Use C++ causal_conv1d kernels for GDN attention on non-AMX AVX-512BF16 CPUs |
| [#49714](https://github.com/vllm-project/vllm/pull/49714) | 2026-07-27 | `DeepSeek V4`, `MoE` | [ROCm][Bugfix] Sanitize AITER paged-MQA logits before sparse top-k for DeepSeek-V4 |
| [#49827](https://github.com/vllm-project/vllm/pull/49827) | 2026-07-27 | `GDN`, `Qwen3.5` | [Model] Enable batch-invariant mixed decode and prefill for Qwen GDN |
| [#49858](https://github.com/vllm-project/vllm/pull/49858) | 2026-07-27 | `MLA` | [KV Offload] Make compact secondary identity TP-independent |
| [#49870](https://github.com/vllm-project/vllm/pull/49870) | 2026-07-27 | `MoE` | [ROCm][Perf] Speed up single-group MoE routing |
| [#49985](https://github.com/vllm-project/vllm/pull/49985) | 2026-07-27 | `MoE` | [Bugfix][CPU] Fall back to torch for unaligned swigluoai on NEON/vec MoE |
| [#49997](https://github.com/vllm-project/vllm/pull/49997) | 2026-07-27 | `KDA` |  3a80b-kda-attnres-dsrouting |
| [#50000](https://github.com/vllm-project/vllm/pull/50000) | 2026-07-27 | `Kimi K3` | [New model] Kimi K3 |
| [#50004](https://github.com/vllm-project/vllm/pull/50004) | 2026-07-27 | `DeepSeek V4`, `MLA` | [DSv4 Perf] Adaptive topk width, 1.0% E2E throughput improvement |
| [#50005](https://github.com/vllm-project/vllm/pull/50005) | 2026-07-27 | `FP4`, `GLM-5`, `GLM-5.2`, `MLA`, `NVFP4` | [Bugfix][DCP] Fix NVIDIA DeepSeek-V3.2 / GLM-5.2 fused attention |
