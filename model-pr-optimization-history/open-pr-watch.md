# Open PR Watch

Generated: `2026-08-23`.

This report is a triage aid for skill updates. Read the linked PR diffs
before changing benchmark, profiler, or model-history guidance.

## NVIDIA/TensorRT-LLM

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#16632](https://github.com/NVIDIA/TensorRT-LLM/pull/16632) | 2026-08-22 | `MoE` | [TRTLLM-14715][feat] preserve native MoE A2A graph VAs across restore |
| [#16913](https://github.com/NVIDIA/TensorRT-LLM/pull/16913) | 2026-08-23 | `MLA` | [None][fix] Prepare offloaded KV blocks for disagg transfer |
| [#16940](https://github.com/NVIDIA/TensorRT-LLM/pull/16940) | 2026-08-22 | `DeepSeek V4`, `FP4`, `MLA`, `MoE`, `NVFP4` | [TRTLLM-14116][feat] Add DeepSeek-V4 Hopper support |
| [#17142](https://github.com/NVIDIA/TensorRT-LLM/pull/17142) | 2026-08-22 | `MoE`, `Qwen3.8` | [TRTLLM-14880][feat] qualify Qwen3 dense for MX |
| [#17236](https://github.com/NVIDIA/TensorRT-LLM/pull/17236) | 2026-08-22 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] Optimize MiniMax-M3 MSA block selection |
| [#17238](https://github.com/NVIDIA/TensorRT-LLM/pull/17238) | 2026-08-22 | `MiniMax M3` | [None][perf] Optimize MiniMax-M3 MXFP8 GEMMs |
| [#17318](https://github.com/NVIDIA/TensorRT-LLM/pull/17318) | 2026-08-22 | `FP4`, `MiniMax M3`, `NVFP4` | [None][perf] Use FP8 MiniMax-M3 MSA indexer QK |
| [#17521](https://github.com/NVIDIA/TensorRT-LLM/pull/17521) | 2026-08-22 | `MoE` | [TRTLLM-15314][feat] Add FP8 LoRA support for B200 |
| [#17564](https://github.com/NVIDIA/TensorRT-LLM/pull/17564) | 2026-08-22 | `FP4` | [https://nvbugs/6590664][fix] Reap idle single-rank CTX transfers |
| [#17693](https://github.com/NVIDIA/TensorRT-LLM/pull/17693) | 2026-08-22 | `FP4`, `NVFP4` | [TRTLLM-15398][perf] VisualGen MLP: cublasLt GELU-tanh epilogue for the unquantized bf16 path |
| [#17821](https://github.com/NVIDIA/TensorRT-LLM/pull/17821) | 2026-08-22 | `DeepSeek V4` | [TRTLLM-15293][perf] Add self-sampling (GVR V2) top-K decode kernels |
| [#17822](https://github.com/NVIDIA/TensorRT-LLM/pull/17822) | 2026-08-22 | `KDA`, `Kimi K3` | [TRTLLM-15498][refactor] consolidate Kimi KDA production frontend |
| [#17831](https://github.com/NVIDIA/TensorRT-LLM/pull/17831) | 2026-08-22 | `MoE` | [https://nvbugs/6601578][fix] Avoid MoE multi-GPU rendezvous port race |
| [#17921](https://github.com/NVIDIA/TensorRT-LLM/pull/17921) | 2026-08-22 | `KDA`, `Kimi K3`, `MLA` | [TRTLLM-15035][test] Wire Kimi K3 spec-dec and suffix-automaton tests into L0 CI |
| [#17948](https://github.com/NVIDIA/TensorRT-LLM/pull/17948) | 2026-08-22 | `Qwen3.5` | [https://nvbugs/6621362][fix] Fix disagg gen stall by aborting peer RX slice on failed KV send |
| [#17971](https://github.com/NVIDIA/TensorRT-LLM/pull/17971) | 2026-08-22 | `DeepSeek V4`, `FP4`, `NVFP4` | [https://nvbugs/6571418][fix] Restore DeepSeek-V4-Pro GSM8K accuracy |
| [#17980](https://github.com/NVIDIA/TensorRT-LLM/pull/17980) | 2026-08-22 | `Kimi K3` | [TRTLLM-15176][fix] Harden Kimi K3 tool-call parsing |
| [#18000](https://github.com/NVIDIA/TensorRT-LLM/pull/18000) | 2026-08-22 | `Qwen3.5` | [TRTLLM-15011][infra] Unwaive TestDeepSeekV4Flash::test_auto_dtype |
| [#18011](https://github.com/NVIDIA/TensorRT-LLM/pull/18011) | 2026-08-22 | `FP4` | [https://nvbugs/6627789][fix] [NVBUG/6627789][fix] Restore CTX-side KV cache transfer overlap flag for… |
| [#18091](https://github.com/NVIDIA/TensorRT-LLM/pull/18091) | 2026-08-22 | `FP4`, `GDN`, `GLM-5`, `GLM-5.2`, `MLA`, `NVFP4` | [None][feat] Add NVFP4 as a cold-page KV Cache Compression Method |
| [#18095](https://github.com/NVIDIA/TensorRT-LLM/pull/18095) | 2026-08-22 | `Qwen3.5` | [https://nvbugs/6625710][fix] Re-attach radix-tree blocks detached under a live request |
| [#18097](https://github.com/NVIDIA/TensorRT-LLM/pull/18097) | 2026-08-22 | `MoE`, `Qwen3.5` | [None][fix] Map qwen3_5/qwen3_5_moe tool-parser auto-selection to qwen3_coder |
| [#18098](https://github.com/NVIDIA/TensorRT-LLM/pull/18098) | 2026-08-22 | `Qwen3.5` | [None][fix] Auto-select reasoning_at_start=True for Qwen3.5-style hybrid templates |

## lightseekorg/tokenspeed

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#616](https://github.com/lightseekorg/tokenspeed/pull/616) | 2026-08-15 | `MLA` | fix(trtllm-mla): make spec-decode CUDA graph capture causal |
| [#617](https://github.com/lightseekorg/tokenspeed/pull/617) | 2026-08-15 | `MLA` | feat(mla): support custom tree masks in decode |
| [#950](https://github.com/lightseekorg/tokenspeed/pull/950) | 2026-08-14 | `MLA`, `MoE` | perf(runtime): avoid Kimi MLA projection layout copies |
| [#968](https://github.com/lightseekorg/tokenspeed/pull/968) | 2026-08-21 | `GLM-5`, `GLM-5.2` | perf(comm): add opt-in Triton AR+RMSNorm backend |
| [#980](https://github.com/lightseekorg/tokenspeed/pull/980) | 2026-08-10 | `FP4`, `MoE` | feat(lora): LoRA adapter serving runtime |
| [#982](https://github.com/lightseekorg/tokenspeed/pull/982) | 2026-08-10 | `MoE` | feat(kernel): Triton LoRA shrink/expand kernels |
| [#983](https://github.com/lightseekorg/tokenspeed/pull/983) | 2026-08-10 | `FP4`, `MoE` | feat(moe): integrate MoE LoRA into the Triton bf16 MoE kernel |
| [#992](https://github.com/lightseekorg/tokenspeed/pull/992) | 2026-08-23 | `DeepSeek V4`, `FP4`, `MLA`, `MoE` | feat(deepseek-v4): support Flash serving on SM120 |
| [#997](https://github.com/lightseekorg/tokenspeed/pull/997) | 2026-08-23 | `DeepSeek V4` | fix(pd): support DeepSeek V4 grouped layerwise cache handoff |
| [#1015](https://github.com/lightseekorg/tokenspeed/pull/1015) | 2026-08-09 | `Kimi K3`, `MLA` | fix(kimi-k3): a DSpark draft's MLA cache cannot diverge from the target's |
| [#1031](https://github.com/lightseekorg/tokenspeed/pull/1031) | 2026-08-22 | `Kimi K3` | feat(kimi-k3): serve DSpark drafts (fc_norm + AttnRes tap) |
| [#1111](https://github.com/lightseekorg/tokenspeed/pull/1111) | 2026-08-15 | `Qwen3.8` | add hopper cookbook for Qwen3.8-27B-FP8 |
| [#1125](https://github.com/lightseekorg/tokenspeed/pull/1125) | 2026-08-21 | `Kimi K3`, `Qwen3.5` | feat(cache): add Mooncake Store as L3 under compact Host KV |
| [#1135](https://github.com/lightseekorg/tokenspeed/pull/1135) | 2026-08-21 | `MoE` | (WIP) perf(kimi3): tune small-batch latent projection |
| [#1137](https://github.com/lightseekorg/tokenspeed/pull/1137) | 2026-08-19 | `FP4` | (WIP) feat(amd): prepare gfx950 SiTU prefill |
| [#1139](https://github.com/lightseekorg/tokenspeed/pull/1139) | 2026-08-19 | `FP4`, `Kimi K3`, `MoE` | (WIP) feat(moe): select gfx950 TP A8W4 SiTU |
| [#1140](https://github.com/lightseekorg/tokenspeed/pull/1140) | 2026-08-20 | `Kimi K3`, `MoE` | (WIP) perf(kimi-k3): tune small-M MoE decode |
| [#1141](https://github.com/lightseekorg/tokenspeed/pull/1141) | 2026-08-21 | `MoE` | (WIP) perf(comm): add two-stage producer-direct reduction |
| [#1144](https://github.com/lightseekorg/tokenspeed/pull/1144) | 2026-08-19 | `MoE` | (WIP) perf(kimi3): join TP MoE reductions |
| [#1145](https://github.com/lightseekorg/tokenspeed/pull/1145) | 2026-08-19 | `MoE` | (WIP) perf(kimi3): shard the TP MoE final projection |
| [#1152](https://github.com/lightseekorg/tokenspeed/pull/1152) | 2026-08-20 | `FP4`, `KDA`, `Kimi K3`, `MLA`, `MoE`, `NVFP4` | fix(cache): support attention-DP for Kimi-K3 by deriving the MLA packing from the KDA state size |
| [#1162](https://github.com/lightseekorg/tokenspeed/pull/1162) | 2026-08-21 | `DeepSeek V4`, `FP4`, `KDA`, `Kimi K3`, `MLA`, `MoE` | feat(mla): add decode context parallelism |
| [#1169](https://github.com/lightseekorg/tokenspeed/pull/1169) | 2026-08-20 | `DFlash` | fix(cache): route null decode pages to dummy slot |
| [#1172](https://github.com/lightseekorg/tokenspeed/pull/1172) | 2026-08-20 | `MoE` | perf(k3): fuse the multi-token MoE front |
| [#1175](https://github.com/lightseekorg/tokenspeed/pull/1175) | 2026-08-20 | `MoE` | [WIP] perf(k3): extend packed top-k to decode batches |
| [#1187](https://github.com/lightseekorg/tokenspeed/pull/1187) | 2026-08-21 | `Inkling`, `Kimi K2.5`, `Kimi K3`, `MoE` | test: kimi-k3 agentic decode-throughput bench |
| [#1201](https://github.com/lightseekorg/tokenspeed/pull/1201) | 2026-08-22 | `DeepSeek V4` | [WIP] feat(deepseek-v4): add AMD MI350 support |
| [#1204](https://github.com/lightseekorg/tokenspeed/pull/1204) | 2026-08-22 | `KDA` | perf(kda): v-major decode and split the verify megafusion for NVIDIA |

## sgl-project/sglang

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#27010](https://github.com/sgl-project/sglang/pull/27010) | 2026-08-23 | `DeepSeek V4`, `FP4`, `MoE` | [HiCache] Fix PP inconsistency with HiCache L3 (#22607) |
| [#30304](https://github.com/sgl-project/sglang/pull/30304) | 2026-08-22 | `MoE` | [Benchmark] Add agentic multi-turn dataset to the serving benchmark |
| [#31768](https://github.com/sgl-project/sglang/pull/31768) | 2026-08-22 | `MoE` | [Model] Add LLaDA2.2 Block Routing MoE support |
| [#33091](https://github.com/sgl-project/sglang/pull/33091) | 2026-08-22 | `Qwen3.5` | [unified-memory] Stop eviction when shared allocation capacity is sufficient |
| [#33569](https://github.com/sgl-project/sglang/pull/33569) | 2026-08-22 | `MiniMax-H3` | [NPU] [Diffusion] Support MiniMax H3 on Ascend NPU's |
| [#33614](https://github.com/sgl-project/sglang/pull/33614) | 2026-08-22 | `DFlash`, `DeepSeek V4`, `MLA` | [Spec] Fix Dspark and Dflash state divergence across TP rank |
| [#33778](https://github.com/sgl-project/sglang/pull/33778) | 2026-08-22 | `GDN`, `Qwen3.5` | Avoid materializing GDN QKV tensors during target verification |
| [#33863](https://github.com/sgl-project/sglang/pull/33863) | 2026-08-22 | `DeepSeek V4`, `FP4`, `Kimi K3`, `MoE` | [Feature] PP Support PD + DSpark |
| [#34198](https://github.com/sgl-project/sglang/pull/34198) | 2026-08-22 | `KDA`, `Kimi K3` | [AMD] Perf Kimi-K3 fuse ROCm KDA decode boundary |
| [#34490](https://github.com/sgl-project/sglang/pull/34490) | 2026-08-22 | `Kimi K3`, `MoE` | [AMD] Add Radix-4 MoE top-k router kernel for Kimi-K3 routing |
| [#34565](https://github.com/sgl-project/sglang/pull/34565) | 2026-08-23 | `DeepSeek V4`, `FP4`, `MoE` | [Unified Tree] Support Branching-Point Caching for the SWA Component |
| [#34613](https://github.com/sgl-project/sglang/pull/34613) | 2026-08-22 | `GDN`, `MLA`, `Qwen3.5` | feat(unified-memory): read unified pool from attention backends fa3/flashinfer/trtllm_mha/flashmla |
| [#34647](https://github.com/sgl-project/sglang/pull/34647) | 2026-08-23 | `Kimi K3`, `MLA` | [AMD] Enable 12-head MLA aiter fp8 Gluon decode (batched bh16bn128). |
| [#34727](https://github.com/sgl-project/sglang/pull/34727) | 2026-08-22 | `Qwen3.8` | [kernel] One rmsnorm kernel for every hidden size, tuned from Python |
| [#35305](https://github.com/sgl-project/sglang/pull/35305) | 2026-08-23 | `Kimi K3` | [Kimi-K3] Fix "wrong grids" crash in DP-sharded vision preprocessing |
| [#35403](https://github.com/sgl-project/sglang/pull/35403) | 2026-08-22 | `DFlash` | [Spec] Route weight updates through the _draft_model_runners() guard |
| [#35457](https://github.com/sgl-project/sglang/pull/35457) | 2026-08-22 | `FP4`, `Qwen3.5` | [AMD][Spec] Pack AITER target-verify GQA for Qwen3.5 |
| [#35954](https://github.com/sgl-project/sglang/pull/35954) | 2026-08-22 | `DFlash`, `DeepSeek V4` | [Fix] Prevent one-at-a-time DFlash/DSpark replacement prefills below the request limit |
| [#35985](https://github.com/sgl-project/sglang/pull/35985) | 2026-08-22 | `Qwen3.5` | Fix FA3 page_table OOB near context wall under speculative decoding |
| [#36004](https://github.com/sgl-project/sglang/pull/36004) | 2026-08-22 | `DeepSeek V4`, `MLA`, `MoE` | [AMD][DSV4] perf: use full 1024-thread block for indexer top-k on ROCm |
| [#36014](https://github.com/sgl-project/sglang/pull/36014) | 2026-08-22 | `GDN`, `KDA`, `Qwen3.8` | [Fix] Align GDN target-verify beta semantics with packed decode |
| [#36020](https://github.com/sgl-project/sglang/pull/36020) | 2026-08-23 | `DFlash`, `FP4`, `NVFP4`, `Qwen3.8` | [docs] Split the Qwen3.8-27B NVFP4 cells by lm_head precision |

## vllm-project/vllm

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#43375](https://github.com/vllm-project/vllm/pull/43375) | 2026-08-22 | `MoE` | [RL] P2P RDT weight sync |
| [#44384](https://github.com/vllm-project/vllm/pull/44384) | 2026-08-22 | `MoE` | [Bugfix][Model] Fix Qwen3 deepstack buffer device mismatch |
| [#44597](https://github.com/vllm-project/vllm/pull/44597) | 2026-08-22 | `Qwen3.8` | Add global cache scope for ngram prompt lookup |
| [#45457](https://github.com/vllm-project/vllm/pull/45457) | 2026-08-22 | `MoE` | [Perf] Reuse topk SparseMatrix routing metadata in GPT-OSS MoE forward |
| [#45535](https://github.com/vllm-project/vllm/pull/45535) | 2026-08-22 | `FP4`, `MoE`, `NVFP4` | [Model][Quant] compressed-tensors WNA16 input embeddings + tied embedding (lm_head) support |
| [#45573](https://github.com/vllm-project/vllm/pull/45573) | 2026-08-23 | `MLA` | [Attention] Porting MLARoPEKVCacheCatFusionPass to manual fusion |
| [#45819](https://github.com/vllm-project/vllm/pull/45819) | 2026-08-22 | `GDN`, `MoE`, `Qwen3.5`, `Qwen3.6` | [Feature] Add batch invariance support to GDN_ATTN backend |
| [#47737](https://github.com/vllm-project/vllm/pull/47737) | 2026-08-22 | `DFlash`, `DeepSeek V4` | [Bugfix] Fix ZeroDivisionError when Dynamic SD schedule includes K=0 for DSpark draft cudagraph capture |
| [#49617](https://github.com/vllm-project/vllm/pull/49617) | 2026-08-22 | `DFlash` | Fix speculators dspark attribute loading |
| [#50514](https://github.com/vllm-project/vllm/pull/50514) | 2026-08-22 | `DFlash`, `Kimi K3` | [Core][MRV2] Support eagle3 spec decode with pipeline parallel |
| [#50519](https://github.com/vllm-project/vllm/pull/50519) | 2026-08-22 | `FP4`, `KDA`, `Kimi K3`, `MLA`, `Qwen3.5` | [ROCm][CI] Add missing test coverage for upstream parity |
| [#52165](https://github.com/vllm-project/vllm/pull/52165) | 2026-08-22 | `DeepSeek V4`, `MLA` | [Misc][Spec Decode] Detect DeepSeek-V4 DSpark checkpoints from config |
| [#52228](https://github.com/vllm-project/vllm/pull/52228) | 2026-08-22 | `DFlash`, `DeepSeek V4`, `FP4`, `Inkling`, `Kimi K2.5`, `NVFP4` | [EXPERIMENTAL][Model Runner V2] Acceptance estimation for non-dspark adaptive verification |
| [#52244](https://github.com/vllm-project/vllm/pull/52244) | 2026-08-22 | `GDN`, `Qwen3.5` | [Bugfix][V1] Restore hybrid GDN prefix-cache hits under MTP spec decoding |
| [#52786](https://github.com/vllm-project/vllm/pull/52786) | 2026-08-22 | `MoE` | [LoRA] Add Qwen3-Omni multimodal LoRA support |
| [#52849](https://github.com/vllm-project/vllm/pull/52849) | 2026-08-22 | `FP4`, `MiniMax M3`, `MoE` | [ROCm][PERF] Enable AITER PA gluon decode for MiniMax-M3 MTP and dense layers |
| [#53247](https://github.com/vllm-project/vllm/pull/53247) | 2026-08-22 | `MoE` | [Kernel][Perf] Per-device tuned configs for batch-invariant persistent matmul (~3x decode kernels on RTX 4090D/H20) |
| [#53351](https://github.com/vllm-project/vllm/pull/53351) | 2026-08-22 | `MLA`, `MiniMax M3` | [ROCm][CI] Restore attention coverage after KV-cache layout refactor |
| [#53388](https://github.com/vllm-project/vllm/pull/53388) | 2026-08-22 | `FP4`, `Kimi K3`, `MLA`, `MoE` | [Feature][Spec] Support disabling trailing prefix-cache block dropping |
| [#53394](https://github.com/vllm-project/vllm/pull/53394) | 2026-08-22 | `MoE` | [Hardware][NVIDIA] Add GB10 fused-MoE fp8 tuning config (E=128, N=704 — Gemma 4 26B A4B) |
| [#53396](https://github.com/vllm-project/vllm/pull/53396) | 2026-08-22 | `KDA`, `Kimi K3` | [K3] Support DS conv-state layout in fused KDA decode kernel |
| [#53397](https://github.com/vllm-project/vllm/pull/53397) | 2026-08-22 | `Qwen3.5` | fix(spec_decode): thread spec_step_idx in llm_base_proposer for multi-layer MTP (#52688) |
| [#53403](https://github.com/vllm-project/vllm/pull/53403) | 2026-08-22 | `GDN` | [Docs] Add Qwen3-0.6B to batch invariance tested models |
| [#53405](https://github.com/vllm-project/vllm/pull/53405) | 2026-08-22 | `DeepSeek V4` | fix(parser): stop leaking partially delivered DSML tags into streamed tool arguments |
| [#53406](https://github.com/vllm-project/vllm/pull/53406) | 2026-08-22 | `FP4`, `GDN`, `NVFP4`, `Qwen3.5`, `Qwen3.8` | [Bugfix] Do not FULL-capture spec-decode batches in TurboQuant attention backend |
| [#53407](https://github.com/vllm-project/vllm/pull/53407) | 2026-08-22 | `Kimi K3` | [Bugfix][MRV2][ROCm] Dispatch uniform decode to a padded FULL cudagraph |
| [#53408](https://github.com/vllm-project/vllm/pull/53408) | 2026-08-22 | `DeepSeek V4`, `Qwen3.5` | [Bugfix] DeepSeekV4MTP: implement SupportsPP so the draft can start under PP |
| [#53410](https://github.com/vllm-project/vllm/pull/53410) | 2026-08-22 | `FP4`, `GDN`, `NVFP4`, `Qwen3.8` | [Perf] TurboQuant: run spec-decode verify batches as decodes with FULL cudagraphs |
| [#53414](https://github.com/vllm-project/vllm/pull/53414) | 2026-08-22 | `FP4`, `Qwen3.5` | fix(quant): bypass fc quantization for compressed-tensors MTP checkpo… |
