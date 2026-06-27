# Open PR Watch

Generated: `2026-06-27`.

This report is a triage aid for skill updates. Read the linked PR diffs
before changing benchmark, profiler, or model-history guidance.

## NVIDIA/TensorRT-LLM

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#12262](https://github.com/NVIDIA/TensorRT-LLM/pull/12262) | 2026-06-27 | `DFlash` | [TRTLLM-11556][feat] Expand dynamic speculation to all spec decode algorithms |
| [#13404](https://github.com/NVIDIA/TensorRT-LLM/pull/13404) | 2026-06-27 | `MoE`, `WideEP` | [TRTLLM-12200][feat] WideEP FT: add active_rank_mask to NVLink AlltoAll kernels (1a.2) |
| [#14599](https://github.com/NVIDIA/TensorRT-LLM/pull/14599) | 2026-06-27 | `MoE`, `Qwen3.5` | [TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE (with the MTP fixes) |
| [#14668](https://github.com/NVIDIA/TensorRT-LLM/pull/14668) | 2026-06-27 | `FP4`, `NVFP4` | [https://nvbugs/5970614][fix] Sync CTA before PDL trigger in quantize_with_block_size |
| [#14810](https://github.com/NVIDIA/TensorRT-LLM/pull/14810) | 2026-06-27 | `FP4`, `MLA`, `MoE`, `NVFP4` | [#14588][feat] AutoDeploy: DeepSeek-R1 optimization for low concurrency |
| [#15099](https://github.com/NVIDIA/TensorRT-LLM/pull/15099) | 2026-06-27 | `FP4`, `NVFP4` | [https://nvbugs/6248837][fix] Free worker GPU memory on executor shutdown |
| [#15343](https://github.com/NVIDIA/TensorRT-LLM/pull/15343) | 2026-06-27 | `MLA` | [https://nvbugs/6287561][fix] Add `get_sm_version() < 90` check at the top of `run_MTP()` in… |
| [#15386](https://github.com/NVIDIA/TensorRT-LLM/pull/15386) | 2026-06-27 | `MoE`, `WideEP` | [TRTLLM-13248][feat] Wave 3: migrate MoE staged hooks |
| [#15414](https://github.com/NVIDIA/TensorRT-LLM/pull/15414) | 2026-06-27 | `FP4`, `MLA`, `MoE`, `NVFP4` | [None][feat] DSv4: model, tokenizer, and integration coverage |
| [#15464](https://github.com/NVIDIA/TensorRT-LLM/pull/15464) | 2026-06-27 | `MLA` | [https://nvbugs/6329052][fix] Add `attn_backend: FLASHINFER` and `model_kwargs: {num_hidden_layers: 4}` to… |
| [#15525](https://github.com/NVIDIA/TensorRT-LLM/pull/15525) | 2026-06-27 | `MoE`, `WideEP` | [TRTLLM-13543][feat] WideEP FT: add EPLB mask-only reconfigure (1b.1) |
| [#15624](https://github.com/NVIDIA/TensorRT-LLM/pull/15624) | 2026-06-27 | `FP4`, `MoE`, `NVFP4`, `WideEP` | [TRTLLM-13629][test] Optimize MoE CI test-db |
| [#15625](https://github.com/NVIDIA/TensorRT-LLM/pull/15625) | 2026-06-27 | `MoE` | [None][perf] DSv4 follow-up: disagg routing improvements |
| [#15633](https://github.com/NVIDIA/TensorRT-LLM/pull/15633) | 2026-06-27 | `DeepSeek V4`, `MLA` | [None][feat] DSv4 follow-up: runtime KV and cache foundations |
| [#15659](https://github.com/NVIDIA/TensorRT-LLM/pull/15659) | 2026-06-27 | `FP4`, `GLM-5`, `NVFP4` | [None][fix] GLM-5.1 NVFP4 fallback to AR-Norm fusion for unquantized dense layers |
| [#15677](https://github.com/NVIDIA/TensorRT-LLM/pull/15677) | 2026-06-27 | `WideEP` | [TRTLLM-13546][feat] Add error classification patterns (1c.1) |
| [#15680](https://github.com/NVIDIA/TensorRT-LLM/pull/15680) | 2026-06-27 | `FP4`, `MoE`, `NVFP4`, `Qwen3.5`, `Qwen3.6` | [#14575][feat] Add Qwen3.5/3.6 MoE NVFP4 + MTP support for SM120/SM121 |
| [#15681](https://github.com/NVIDIA/TensorRT-LLM/pull/15681) | 2026-06-27 | `MoE` | [https://nvbugs/6379316][fix] Keep the prior `7e134dd249` gate that adds `_is_pcie_nvl_sku()` + DeepEP-LL… |
| [#15683](https://github.com/NVIDIA/TensorRT-LLM/pull/15683) | 2026-06-27 | `FP4`, `MLA` | [None][feat] DSA: adaptive indexer prefill chunk size for long sequences |

## lightseekorg/tokenspeed

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#6](https://github.com/lightseekorg/tokenspeed/pull/6) | 2026-06-14 | `MLA` | perf(cache): overlap target and draft KV loadback independently  |
| [#238](https://github.com/lightseekorg/tokenspeed/pull/238) | 2026-06-24 | `FP4` | perf(deepseek-v4): vectorize read_deepseek_v4_indexer_fp8_cache |
| [#280](https://github.com/lightseekorg/tokenspeed/pull/280) | 2026-06-27 | `MoE` | Add Triton sampling backends alongside FlashInfer |
| [#382](https://github.com/lightseekorg/tokenspeed/pull/382) | 2026-06-27 | `GDN` | perf(gdn): fuse causal_conv1d and QKV split for GDN prefill |
| [#383](https://github.com/lightseekorg/tokenspeed/pull/383) | 2026-06-25 | `MoE`, `Qwen3.5` | [WIP] feat(config): runtime config decoupling(design for reference) |
| [#412](https://github.com/lightseekorg/tokenspeed/pull/412) | 2026-06-26 | `FP4`, `NVFP4` | Port mamba2 kernels and runtime from sglang#03c77dc |
| [#416](https://github.com/lightseekorg/tokenspeed/pull/416) | 2026-06-22 | `FP4`, `NVFP4` | Fix EP8 DP/TP RSAG init and empty LM head |
| [#437](https://github.com/lightseekorg/tokenspeed/pull/437) | 2026-06-25 | `FP4`, `NVFP4` | [WIP] EPD: encode-worker path, async embedding receive, E2P row-sharding |
| [#461](https://github.com/lightseekorg/tokenspeed/pull/461) | 2026-06-22 | `FP4`, `NVFP4` | test(ci): add DeepSeek-V4-Flash MTP AIME25 eval |
| [#510](https://github.com/lightseekorg/tokenspeed/pull/510) | 2026-06-25 | `DFlash`, `Qwen3.5` | [WIP] feat:support qwen3.5 dflash |
| [#520](https://github.com/lightseekorg/tokenspeed/pull/520) | 2026-06-25 | `GDN`, `MoE`, `Qwen3.5` | feat: support qwen3.5 on Hopper GPUs |
| [#528](https://github.com/lightseekorg/tokenspeed/pull/528) | 2026-06-26 | `GLM-5`, `GLM-5.2`, `MoE` | [WIP] Initial glm 5.2 support on amd |
| [#532](https://github.com/lightseekorg/tokenspeed/pull/532) | 2026-06-26 | `GLM-5`, `GLM-5.2` | test: glm-5.2 agentic bench |
| [#534](https://github.com/lightseekorg/tokenspeed/pull/534) | 2026-06-26 | `FP4`, `MoE` | Fix gathered MXFP4 activation scales in Gluon MoE |

## sgl-project/sglang

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#27397](https://github.com/sgl-project/sglang/pull/27397) | 2026-06-27 | `FP4`, `GLM-5`, `MLA`, `NVFP4` | Support JIT fused A GEMM (MLA down projection) and support GLM-5 hidden size, SM120 |
| [#27705](https://github.com/sgl-project/sglang/pull/27705) | 2026-06-27 | `FP4`, `GLM-5`, `GLM-5.2`, `NVFP4` | Fuse the DSA (V3.2, GLM-5.x) indexer Q/K paths into single kernels |
| [#28245](https://github.com/sgl-project/sglang/pull/28245) | 2026-06-27 | `FP4`, `MLA`, `MoE`, `NVFP4` | [JIT Kernel] Unify all jit_kernel benchmarks onto the marker framework (drop triton.testing) |
| [#28417](https://github.com/sgl-project/sglang/pull/28417) | 2026-06-27 | `MoE`, `Qwen3.5` |  [NPU] Enable piecewise CUDA graph support on NPU |
| [#28713](https://github.com/sgl-project/sglang/pull/28713) | 2026-06-27 | `MoE` | [minimax-m3] Split 2/4: mem-cache / HiCache / sparse KV pool |
| [#28980](https://github.com/sgl-project/sglang/pull/28980) | 2026-06-27 | `DeepSeek V4` | [NPU] Support DeepSeek V4 Flash MTP on Ascend |
| [#28994](https://github.com/sgl-project/sglang/pull/28994) | 2026-06-27 | `GLM-5`, `GLM-5.2` | [server_args][deepep] DeepEP-safe reland of DP mem_fraction_static + DeepEP capacity auto-tuning |
| [#29029](https://github.com/sgl-project/sglang/pull/29029) | 2026-06-27 | `GLM-5`, `GLM-5.2` | [NPU][Bugfix] Fix a ModelSlim loading failure |
| [#29151](https://github.com/sgl-project/sglang/pull/29151) | 2026-06-27 | `FP4`, `NVFP4` | Fix ModelOpt NVFP4 scalar scales for merged linears |
| [#29161](https://github.com/sgl-project/sglang/pull/29161) | 2026-06-27 | `MLA` | [Fix]: Defer DSA MLA CP KV gather for fp8 trtllm prefill in PD mode |
| [#29186](https://github.com/sgl-project/sglang/pull/29186) | 2026-06-27 | `MoE` | [New Model] Add Baidu Unlimited-OCR |
| [#29223](https://github.com/sgl-project/sglang/pull/29223) | 2026-06-27 | `FP4`, `NVFP4` | (perf): Shard Kimi-K2.5 Eagle3 draft fc + symm-mem AG |
| [#29304](https://github.com/sgl-project/sglang/pull/29304) | 2026-06-27 | `FP4`, `NVFP4` | [Feature] NVFP4 KV cache: SM120 + SM121, Gemma-4 VO-split, FP4 prefix-cache correctness (builds on #21954) |
| [#29305](https://github.com/sgl-project/sglang/pull/29305) | 2026-06-27 | `FP4`, `NVFP4` | [Feature] DiffusionGemma: retire Triton onto FA2 NVFP4 KV cache (stacked on #29304 + #28054) |
| [#29325](https://github.com/sgl-project/sglang/pull/29325) | 2026-06-27 | `DFlash`, `Qwen3.6` | Responses support |
| [#29393](https://github.com/sgl-project/sglang/pull/29393) | 2026-06-27 | `FP4`, `GLM-5`, `GLM-5.2`, `NVFP4` | [Deps] Bump transformers to 5.12.1 |
| [#29439](https://github.com/sgl-project/sglang/pull/29439) | 2026-06-27 | `MoE` | [Apple Silicon][MLX] Fix scheduler OOM on long prompts and concurrent requests |
| [#29464](https://github.com/sgl-project/sglang/pull/29464) | 2026-06-27 | `DFlash` | Fix EAGLE draft hidden dim extraction and centralize spec helpers |
| [#29472](https://github.com/sgl-project/sglang/pull/29472) | 2026-06-27 | `KDA` | [KDA] Add FlashKDA prefill backend for safe-gate KDA linear attention |
| [#29484](https://github.com/sgl-project/sglang/pull/29484) | 2026-06-27 | `MoE` | [Benchmark] Filter fused MoE Triton tuning configs by estimated shared memory |
| [#29485](https://github.com/sgl-project/sglang/pull/29485) | 2026-06-27 | `FP4`, `GLM-5`, `MoE`, `NVFP4` | [BUGFIX] Fix FlashInfer allreduce fusion auto backend on single-node sm_103 |
| [#29490](https://github.com/sgl-project/sglang/pull/29490) | 2026-06-27 | `GDN`, `MoE`, `Qwen3.6` | [MLX] Minimal Apple Silicon server startup for hybrid GDN models |
| [#29491](https://github.com/sgl-project/sglang/pull/29491) | 2026-06-27 | `MLA` | fix(kv-events): offset ZMQ publisher port by dp_rank, not attn_dp_rank |
| [#29493](https://github.com/sgl-project/sglang/pull/29493) | 2026-06-27 | `MoE` | [NPU][Bugfix] Add scoring_func for mimo_v2 |
| [#29495](https://github.com/sgl-project/sglang/pull/29495) | 2026-06-27 | `GDN`, `Qwen3.6` |  [NPU] correct SSM state transpose for spec decode extend with prefix |
| [#29497](https://github.com/sgl-project/sglang/pull/29497) | 2026-06-27 | `FP4`, `MoE`, `Qwen3.5` | [CPU] Fix model failures on Xeon |
| [#29499](https://github.com/sgl-project/sglang/pull/29499) | 2026-06-27 | `FP4`, `GLM-5`, `GLM-5.2`, `NVFP4` | [DSA] Optimize DSA CUDA graph replay metadata generation |

## vllm-project/vllm

| PR | Updated | Matched terms | Title |
| --- | --- | --- | --- |
| [#38433](https://github.com/vllm-project/vllm/pull/38433) | 2026-06-27 | `MLA` | feat(nixl,dcp): Supports MLA DCP for PD disaggregation with nixl connector |
| [#38434](https://github.com/vllm-project/vllm/pull/38434) | 2026-06-27 | `GDN`, `MoE`, `Qwen3.5` | [Fix] Improve ROCm detection in WSL environments |
| [#43950](https://github.com/vllm-project/vllm/pull/43950) | 2026-06-27 | `DeepSeek V4`, `MLA`, `MoE` | [ROCm][DSV4] Use aiter mHC pre/post as the default ROCm path |
| [#44297](https://github.com/vllm-project/vllm/pull/44297) | 2026-06-27 | `Qwen3.5` | [Bugfix][Structured Output][Spec Decode] Constrain bitmask and trim grammar advance at the reasoning boundary |
| [#44573](https://github.com/vllm-project/vllm/pull/44573) | 2026-06-27 | `MLA`, `MoE` | Add DeepSeek-V4  DCP decode support |
| [#44677](https://github.com/vllm-project/vllm/pull/44677) | 2026-06-27 | `MoE` | [Core] DBO ++: Overlap TP all-reduce with compute |
| [#45033](https://github.com/vllm-project/vllm/pull/45033) | 2026-06-27 | `FP4`, `MLA` | [ROCm][Perf][MLA] Add AITER FlashAttention MLA prefill backend (`ROCM_AITER_FA`) |
| [#45222](https://github.com/vllm-project/vllm/pull/45222) | 2026-06-27 | `WideEP` | [Bugfix] MoRIIO toy P/D proxy: honor max_completion_tokens, add /health, fix round-robin off-by-one |
| [#45223](https://github.com/vllm-project/vllm/pull/45223) | 2026-06-27 | `WideEP` | [Bugfix][Core] DP: sync engines_running finish-state every step to prevent multi-node deadlock |
| [#45225](https://github.com/vllm-project/vllm/pull/45225) | 2026-06-27 | `FP4`, `MoE`, `WideEP` | [Bugfix][ROCm][MoE] MoRI: pass num_qp_per_pe/quant_type explicitly; preserve router top-k for finalize |
| [#45226](https://github.com/vllm-project/vllm/pull/45226) | 2026-06-27 | `MoE`, `WideEP` | [ROCm][MoE] Route batched expert layout through flat-reshape wrapper for AITER FP8 |
| [#45227](https://github.com/vllm-project/vllm/pull/45227) | 2026-06-27 | `DeepSeek V4`, `FP4`, `MLA`, `MoE`, `WideEP` | [Bugfix][ROCm] MLA MTP decode: size verification metadata for real qlen/dtype, gate persistent path, fix CUDA-graph padding |
| [#45228](https://github.com/vllm-project/vllm/pull/45228) | 2026-06-27 | `MLA` | [Core][KV-transfer] MoRIIO: multi-node TP prefill→decode dispatch via published host list |
| [#45229](https://github.com/vllm-project/vllm/pull/45229) | 2026-06-27 | `MLA`, `MoE` | [V1][Spec Decode] Relaxed acceptance for thinking-phase tokens (port of #22238) |
| [#45230](https://github.com/vllm-project/vllm/pull/45230) | 2026-06-27 | `MoE` | [Bugfix][KV-transfer] MoRIIO: READ-mode stability fixes (completion IDs, DP routing, drain, keepalive) |
| [#46182](https://github.com/vllm-project/vllm/pull/46182) | 2026-06-27 | `MLA` | [Feat][1/N] CuTeDSL warmup infrastructure, FA4 MLA |
| [#46184](https://github.com/vllm-project/vllm/pull/46184) | 2026-06-27 | `MoE` | [ROCm][Perf] Use flydsl moe with Minimax-M3 mxfp8 weights on gfx950 and implemented moe-backend selection |
| [#46267](https://github.com/vllm-project/vllm/pull/46267) | 2026-06-27 | `FP4`, `MoE` | [ROCm][Test] Resurrect test_rocm_mxfp4_moe_oracle |
| [#46474](https://github.com/vllm-project/vllm/pull/46474) | 2026-06-27 | `FP4`, `MoE` | [ROCm][Perf] Fused shared expert for Minimax M3 |
| [#46724](https://github.com/vllm-project/vllm/pull/46724) | 2026-06-27 | `FP4`, `NVFP4` | [Attention] Occupancy-gated 3D segmented decode for multi-query (diffusion-LM canvas) over long KV |
| [#46786](https://github.com/vllm-project/vllm/pull/46786) | 2026-06-27 | `GLM-5`, `GLM-5.2` | [Model Runner V2][Spec Decode] Handle tuple hidden states from MTP draft models |
| [#46883](https://github.com/vllm-project/vllm/pull/46883) | 2026-06-27 | `FP4`, `NVFP4`, `Qwen3.6` | [Core] Use FlashInfer workspace sizing helper |
| [#46901](https://github.com/vllm-project/vllm/pull/46901) | 2026-06-27 | `MoE` | [MoE] [MoE Refactor] Migrate int8 w4a8int8 oracle 37753 |
| [#46907](https://github.com/vllm-project/vllm/pull/46907) | 2026-06-27 | `MoE` | [CPU][Bugfix] Build cpu_fused_moe on Apple Silicon |
| [#46908](https://github.com/vllm-project/vllm/pull/46908) | 2026-06-27 | `MLA` | [Bugfix] Handle list slot mappings in attention context |
