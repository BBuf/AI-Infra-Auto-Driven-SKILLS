# SGLang Kimi K2 / K2 Thinking / K2.5 Support and Optimization Timeline

This document is based on the latest SGLang `origin/main` snapshot `47c4b3825`, plus patch-level reading of the related merged, open, and closed PRs. It covers the main line originally represented by the `sglang-kimi-k2-k25-optimization` skill and adds the newer Kimi K2.5 DeepEP, W4AFP8, AMD MXFP4, and related tracks.

The short conclusion is: as of `47c4b3825`, Kimi K2 and Kimi K2 Thinking have mainline support for regular MoE routing, Marlin W4A16 MoE, EP, and PCG. Kimi K2.5 has a dedicated multimodal wrapper and runtime plumbing for PP, DP ViT, Eagle3, PD disaggregation, and EPLB. The Kimi K2 Thinking `DeepEP + int4/Marlin` PR `#13789` was closed without merging; the active DeepEP direction is Kimi K2.5 W4A16 low-latency DeepEP in `#22496`.

## 1. Chronological Overview

| Created    |     PR | State  | Track            | Code Area                                       | Effect                                                                                  |
| ---------- | -----: | ------ | ---------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| 2025-07-14 |  #8021 | merged | Kimi K2          | `fused_moe_triton/configs`                      | Added H20-3e FP8 MoE tuning config.                                                     |
| 2025-07-14 |  #8013 | merged | Kimi K2          | `sgl-kernel/csrc/gemm/dsv3_router_gemm_*`       | Made `dsv3_router_gemm` support 384 experts.                                            |
| 2025-07-15 |  #8047 | merged | Kimi K2          | `fused_moe_triton/configs`                      | Added H20 FP8 MoE tuning config.                                                        |
| 2025-07-20 |  #8176 | merged | Kimi K2          | `fused_moe_triton/configs`                      | Added H200 TP16 Kimi K2 MoE config.                                                     |
| 2025-07-20 |  #8178 | merged | Kimi K2          | `fused_moe_triton/configs`                      | Added B200 TP16 Kimi K2 MoE config.                                                     |
| 2025-07-20 |  #8183 | merged | Kimi K2          | `fused_moe_triton/configs`                      | Corrected the H200 Kimi K2 MoE expert/N shape.                                          |
| 2025-08-09 |  #9010 | merged | Kimi K2          | `fused_moe_triton/configs/triton_3_4_0`         | Added B200 FP8 MoE config for the newer Triton path.                                    |
| 2025-11-12 | #13150 | merged | Kimi K2 Thinking | `python/sglang/srt/layers/moe/topk.py`          | Added a torch.compile optimized biased top-k path for 384 experts and one expert group. |
| 2025-11-14 | #13287 | merged | Kimi K2 Thinking | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` | Added the Kimi K2 dedicated fused gate CUDA op.                                         |
| 2025-11-15 | #13332 | merged | Kimi K2 Thinking | `topk.py`                                       | Routed Kimi K2 Thinking through the fused gate.                                         |
| 2025-11-16 | #13374 | merged | Kimi K2 Thinking | `kimi_k2_moe_fused_gate.cu`                     | Optimized fused gate vectorized loads and the small-token path.                         |
| 2025-11-19 | #13587 | merged | Kimi K2 Thinking | `moe_align_block_size.py`                       | Removed a useless padding kernel from `sgl_moe_align_block_size`.                       |
| 2025-11-19 | #13596 | merged | Kimi K2 Thinking | `fused_marlin_moe.py`, quant method             | Avoided useless `torch.zeros_` in fake-EP Marlin MoE cases.                             |
| 2025-11-21 | #13725 | merged | Kimi K2 Thinking | `compressed_tensors_moe.py`                     | Added EP support for Kimi K2 Thinking compressed-tensors MoE.                           |
| 2025-11-23 | #13789 | closed | Kimi K2 Thinking | DeepEP + Marlin path                            | Tried to support K2 Thinking DeepEP, but closed after illegal memory access.            |
| 2025-12-14 | #15100 | merged | Kimi K2 Thinking | `fused_marlin_moe.py`, MoE runner               | Made fused Marlin MoE support piecewise CUDA graph.                                     |
| 2025-12-17 | #15306 | merged | Kimi K2 Thinking | `kimi_k2_moe_fused_gate.cu`                     | Fixed a warp illegal instruction under PCG.                                             |
| 2025-12-18 | #15347 | merged | Kimi K2 Thinking | `topk.py`                                       | Preferred FlashInfer `fused_topk_deepseek` over the Kimi fused gate where valid.        |
| 2026-01-19 | #17325 | merged | Kimi K2 Thinking | `topk.py`                                       | Fixed kernel selection conditions in biased grouped top-k.                              |
| 2026-01-27 | #17789 | merged | Kimi K2.5        | `models/kimi_k25.py`, processor, parser         | Added Kimi K2.5 multimodal model support.                                               |
| 2026-01-30 | #17991 | merged | Kimi K2.5        | `vision.py`, `kimi_k25.py`                      | Fixed double reduce in VLM DP attention.                                                |
| 2026-02-01 | #18064 | merged | Kimi K2.5        | `scheduler.py`                                  | Fixed MoE GEMM config initialization under the K2.5 wrapper.                            |
| 2026-02-06 | #18370 | merged | Kimi K2.5        | `modelopt_quant.py`, `kimi_k25.py`              | Fixed NVFP4 weight mapping and the exclude list.                                        |
| 2026-02-08 | #18440 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Stored the missing `quant_config`.                                                      |
| 2026-02-08 | #18434 | merged | Kimi K2.5        | `deepseek_v2.py`, `kimi_k25.py`                 | Added pipeline parallel support.                                                        |
| 2026-02-12 | #18689 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Added DP ViT encoder support.                                                           |
| 2026-02-23 | #19181 | merged | Kimi K2/K2.5     | `python/sglang/jit_kernel/moe_wna16_marlin.py`  | Migrated the Marlin MoE kernel from AOT to JIT.                                         |
| 2026-02-24 | #19228 | merged | Kimi K2.5        | AMD tuning, `fused_moe_triton_config.py`        | Tuned fused MoE config for K2.5 int4 W4A16 on AMD.                                      |
| 2026-03-02 | #19689 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Added Eagle3 capture and embed/head interfaces.                                         |
| 2026-03-02 | #19703 | open   | Kimi K2 Thinking | `jit_kernel` fused gate                         | Migrates `kimi_k2_moe_fused_gate` to JIT; not merged yet.                               |
| 2026-03-05 | #19959 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Exposed PP layer ranges for PD disaggregation.                                          |
| 2026-03-17 | #20747 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Fixed the K2.5 wrapper surface for piecewise CUDA graph.                                |
| 2026-03-20 | #21004 | merged | Kimi K2.5        | `kimi_k25.py`                                   | Added the routed expert weight interface needed by EPLB rebalance.                      |
| 2026-03-25 | #21391 | merged | Kimi K2.5        | `llama_eagle3.py`, test                         | Fixed the DP attention + speculative decoding multimodal launch crash.                  |
| 2026-03-31 | #21741 | open   | Kimi K2.5        | W4AFP8 MoE                                      | Adds generic compressed-tensors W4AFP8 MoE support.                                     |
| 2026-04-06 | #22208 | open   | Kimi K2.5        | AMD Triton config                               | Tunes gfx950 small-M decode fused MoE.                                                  |
| 2026-04-10 | #22488 | open   | Kimi K2 Thinking | JIT fused gate                                  | Generalizes the Kimi2 ungrouped fused gate to GLM-5 256 experts.                        |
| 2026-04-10 | #22496 | open   | Kimi K2.5        | `deepep_moe_wna16_marlin_direct.py`, etc.       | Adds the K2.5 W4A16 DeepEP low-latency direct Marlin path.                              |
| 2026-04-14 | #22806 | open   | Kimi K2.5        | `quantization/w4afp8.py`                        | Adds `KimiW4AFp8Config` for loading K2.5 W4AFP8.                                        |
| 2026-04-16 | #22964 | open   | Kimi K2.5        | `KimiGPUProcessorWrapper`                       | Fixes CPU processor output keys to match the GPU path.                                  |
| 2026-04-19 | #23186 | open   | Kimi K2.5        | AMD MLA attention                               | Adds fused q/k RMSNorm BF16 for `amd/Kimi-K2.5-MXFP4`.                                  |

## 2. Kimi K2 Stage One: 384 Experts and MoE Tuning

The first Kimi K2 integration problem was not a large model wrapper. The main issue was that the DeepSeek-V3-style MoE infrastructure was more naturally shaped around 256 experts, while Kimi K2 needs 384 experts and device-specific fused MoE tuning configs for H20, H20-3e, H200, and B200.

`#8013` is the central code PR in this stage. It expands `dsv3_router_gemm` from a single 256-expert shape to both 256 and 384:

- Adds constants such as `DEFAULT_NUM_EXPERTS = 256`, `KIMI_K2_NUM_EXPERTS = 384`, and `DEFAULT_HIDDEN_DIM = 7168` in the `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` family.
- Dispatches at runtime based on `mat_b.size(0)` and uses `TORCH_CHECK` to only allow 256 or 384 experts, avoiding silent dispatch to the wrong template.
- Instantiates 384-expert templates for token counts `1..16` and both `float` and `bfloat16` outputs.
- Extends `bench_dsv3_router_gemm.py` and tests to cover `num_experts in [256, 384]`, so the Kimi K2 path is benchmarked and tested instead of merely compiling.

`#8021`, `#8047`, `#8176`, `#8178`, `#8183`, and `#9010` cover the device-specific tuning config side. They add or correct JSON files under:

- `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/`
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/`

The file names encode the key shape information, such as `E=384` or `E=385`, `N=128/256`, `dtype=fp8_w8a8`, `block_shape=[128, 128]`, and `device_name=NVIDIA_H20/H20-3e/H200/B200`. The `E=385` configs reflect real routed/shared-expert shape variants. Later scheduling logic matches these JSON files using model config, quant config, device name, and Triton version.

## 3. Kimi K2 Thinking: Top-k Routing, Fused Gate, and Marlin MoE

`#13150` first optimized Kimi K2 Thinking biased top-k. The characteristic routing shape is `num_experts == 384`, `num_expert_group == 1`, and small `topk`. The generic grouped top-k path still carried group masking and group score logic. The PR added `kimi_k2_biased_topk_impl` in `topk.py`:

- Computes `scores.sigmoid() + correction_bias` directly.
- Runs `torch.topk` over all 384 experts to get top-k expert ids.
- Uses `torch.gather` to recover the original sigmoid weights.
- Applies optional renormalization and routed scaling.
- Maps logical expert ids to physical expert ids if a logical-to-physical expert map exists.
- Filters padding token masks.
- Uses `@torch.compile` to keep this dedicated path out of Python-level generic branching during decode.

`#13287` lowered that routing path into a CUDA op, `sgl_kernel::kimi_k2_moe_fused_gate`. The kernel is specialized for Kimi K2 Thinking:

- `NUM_EXPERTS = 384`.
- `topk = 6`.
- `WARPS_PER_CTA = 6`.
- Initial `VPT = 12`, fusing sigmoid, bias add, top-k, renormalization, and scaling per token.
- Separate small-token and large-token launch strategies.
- Python wrapper, benchmark, and test coverage, with `kimi_k2_biased_topk_impl` as the correctness baseline.

`#13332` wired this kernel into `biased_grouped_topk_gpu`: when the device is CUDA, the expert count is 384, and there is only one expert group, SGLang uses `kimi_k2_moe_fused_gate`; otherwise it falls back to the generic paths.

`#13374` then optimized the fused gate kernel:

- Narrows the score and correction-bias path to `float32`, reducing dtype-generalization overhead.
- Adds `VEC_SIZE = 4` `float4` vectorized loads.
- Uses 384 threads per token in the small-token kernel, one thread per expert.
- Stores intermediate top-k state in shared memory, including `selected_experts[8]`, `warp_experts`, and `warp_maxs`.
- Reduces `__syncthreads()` and keeps top-k selection, renormalization, and output writes inside a tighter kernel.

`#13587` removes a useless padding kernel from `sgl_moe_align_block_size`. It is small but meaningful in MoE decode, where extra launches and unnecessary padding sit directly on the critical path.

`#13596` added the SGLang-side `fused_marlin_moe` wrapper for Kimi K2 Thinking W4A16 Marlin MoE. The important details are:

- Uses `moe_align_block_size` to align token/expert pairs.
- Selects `block_size_m` from `[8, 16, 32, 48, 64]`.
- Calls `moe_wna16_marlin_gemm` for the gate/up projection.
- Runs `silu_and_mul` for activation fusion.
- Calls `moe_wna16_marlin_gemm` again for the down projection.
- Runs `moe_sum_reduce` to merge top-k expert outputs.
- Previously, the fake-EP path zeroed an intermediate cache unconditionally; the PR narrows `torch.zeros_` to the real `expert_map is not None` case, avoiding zero-fill cost when there is no real EP expert map.

In current main, this Marlin MoE path has been moved by `#19181` to call the JIT kernel from `python/sglang/jit_kernel/moe_wna16_marlin.py` instead of directly depending on an AOT sgl-kernel symbol.

## 4. Kimi K2 Thinking: EP, PCG, and Routing Kernel Selection

`#13725` added Expert Parallelism support to the compressed-tensors MoE path for Kimi K2 Thinking. The key change is that the compressed-tensors quant method no longer treats EP information as fake metadata; it passes the real `expert_map`, top-k ids, weights, and runner metadata into Marlin MoE.

`#15100` made fused Marlin MoE support piecewise CUDA graph. PCG is sensitive to dynamic shapes, temporary tensors, custom ops, and fake ops. This PR adjusted boundaries across `fused_marlin_moe.py`, the MoE runner, and the quant method so the path can be captured by segmented CUDA graphs.

`#15306` is the follow-up PCG fix. It fixed a warp illegal instruction in `kimi_k2_moe_fused_gate.cu`. The issue appeared after the fused gate became captured by PCG and token shapes or expert-selection state became more stable, indicating insufficient protection around invalid expert or warp selection state inside the kernel.

`#15347` changed the routing priority for Kimi K2 Thinking. When FlashInfer `fused_topk_deepseek` is valid, SGLang now prefers it over the Kimi-specific `moe_fused_gate`. Current main roughly orders `biased_grouped_topk_gpu` like this:

1. If `fused_topk_deepseek` is available, the device is CUDA, the expert count is a power of two, and group/top-k constraints are satisfied, use the FlashInfer fused top-k path. For `num_expert_group == 1`, current conditions allow `num_experts <= 384`.
2. Otherwise try the generic `moe_fused_gate`.
3. Then try the AITER path.
4. Then fall back to the Kimi 384-expert `kimi_k2_moe_fused_gate`.
5. Finally fall back to the torch.compile generic biased top-k.

`#17325` fixed the kernel selection conditions above, avoiding selection of a faster but invalid path when shape or group constraints are not met. After this PR, the Kimi fused gate still exists, but it is a fallback rather than the first-priority route.

`#19703` remains open and aims to migrate `kimi_k2_moe_fused_gate` from AOT `sgl-kernel` into `python/sglang/jit_kernel`. `#22488` goes further by generalizing the Kimi2 ungrouped fused gate to GLM-5's 256-expert shape. Together they suggest this dedicated routing kernel is moving toward a JIT-managed variable-expert implementation rather than a Kimi-only AOT file.

## 5. Kimi K2 Thinking DeepEP Status: Not Mainline-Ready

`#13789` is titled `[DeepEP Support] Support kimi-k2-thinking deepep`, but it is closed and was not merged. The attempted launch command included:

```bash
SGLANG_DEEPEP_BF16_DISPATCH=1 python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2-Thinking \
  --tp 8 --ep 4 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2
```

The patch and discussion exposed an illegal memory access around `DeepEPMoE.forward_marlin_moe -> quant_method.apply_deepep_normal -> fused_marlin_moe`. In other words, Kimi K2 Thinking `DeepEP + int4/Marlin` should not be treated as mainline-ready merely because Marlin MoE was JIT-migrated or regular EP support exists.

`#22496` is different. It is the new open Kimi K2.5 W4A16 DeepEP low-latency route. Instead of reusing the full regular `fused_marlin_moe` layout, it adds:

- `deepep_moe_wna16_marlin_direct.py`
- `mask_silu_and_mul.py` / `.cuh`
- `marlin_direct_template.h`
- `kernel_direct.h`
- `marlin_tma_utils.h`
- modifications to `moe_wna16_marlin.cuh`, `ep_moe/layer.py`, `token_dispatcher/deepep.py`, and compressed-tensors quant methods

The core idea is to add `apply_deepep_normal` and `apply_deepep_ll` to the compressed-tensors quant method. `apply_deepep_ll` requires BF16 dispatch and handles DeepEP's three-dimensional `[E, M, K]` hidden states. It builds and caches prefix/layout buffers, compacts active hidden states, runs direct Marlin gate/up and down projections with `mask_silu_and_mul` in between, then expands results back to the DeepEP layout. It also adds `DEEPEP_LL_PROFILE_COMPUTE` profiling logs. This PR targets Kimi K2.5 W4A16 DeepEP low latency, not the closed K2 Thinking DeepEP attempt.

## 6. Kimi K2.5: Multimodal Wrapper and Runtime Interfaces

`#17789` is the starting point for Kimi K2.5 support. It added `python/sglang/srt/models/kimi_k25.py` with the following structure:

- The language model reuses `DeepseekV3ForCausalLM`.
- The vision tower uses MoonViT3d.
- A projector maps vision features into the language hidden size.
- `hf_to_sglang_mapper` maps HF names such as `language_model.layers.` to SGLang internal names such as `language_model.model.layers.`.
- Processor and parser hooks support Kimi K2.5 multimodal input, reasoning parsing, and tool-call parsing.
- `pad_input_ids` handles image token padding.
- `forward` uses `general_mm_embed_routine` to merge image embeddings and text embeddings.

Many later K2.5 PRs make this wrapper transparent to the rest of SGLang. A lot of generic runtime logic expects the model object itself to be a CausalLM, but K2.5 adds a multimodal wrapper, so the wrapper must re-expose the underlying language model's interfaces.

`#18440` stores `self.quant_config`; without it, ModelOpt/NVFP4 logic cannot read quantization config from the wrapper. `#18370` then fixes NVFP4 weight-name mapping and the exclude list so quantization code understands names under the `language_model` wrapper. `#18064` fixes scheduler MoE GEMM config initialization by reading MoE shapes from K2.5 `text_config`.

`#18434` adds PP support. The K2.5 wrapper can pass `pp_proxy_tensors` into the underlying `DeepseekV3ForCausalLM` and handle pipeline-stage forward outputs. `#19959` further exposes `start_layer` and `end_layer`, which are needed by PD disaggregation and other logic that must know the layer range covered by the current PP shard.

`#18689` adds DP ViT. In current main, `KimiK25ForConditionalGeneration` reads `get_global_server_args().mm_enable_dp_encoder` and passes `use_data_parallel` to the vision tower. In `get_image_feature`, if the DP encoder is enabled, it calls `run_dp_sharded_mrope_vision_model`, allowing the multimodal encoder to run sharded across DP.

`#17991` fixes double reduce in VLM DP attention, avoiding a second upper-level reduce after the visual-side DP attention has already reduced. `#21391` fixes a DP attention + speculative decoding launch crash: when Eagle/spec decode extends a multimodal batch, it should reuse `forward_batch.mm_input_embeds` and only append the final token embedding, instead of re-embedding the full multimodal prefix.

`#19689` adds K2.5 Eagle3 interfaces: `set_eagle3_layers_to_capture`, `get_embed_and_head`, and `set_embed_and_head`. `#20747` sets `self.model = self.language_model.model` in the wrapper, fixing piecewise CUDA graph assumptions about the underlying model surface.

`#21004` adds the EPLB rebalance interface. In current main, K2.5's `routed_experts_weights_of_layer` property returns the underlying language model's `_routed_experts_weights_of_layer.value`, allowing EPLB to read per-layer routed expert weights across the wrapper.

## 7. Kimi K2.5 Quantization and Platform Optimization

`#19181` migrates the Marlin MoE kernel to JIT. It adds `python/sglang/jit_kernel/moe_wna16_marlin.py`, compiles through `_jit_moe_wna16_marlin_module`, and exports `moe_wna16_marlin_gemm`. The tests cover:

- `m = 1` and `m = 123`.
- `n = 128` and `n = 1024`.
- `fp16` / `bf16`.
- act-order and non-act-order.
- `uint4` / `uint4b8` weight layouts.
- bitwise matching between JIT and the old AOT implementation.

This matters for Kimi because Kimi K2 Thinking and K2.5 W4A16 MoE use Marlin MoE. It is not, however, sufficient to claim DeepEP support: DeepEP still needs token dispatch layout, active-token compaction, expert buffers, and direct Marlin calls to be correct.

`#19228` is the AMD Kimi K2.5 fused MoE tuning PR. It lets config-reading logic pass through K2.5 `text_config`, detect int4 W4A16 group size and block shape from quant config, and generate the correct config filename for `dtype=int4_w4a16`. For int4 packed layout, `N` must be derived from shard intermediate size and then adjusted again for packing.

`#22208` is still open and continues gfx950 small-M decode fused MoE tuning on AMD. `#23186` is another AMD track: in MLA absorb prepare, when AITER is enabled and dtype is BF16, it uses `fused_qk_rmsnorm_bf16` to fuse q_a and kv_a RMSNorm for `amd/Kimi-K2.5-MXFP4`.

`#21741` and `#22806` are the W4AFP8 track. `#21741` adds generic compressed-tensors W4AFP8 MoE support, including FP8 activation scale and CUTLASS W4A8 MoE pieces. `#22806` adds Kimi-specific `KimiW4AFp8Config`:

- Quant method name: `kimi_w4afp8`.
- Parses all important quant config fields.
- Distinguishes `ignored_layers` from `unquantized_layers`: ignored layers skip W4 but may still use FP8, while truly unquantized layers such as `lm_head` stay unquantized.
- Normalizes `model.` prefixes.
- Returns `Fp8LinearMethod` or `UnquantizedLinearMethod` for ordinary `LinearBase`.
- Returns `W4AFp8MoEMethod` for `FusedMoE`.
- Adds expert input scale mapping for HF-standard `gate_proj/down_proj/up_proj`.

`#22964` fixes a processor mismatch. The GPU processor `_call` currently returns `image_grid_thw`, while CPU `_cpu_call` can return `grid_thws` in some paths. The open PR maps the CPU path to `image_grid_thw` as well, avoiding key mismatch during multimodal feature packing.

## 8. Current Main Code Shape

As of `47c4b3825`, the Kimi mainline looks like this:

- In `topk.py`, Kimi K2 Thinking 384-expert routing is a multi-stage selection among FlashInfer `fused_topk_deepseek`, generic `moe_fused_gate`, AITER, Kimi fused gate, and torch.compile generic fallback.
- `fused_marlin_moe.py` uses JIT `moe_wna16_marlin_gemm` and keeps EP zero-fill only under `expert_map is not None`.
- `kimi_k25.py` is the central K2.5 wrapper for the language model, vision tower, projector, processor, DP ViT, PP range, Eagle3, PCG, and EPLB interfaces.
- K2.5 quantization and platform optimization are still moving quickly: NVFP4 has mainline fixes, while W4AFP8, K2.5 W4A16 DeepEP low latency, and AMD MXFP4 fused q/k RMSNorm are still open PR tracks.

So if the Kimi skill is updated again, it should probably split the old “K2/K2.5 optimization” story into two explicit narratives:

1. Kimi K2 Thinking runtime path: 384-expert top-k, fused gate, Marlin MoE, EP, PCG, and the boundary that DeepEP is not mainline-ready yet.
2. Kimi K2.5 multimodal and quantization path: wrapper transparency, PP/DP/Eagle/EPLB, and the latest NVFP4/W4AFP8/AMD/DeepEP-low-latency state.
