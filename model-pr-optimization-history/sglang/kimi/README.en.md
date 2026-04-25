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

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Kimi K2 / K2.5` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-14 | [#8013](https://github.com/sgl-project/sglang/pull/8013) | merged | [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384 | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` |
| 2025-07-14 | [#8021](https://github.com/sgl-project/sglang/pull/8021) | merged | perf: add kimi k2 fused_moe tuning config for h30_3e | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-15 | [#8043](https://github.com/sgl-project/sglang/pull/8043) | merged | feat(function call): complete utility method for KimiK2Detector and enhance documentation | tests/benchmarks | `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py` |
| 2025-07-15 | [#8047](https://github.com/sgl-project/sglang/pull/8047) | merged | H20 tune config for Kimi | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8176](https://github.com/sgl-project/sglang/pull/8176) | merged | feat: add h200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8178](https://github.com/sgl-project/sglang/pull/8178) | merged | feat: add b200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8183](https://github.com/sgl-project/sglang/pull/8183) | merged | feat: add h200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-08-09 | [#9010](https://github.com/sgl-project/sglang/pull/9010) | merged | [perf] add kimi-k2 b200 fused moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-08-25 | [#9606](https://github.com/sgl-project/sglang/pull/9606) | merged | Fix kimi k2 function calling format | tests/benchmarks | `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-09-18 | [#10612](https://github.com/sgl-project/sglang/pull/10612) | merged | Replace the Kimi-K2 generated tool call idx with history tool call count | tests/benchmarks | `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-09-26 | [#10972](https://github.com/sgl-project/sglang/pull/10972) | merged | fix: KimiK2Detector Improve tool call ID parsing with regex | misc | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-11-06 | [#12759](https://github.com/sgl-project/sglang/pull/12759) | merged | [Ascend] support Kimi-K2-Thinking | model wrapper, MoE/router, quantization, scheduler/runtime | `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-12 | [#13150](https://github.com/sgl-project/sglang/pull/13150) | merged | Opt kimi_k2_thinking biased topk module | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-14 | [#13287](https://github.com/sgl-project/sglang/pull/13287) | merged | [opt kimi k2 1 / n] Add kimi k2 moe fused gate | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` |
| 2025-11-15 | [#13332](https://github.com/sgl-project/sglang/pull/13332) | merged | [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-16 | [#13374](https://github.com/sgl-project/sglang/pull/13374) | merged | [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel | MoE/router, kernel | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2025-11-18 | [#13466](https://github.com/sgl-project/sglang/pull/13466) | merged | [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking) | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-19 | [#13587](https://github.com/sgl-project/sglang/pull/13587) | merged | [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size | MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2025-11-19 | [#13596](https://github.com/sgl-project/sglang/pull/13596) | merged | [kimi k2 thinking] Avoid useless torch.zeros_ | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-11-21 | [#13725](https://github.com/sgl-project/sglang/pull/13725) | merged | Add Expert Parallelism (EP) support for kimi-k2-thinking | MoE/router, quantization | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-11-23 | [#13789](https://github.com/sgl-project/sglang/pull/13789) | closed | [DeepEP Support] Support kimi-k2-thinking deepep | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu` |
| 2025-12-14 | [#15100](https://github.com/sgl-project/sglang/pull/15100) | merged | Support piecewise cuda graph for fused marlin moe | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` |
| 2025-12-17 | [#15306](https://github.com/sgl-project/sglang/pull/15306) | merged | Fix warp illegal instruction in kimi k2 thinking PCG | MoE/router, kernel | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | MoE/router, kernel, tests/benchmarks | `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2026-01-19 | [#17325](https://github.com/sgl-project/sglang/pull/17325) | merged | Fix kernel selection in biased_grouped_topk_gpu | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2026-01-21 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | quantization, tests/benchmarks | `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` |
| 2026-01-23 | [#17656](https://github.com/sgl-project/sglang/pull/17656) | merged | [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases | tests/benchmarks | `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml` |
| 2026-01-27 | [#17789](https://github.com/sgl-project/sglang/pull/17789) | merged | Support Kimi-K2.5 model | model wrapper, attention/backend, multimodal/processor, docs/config | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-01-30 | [#17991](https://github.com/sgl-project/sglang/pull/17991) | merged | Fix: Avoid Double Reduce in VLM DP Attention | model wrapper, attention/backend, multimodal/processor, tests/benchmarks | `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-01 | [#18064](https://github.com/sgl-project/sglang/pull/18064) | merged | fix kimi k2.5's moe gemm config init | scheduler/runtime | `python/sglang/srt/managers/scheduler.py` |
| 2026-02-04 | [#18269](https://github.com/sgl-project/sglang/pull/18269) | merged | [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test | model wrapper, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml` |
| 2026-02-06 | [#18370](https://github.com/sgl-project/sglang/pull/18370) | merged | [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-08 | [#18434](https://github.com/sgl-project/sglang/pull/18434) | merged | [Fix] Kimi K2.5 support pp | model wrapper | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-08 | [#18440](https://github.com/sgl-project/sglang/pull/18440) | merged | [Kimi-K2.5] Fix missing `quant_config` in `KimiK25` | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-12 | [#18689](https://github.com/sgl-project/sglang/pull/18689) | merged | Add DP ViT support for Kimi K2.5 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-21 | [#19120](https://github.com/sgl-project/sglang/pull/19120) | merged | fix KimiK2Detector regex patterns with re.DOTALL | misc | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-02-23 | [#19181](https://github.com/sgl-project/sglang/pull/19181) | merged | [Kernel Slimming] Migrate marlin moe kernel to JIT | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` |
| 2026-02-24 | [#19228](https://github.com/sgl-project/sglang/pull/19228) | merged | [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning | MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` |
| 2026-02-25 | [#19331](https://github.com/sgl-project/sglang/pull/19331) | merged | [NPU] support Kimi-K2.5 on NPU | model wrapper, MoE/router, quantization | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-02-28 | [#19552](https://github.com/sgl-project/sglang/pull/19552) | merged | [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection | tests/benchmarks | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-03-02 | [#19689](https://github.com/sgl-project/sglang/pull/19689) | merged | feat: support Kimi K2.5 for Eagle3 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-02 | [#19703](https://github.com/sgl-project/sglang/pull/19703) | open | [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT | MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` |
| 2026-03-03 | [#19802](https://github.com/sgl-project/sglang/pull/19802) | merged | [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2 | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py` |
| 2026-03-05 | [#19959](https://github.com/sgl-project/sglang/pull/19959) | merged | Fix Kimi K2.5 PP layer range exposure for PD disaggregation | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-17 | [#20747](https://github.com/sgl-project/sglang/pull/20747) | merged | fix piecewise cuda graph support for Kimi-K2.5 model | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-20 | [#21004](https://github.com/sgl-project/sglang/pull/21004) | merged | [Fix] Add EPLB rebalance support for Kimi K2.5 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-25 | [#21391](https://github.com/sgl-project/sglang/pull/21391) | merged | Fix Kimi K2.5 dp attention+ spec decoding launch crash | model wrapper, tests/benchmarks | `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-03-31 | [#21741](https://github.com/sgl-project/sglang/pull/21741) | open | [1/N] feat: support compressed-tensors w4afp8 MoE | MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` |
| 2026-04-06 | [#22208](https://github.com/sgl-project/sglang/pull/22208) | open | [AMD] Optimize fused MoE kernel config for small-M decode on gfx950 | MoE/router, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2026-04-10 | [#22488](https://github.com/sgl-project/sglang/pull/22488) | open | Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation | MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py` |
| 2026-04-10 | [#22496](https://github.com/sgl-project/sglang/pull/22496) | open | [Feature] kimi k25 w4a16 support deepep low latency | MoE/router, quantization, kernel | `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` |
| 2026-04-14 | [#22806](https://github.com/sgl-project/sglang/pull/22806) | open | feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-04-16 | [#22964](https://github.com/sgl-project/sglang/pull/22964) | open | [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output | multimodal/processor | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-19 | [#23186](https://github.com/sgl-project/sglang/pull/23186) | merged | [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |

### File-level PR diff reading notes

### PR #8013 - [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384

- Link: https://github.com/sgl-project/sglang/pull/8013
- Status/date: `merged`, created 2025-07-14, merged 2025-08-01; author `panpan0000`.
- Diff scope read: `5` files, `+188/-30`; areas: MoE/router, kernel, tests/benchmarks; keywords: expert, router, cuda, benchmark, quant, test.
- Code diff details:
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunks: #include "cuda_runtime.h"; void dsv3_router_gemm(; symbols: int, int, int, int
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunks: void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a, T const*; template void invokeRouterGemmBf16Output<__nv_bfloat16, 15, 256, 7168>(; symbols: void, void, void, void
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunks: void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,; template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 256, 7168>(; symbols: void, void, void, void
  - `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunks: x_vals=[i + 1 for i in range(16)],; def tflops(t_ms):; symbols: benchmark_bf16_output, runner, runner, tflops
  - `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2 (4 lines); hunks: @pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)]); symbols: test_dsv3_router_gemm, test_dsv3_router_gemm
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`; keywords observed in patches: expert, router, cuda, benchmark, quant, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8021 - perf: add kimi k2 fused_moe tuning config for h30_3e

- Link: https://github.com/sgl-project/sglang/pull/8021
- Status/date: `merged`, created 2025-07-14, merged 2025-07-14; author `GaoYusong`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8043 - feat(function call): complete utility method for KimiK2Detector and enhance documentation

- Link: https://github.com/sgl-project/sglang/pull/8043
- Status/date: `merged`, created 2025-07-15, merged 2025-07-24; author `CatherineSue`.
- Diff scope read: `8` files, `+205/-56`; areas: tests/benchmarks; keywords: spec, kv, config, doc, test.
- Code diff details:
  - `python/sglang/srt/function_call/base_format_detector.py` modified +70/-12 (82 lines); hunks: class BaseFormatDetector(ABC):; def parse_streaming_increment(; symbols: BaseFormatDetector, providing, __init__, parse_base_json
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunks: class KimiK2Detector(BaseFormatDetector):; def parse_streaming_increment(; symbols: KimiK2Detector, __init__, parse_streaming_increment, parse_streaming_increment
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +25/-10 (35 lines); hunks: class DeepSeekV3Detector(BaseFormatDetector):; def parse_streaming_increment(; symbols: DeepSeekV3Detector, __init__, parse_streaming_increment, parse_streaming_increment
  - `test/srt/test_function_call_parser.py` modified +28/-0 (28 lines); hunks: def setUp(self):; def test_deepseekv3_detector_ebnf(self):; symbols: setUp, test_pythonic_detector_ebnf, test_deepseekv3_detector_ebnf, test_kimik2_detector_ebnf
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +12/-9 (21 lines); hunks: class PythonicDetector(BaseFormatDetector):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; symbols: PythonicDetector, __init__, detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py`; keywords observed in patches: spec, kv, config, doc, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8047 - H20 tune config for Kimi

- Link: https://github.com/sgl-project/sglang/pull/8047
- Status/date: `merged`, created 2025-07-15, merged 2025-07-15; author `artetaout`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8176 - feat: add h200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8176
- Status/date: `merged`, created 2025-07-20, merged 2025-07-20; author `zhyncs`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8178 - feat: add b200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8178
- Status/date: `merged`, created 2025-07-20, merged 2025-07-20; author `zhyncs`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #8183 - feat: add h200 tp 16 kimi k2 moe config

- Link: https://github.com/sgl-project/sglang/pull/8183
- Status/date: `merged`, created 2025-07-20, merged 2025-07-20; author `Qiaolin-Yu`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9010 - [perf] add kimi-k2 b200 fused moe config

- Link: https://github.com/sgl-project/sglang/pull/9010
- Status/date: `merged`, created 2025-08-09, merged 2025-08-09; author `Alcanderian`.
- Diff scope read: `1` files, `+146/-0`; areas: MoE/router, quantization, kernel, docs/config; keywords: config, fp8, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunks: +{
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; keywords observed in patches: config, fp8, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9606 - Fix kimi k2 function calling format

- Link: https://github.com/sgl-project/sglang/pull/9606
- Status/date: `merged`, created 2025-08-25, merged 2025-08-26; author `XiaotongJiang`.
- Diff scope read: `2` files, `+117/-9`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunks: python -m unittest discover -s tests -p "test_*unit.py" -v; async def test_unstreamed_tool_args_no_parser_data(self):; symbols: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunks: def _process_tool_calls(; async def _process_tool_call_stream(; symbols: _process_tool_calls, _process_tool_call_stream
- Optimization/support interpretation: The concrete diff surface is `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10612 - Replace the Kimi-K2 generated tool call idx with history tool call count

- Link: https://github.com/sgl-project/sglang/pull/10612
- Status/date: `merged`, created 2025-09-18, merged 2025-09-26; author `eraser00`.
- Diff scope read: `2` files, `+226/-15`; areas: tests/benchmarks; keywords: test.
- Code diff details:
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunks: async def collect_first_tool_chunk():; symbols: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history, collect_first_tool_chunk
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunks: process_hidden_states_from_ret,; def _build_chat_response(; symbols: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls
- Optimization/support interpretation: The concrete diff surface is `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; keywords observed in patches: test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10972 - fix: KimiK2Detector Improve tool call ID parsing with regex

- Link: https://github.com/sgl-project/sglang/pull/10972
- Status/date: `merged`, created 2025-09-26, merged 2025-10-01; author `JustinTong0323`.
- Diff scope read: `1` files, `+17/-4`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunks: def __init__(self):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; symbols: __init__, has_tool_call, detect_and_parse, parse_streaming_increment
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/kimik2_detector.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/kimik2_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12759 - [Ascend] support Kimi-K2-Thinking

- Link: https://github.com/sgl-project/sglang/pull/12759
- Status/date: `merged`, created 2025-11-06, merged 2025-11-22; author `zhuyijie88`.
- Diff scope read: `4` files, `+549/-170`; areas: model wrapper, MoE/router, quantization, scheduler/runtime; keywords: expert, config, moe, quant, attention, cache, cuda, deepep, fp8, kv.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunks: from __future__ import annotations; QuantizationConfig,; symbols: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunks: _is_fp8_fnuz = is_fp8_fnuz(); def forward_npu(; symbols: forward_npu, _forward_normal, _forward_ll, _forward_ll
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; symbols: load_weights, load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunks: def add_chunked_prefix_cache_attention_backend(backend_name):; symbols: add_chunked_prefix_cache_attention_backend
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`; keywords observed in patches: expert, config, moe, quant, attention, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13150 - Opt kimi_k2_thinking biased topk module

- Link: https://github.com/sgl-project/sglang/pull/13150
- Status/date: `merged`, created 2025-11-12, merged 2025-11-13; author `BBuf`.
- Diff scope read: `1` files, `+71/-14`; areas: MoE/router; keywords: expert, moe, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunks: def grouped_topk_cpu(; def biased_grouped_topk_gpu(; symbols: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: expert, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13287 - [opt kimi k2 1 / n] Add kimi k2 moe fused gate

- Link: https://github.com/sgl-project/sglang/pull/13287
- Status/date: `merged`, created 2025-11-14, merged 2025-11-15; author `BBuf`.
- Diff scope read: `8` files, `+646/-0`; areas: MoE/router, kernel, tests/benchmarks; keywords: moe, topk, cuda, expert, fp8, config, spec, test, benchmark, fp4.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunks: +#include <ATen/cuda/CUDAContext.h>; symbols: int, int, int, int
  - `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunks: +import pytest; symbols: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case
  - `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunks: +import itertools; symbols: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunks: def moe_fused_gate(; symbols: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunks: std::vector<at::Tensor> moe_fused_gate(
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`; keywords observed in patches: moe, topk, cuda, expert, fp8, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13332 - [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate

- Link: https://github.com/sgl-project/sglang/pull/13332
- Status/date: `merged`, created 2025-11-15, merged 2025-11-16; author `BBuf`.
- Diff scope read: `1` files, `+6/-9`; areas: MoE/router; keywords: cuda, expert, moe, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunks: _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip; def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: cuda, expert, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13374 - [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel

- Link: https://github.com/sgl-project/sglang/pull/13374
- Status/date: `merged`, created 2025-11-16, merged 2025-11-18; author `BBuf`.
- Diff scope read: `1` files, `+130/-173`; areas: MoE/router, kernel; keywords: cuda, expert, moe, spec, topk.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunks: #include <ATen/cuda/CUDAContext.h>; static constexpr int SMALL_TOKEN_THRESHOLD = 512;; symbols: int, int, int, int
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; keywords observed in patches: cuda, expert, moe, spec, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13466 - [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)

- Link: https://github.com/sgl-project/sglang/pull/13466
- Status/date: `merged`, created 2025-11-18, merged 2025-11-21; author `b8zhong`.
- Diff scope read: `1` files, `+23/-0`; areas: MoE/router; keywords: cuda, moe, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunks: if _is_cuda:; symbols: _kimi_k2_moe_fused_gate
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: cuda, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13587 - [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size

- Link: https://github.com/sgl-project/sglang/pull/13587
- Status/date: `merged`, created 2025-11-19, merged 2025-11-21; author `BBuf`.
- Diff scope read: `1` files, `+1/-6`; areas: MoE/router, kernel; keywords: benchmark, expert, moe, topk, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunks: def moe_align_block_size(; def moe_align_block_size(; symbols: moe_align_block_size, moe_align_block_size
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; keywords observed in patches: benchmark, expert, moe, topk, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13596 - [kimi k2 thinking] Avoid useless torch.zeros_

- Link: https://github.com/sgl-project/sglang/pull/13596
- Status/date: `merged`, created 2025-11-19, merged 2025-11-21; author `BBuf`.
- Diff scope read: `7` files, `+252/-256`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: marlin, moe, quant, triton, cuda, config, expert, awq, topk, cache.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunks: +import functools; symbols: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake
  - `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232 (232 lines); hunks: -import functools; def moe_wna16_marlin_gemm(; symbols: get_scalar_type, moe_wna16_marlin_gemm, moe_wna16_marlin_gemm, fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunks: from enum import Enum; from aiter.ops.shuffle import shuffle_weight; symbols: apply, apply
  - `python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunks: import torch_npu; def apply(; symbols: apply
  - `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunks: _is_cuda = is_cuda(); def apply(; symbols: apply
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; keywords observed in patches: marlin, moe, quant, triton, cuda, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13725 - Add Expert Parallelism (EP) support for kimi-k2-thinking

- Link: https://github.com/sgl-project/sglang/pull/13725
- Status/date: `merged`, created 2025-11-21, merged 2025-12-07; author `BBuf`.
- Diff scope read: `1` files, `+12/-0`; areas: MoE/router, quantization; keywords: config, expert, marlin, moe, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunks: def apply(; def apply(; symbols: apply, apply
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; keywords observed in patches: config, expert, marlin, moe, quant, router. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13789 - [DeepEP Support] Support kimi-k2-thinking deepep

- Link: https://github.com/sgl-project/sglang/pull/13789
- Status/date: `closed`, created 2025-11-23, closed 2026-04-16; author `BBuf`.
- Diff scope read: `10` files, `+674/-0`; areas: MoE/router, quantization, kernel; keywords: moe, deepep, expert, marlin, quant, topk, config, cuda, triton, fp8.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunks: def fused_marlin_moe_fake(; symbols: fused_marlin_moe_fake, batched_fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunks: def apply(; symbols: apply, apply_deepep_normal, apply_deepep_ll
  - `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +140/-0 (140 lines); hunks: limitations under the License.; void moe_align_block_size(; symbols: int32_t, int32_t, void
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunks: def moe_align_block_size(; symbols: moe_align_block_size, batched_moe_align_block_size
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunks: def run_moe_core(; def run_moe_core(; symbols: run_moe_core, run_moe_core, combine, _is_marlin_moe
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu`; keywords observed in patches: moe, deepep, expert, marlin, quant, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15100 - Support piecewise cuda graph for fused marlin moe

- Link: https://github.com/sgl-project/sglang/pull/15100
- Status/date: `merged`, created 2025-12-14, merged 2025-12-16; author `ispobock`.
- Diff scope read: `5` files, `+55/-36`; areas: MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks; keywords: expert, marlin, moe, quant, triton, config, cuda, topk, fp8, test.
- Code diff details:
  - `test/srt/test_piecewise_cuda_graph.py` modified +35/-0 (35 lines); hunks: def test_mgsm_accuracy(self):; symbols: test_mgsm_accuracy, TestPiecewiseCudaGraphGPTQ, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunks: def _(b_q_weight, perm, size_k, size_n, num_bits):; symbols: _, _, _
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunks: import torch; def fused_marlin_moe(; symbols: fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunks: def fused_experts_none_to_marlin(; def fused_experts_none_to_marlin(; symbols: fused_experts_none_to_marlin, fused_experts_none_to_marlin
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunks: def apply(; def apply(; symbols: apply, apply
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`; keywords observed in patches: expert, marlin, moe, quant, triton, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15306 - Fix warp illegal instruction in kimi k2 thinking PCG

- Link: https://github.com/sgl-project/sglang/pull/15306
- Status/date: `merged`, created 2025-12-17, merged 2025-12-18; author `BBuf`.
- Diff scope read: `1` files, `+12/-4`; areas: MoE/router, kernel; keywords: expert, moe, topk.
- Code diff details:
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunks: __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; __global__ void kimi_k2_moe_fused_gate_kernel(; symbols: void, void
- Optimization/support interpretation: The concrete diff surface is `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; keywords observed in patches: expert, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

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

### PR #17325 - Fix kernel selection in biased_grouped_topk_gpu

- Link: https://github.com/sgl-project/sglang/pull/17325
- Status/date: `merged`, created 2026-01-19, merged 2026-01-19; author `yudian0504`.
- Diff scope read: `1` files, `+0/-1`; areas: MoE/router; keywords: cuda, expert, moe, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunks: def biased_grouped_topk_gpu(; symbols: biased_grouped_topk_gpu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: cuda, expert, moe, topk. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- Link: https://github.com/sgl-project/sglang/pull/17523
- Status/date: `merged`, created 2026-01-21, merged 2026-01-28; author `michaelzhang-ai`.
- Diff scope read: `27` files, `+1540/-43`; areas: quantization, tests/benchmarks; keywords: test, benchmark, config, kv, attention, spec, cache, eagle, topk, cuda.
- Code diff details:
  - `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunks: +"""AMD DeepSeek-V3.2 GSM8K Completion Evaluation Test (8-GPU); symbols: ModelConfig:, __post_init__, get_display_name, get_one_example
  - `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunks: on:; jobs:
  - `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunks: +"""AMD Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).; symbols: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunks: +"""MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0 (142 lines); hunks: +"""AMD DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); symbols: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`; keywords observed in patches: test, benchmark, config, kv, attention, spec. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17656 - [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases

- Link: https://github.com/sgl-project/sglang/pull/17656
- Status/date: `merged`, created 2026-01-23, merged 2026-01-26; author `sogalin`.
- Diff scope read: `2` files, `+97/-2`; areas: tests/benchmarks; keywords: test, attention, cache, config, mla, triton.
- Code diff details:
  - `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunks: +import os; symbols: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k
  - `.github/workflows/pr-test-amd.yml` modified +2/-2 (4 lines); hunks: jobs:; jobs:
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml`; keywords observed in patches: test, attention, cache, config, mla, triton. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17789 - Support Kimi-K2.5 model

- Link: https://github.com/sgl-project/sglang/pull/17789
- Status/date: `merged`, created 2026-01-27, merged 2026-01-27; author `yhyang201`.
- Diff scope read: `11` files, `+1053/-12`; areas: model wrapper, attention/backend, multimodal/processor, docs/config; keywords: config, attention, vision, kv, quant, eagle, flash, lora, mla, processor.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunks: +import logging; symbols: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__
  - `python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunks: +"""; symbols: KimiK25VisionConfig, __init__, KimiK25Config, __init__
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunks: +import re; symbols: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items
  - `python/sglang/srt/parser/reasoning_parser.py` modified +21/-1 (22 lines); hunks: def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):; class ReasoningParser:; symbols: __init__, KimiK2Detector, __init__, Qwen3Detector
  - `python/sglang/srt/configs/model_config.py` modified +11/-9 (20 lines); hunks: def _derive_model_shapes(self):; def _derive_model_shapes(self):; symbols: _derive_model_shapes, _derive_model_shapes, is_generation_model
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; keywords observed in patches: config, attention, vision, kv, quant, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17991 - Fix: Avoid Double Reduce in VLM DP Attention

- Link: https://github.com/sgl-project/sglang/pull/17991
- Status/date: `merged`, created 2026-01-30, merged 2026-02-02; author `yhyang201`.
- Diff scope read: `4` files, `+51/-12`; areas: model wrapper, attention/backend, multimodal/processor, tests/benchmarks; keywords: attention, test, config, cuda, mla, quant, spec, vision.
- Code diff details:
  - `test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunks: import requests; from sglang.test.kits.regex_constrained_kit import TestRegexConstrainedMixin; symbols: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass
  - `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunks: from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm; def __init__(; symbols: __init__, forward, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunks: KIMIV_VT_INFER_MAX_PATCH_NUM = 16328; def __init__(; symbols: apply_rope, __init__, forward
  - `test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunks: def test_gsm8k(self):; symbols: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass
- Optimization/support interpretation: The concrete diff surface is `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: attention, test, config, cuda, mla, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18064 - fix kimi k2.5's moe gemm config init

- Link: https://github.com/sgl-project/sglang/pull/18064
- Status/date: `merged`, created 2026-02-01, merged 2026-02-05; author `cicirori`.
- Diff scope read: `1` files, `+6/-1`; areas: scheduler/runtime; keywords: config, expert, fp4, fp8, moe, scheduler.
- Code diff details:
  - `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunks: def init_tokenizer(self):; symbols: init_tokenizer, init_moe_gemm_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/managers/scheduler.py`; keywords observed in patches: config, expert, fp4, fp8, moe, scheduler. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/managers/scheduler.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18269 - [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test

- Link: https://github.com/sgl-project/sglang/pull/18269
- Status/date: `merged`, created 2026-02-04, merged 2026-02-11; author `michaelzhang-ai`.
- Diff scope read: `4` files, `+250/-10`; areas: model wrapper, tests/benchmarks; keywords: config, test, attention, benchmark, cache, mla, triton, doc, fp4, processor.
- Code diff details:
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunks: +"""MI35x Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU); symbols: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunks: +"""AMD Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU); symbols: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy
  - `.github/workflows/nightly-test-amd.yml` modified +39/-9 (48 lines); hunks: on:; jobs:
  - `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__, get_image_feature
- Optimization/support interpretation: The concrete diff surface is `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`; keywords observed in patches: config, test, attention, benchmark, cache, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18370 - [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list

- Link: https://github.com/sgl-project/sglang/pull/18370
- Status/date: `merged`, created 2026-02-06, merged 2026-02-08; author `mmangkad`.
- Diff scope read: `2` files, `+30/-1`; areas: model wrapper, quantization; keywords: config, fp4, fp8, kv, quant, vision.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +17/-0 (17 lines); hunks: CombineInput,; def get_config_filenames(cls) -> List[str]:; symbols: get_config_filenames, get_scaled_act_names, apply_weight_name_mapper, ModelOptFp8Config
  - `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunks: from sglang.srt.model_loader.weight_utils import default_weight_loader; def vision_tower_forward_auto(; symbols: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: config, fp4, fp8, kv, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18434 - [Fix] Kimi K2.5 support pp

- Link: https://github.com/sgl-project/sglang/pull/18434
- Status/date: `merged`, created 2026-02-08, merged 2026-02-25; author `lw9527`.
- Diff scope read: `2` files, `+14/-13`; areas: model wrapper; keywords: kv.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-12 (23 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunks: MultimodalDataItem,; def forward(; symbols: forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18440 - [Kimi-K2.5] Fix missing `quant_config` in `KimiK25`

- Link: https://github.com/sgl-project/sglang/pull/18440
- Status/date: `merged`, created 2026-02-08, merged 2026-02-08; author `mmangkad`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: config, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: config, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18689 - Add DP ViT support for Kimi K2.5

- Link: https://github.com/sgl-project/sglang/pull/18689
- Status/date: `merged`, created 2026-02-12, merged 2026-02-18; author `yhyang201`.
- Diff scope read: `1` files, `+20/-4`; areas: model wrapper; keywords: config, flash, kv, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunks: from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM; class MoonViT3dPretrainedModel(nn.Module):; symbols: MoonViT3dPretrainedModel, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: config, flash, kv, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19120 - fix KimiK2Detector regex patterns with re.DOTALL

- Link: https://github.com/sgl-project/sglang/pull/19120
- Status/date: `merged`, created 2026-02-21, merged 2026-02-21; author `JustinTong0323`.
- Diff scope read: `1` files, `+5/-3`; areas: misc; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunks: def __init__(self):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; symbols: __init__, detect_and_parse
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/kimik2_detector.py`; keywords observed in patches: n/a. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/kimik2_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19181 - [Kernel Slimming] Migrate marlin moe kernel to JIT

- Link: https://github.com/sgl-project/sglang/pull/19181
- Status/date: `merged`, created 2026-02-23, merged 2026-02-26; author `celve`.
- Diff scope read: `7` files, `+3780/-4`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: expert, marlin, moe, topk, cuda, cache, processor, quant, triton, awq.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunks: +/*; symbols: void, void, auto, auto
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunks: +/*; symbols: void, void, void, auto
  - `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunks: +import itertools; symbols: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm
  - `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0 (251 lines); hunks: +import os; symbols: stack_and_dev, _make_inputs, _run_jit, _run_aot
  - `python/sglang/jit_kernel/moe_wna16_marlin.py` added +172/-0 (172 lines); hunks: +from __future__ import annotations; symbols: _jit_moe_wna16_marlin_module, _or_empty, moe_wna16_marlin_gemm
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`; keywords observed in patches: expert, marlin, moe, topk, cuda, cache. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19228 - [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning

- Link: https://github.com/sgl-project/sglang/pull/19228
- Status/date: `merged`, created 2026-02-24, merged 2026-02-26; author `ZiguanWang`.
- Diff scope read: `5` files, `+486/-23`; areas: MoE/router, kernel, tests/benchmarks, docs/config; keywords: config, moe, triton, benchmark, expert, fp8, quant, cuda, scheduler, spec.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunks: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunks: +{
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunks: ); def benchmark_config(; symbols: benchmark_config, benchmark_config, benchmark_config, get_kernel_wrapper
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunks: ); def benchmark_config(; symbols: benchmark_config, benchmark_config, benchmark_config, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5 (28 lines); hunks: def get_model_config(; def get_model_config(; symbols: get_model_config, get_model_config, get_config_filename, get_config_filename
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`; keywords observed in patches: config, moe, triton, benchmark, expert, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19331 - [NPU] support Kimi-K2.5 on NPU

- Link: https://github.com/sgl-project/sglang/pull/19331
- Status/date: `merged`, created 2026-02-25, merged 2026-02-26; author `khalil2ji3mp6`.
- Diff scope read: `3` files, `+23/-3`; areas: model wrapper, MoE/router, quantization; keywords: quant, config, moe, attention, deepep, expert, topk, vision.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunks: from transformers import activations; from sglang.srt.models.utils import WeightsMapper; symbols: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunks: from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPNormalCombineInput; def forward_npu(; symbols: forward_npu
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +1/-0 (1 lines); hunks: def _add_fused_moe_to_target_scheme_map(self):; symbols: _add_fused_moe_to_target_scheme_map, weight_block_size
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; keywords observed in patches: quant, config, moe, attention, deepep, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19552 - [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection

- Link: https://github.com/sgl-project/sglang/pull/19552
- Status/date: `merged`, created 2026-02-28, merged 2026-03-19; author `AlfredYyong`.
- Diff scope read: `2` files, `+700/-19`; areas: tests/benchmarks; keywords: doc, spec, test.
- Code diff details:
  - `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunks: +import json; symbols: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunks: logger = logging.getLogger(__name__); def __init__(self):; symbols: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call
- Optimization/support interpretation: The concrete diff surface is `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`; keywords observed in patches: doc, spec, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19689 - feat: support Kimi K2.5 for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/19689
- Status/date: `merged`, created 2026-03-02, merged 2026-03-03; author `yefei12`.
- Diff scope read: `1` files, `+29/-0`; areas: model wrapper; keywords: config, eagle, expert, spec.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunks: def get_model_config_for_expert_location(cls, config: KimiK25Config):; symbols: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: config, eagle, expert, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19703 - [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT

- Link: https://github.com/sgl-project/sglang/pull/19703
- Status/date: `open`, created 2026-03-02; author `xingsy97`.
- Diff scope read: `5` files, `+576/-1`; areas: MoE/router, kernel, tests/benchmarks; keywords: moe, topk, cuda, expert, config, test, benchmark, cache, kv, triton.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunks: +#include <sgl_kernel/tensor.h>; symbols: int, int, int, int
  - `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunks: +import itertools; symbols: check_correctness, benchmark, fn, fn
  - `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunks: +import itertools; symbols: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts
  - `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0 (63 lines); hunks: +from __future__ import annotations; symbols: _jit_kimi_k2_moe_fused_gate_module, _kimi_k2_moe_fused_gate_op, kimi_k2_moe_fused_gate
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunks: fused_topk_deepseek = None
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py`; keywords observed in patches: moe, topk, cuda, expert, config, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19802 - [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2

- Link: https://github.com/sgl-project/sglang/pull/19802
- Status/date: `merged`, created 2026-03-03, merged 2026-03-07; author `alisonshao`.
- Diff scope read: `2` files, `+72/-53`; areas: model wrapper, tests/benchmarks; keywords: benchmark, cuda, test, config, eagle, spec, topk.
- Code diff details:
  - `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunks: +import unittest; symbols: TestKimiK25, for, test_kimi_k25
  - `test/registered/8-gpu-models/test_kimi_k2.py` removed +0/-53 (53 lines); hunks: -import unittest; symbols: TestKimiK2, for, test_kimi_k2
- Optimization/support interpretation: The concrete diff surface is `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py`; keywords observed in patches: benchmark, cuda, test, config, eagle, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19959 - Fix Kimi K2.5 PP layer range exposure for PD disaggregation

- Link: https://github.com/sgl-project/sglang/pull/19959
- Status/date: `merged`, created 2026-03-05, merged 2026-03-07; author `yafengio`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunks: def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):; symbols: pad_input_ids, start_layer, end_layer, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20747 - fix piecewise cuda graph support for Kimi-K2.5 model

- Link: https://github.com/sgl-project/sglang/pull/20747
- Status/date: `merged`, created 2026-03-17, merged 2026-03-17; author `yhyang201`.
- Diff scope read: `1` files, `+2/-0`; areas: model wrapper; keywords: vision.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21004 - [Fix] Add EPLB rebalance support for Kimi K2.5

- Link: https://github.com/sgl-project/sglang/pull/21004
- Status/date: `merged`, created 2026-03-20, merged 2026-03-26; author `yafengio`.
- Diff scope read: `1` files, `+4/-0`; areas: model wrapper; keywords: expert.
- Code diff details:
  - `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunks: def start_layer(self) -> int:; symbols: start_layer, end_layer, routed_experts_weights_of_layer, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/kimi_k25.py`; keywords observed in patches: expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21391 - Fix Kimi K2.5 dp attention+ spec decoding launch crash

- Link: https://github.com/sgl-project/sglang/pull/21391
- Status/date: `merged`, created 2026-03-25, merged 2026-03-26; author `Qiaolin-Yu`.
- Diff scope read: `2` files, `+23/-2`; areas: model wrapper, tests/benchmarks; keywords: eagle, attention, config, spec, test, topk.
- Code diff details:
  - `python/sglang/srt/models/llama_eagle3.py` modified +12/-1 (13 lines); hunks: def forward(; symbols: forward
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunks: def test_kimi_k25(self):; def test_kimi_k25(self):; symbols: test_kimi_k25, test_kimi_k25
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py`; keywords observed in patches: eagle, attention, config, spec, test, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21741 - [1/N] feat: support compressed-tensors w4afp8 MoE

- Link: https://github.com/sgl-project/sglang/pull/21741
- Status/date: `open`, created 2026-03-31; author `guzekai01`.
- Diff scope read: `13` files, `+1664/-40`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: fp8, cuda, config, moe, quant, test, triton, expert, topk, benchmark.
- Code diff details:
  - `benchmark/kernels/quantization/bench_w4a8_moe_decode.py` added +887/-0 (887 lines); hunks: +"""Benchmark breakdown for CUTLASS W4A8 MoE decode (TP=8 dimensions).; symbols: init_dist, pack_int4_to_int8, pack_interleave, CUDATimer:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunks: +"""W4AFP8 MoE scheme: INT4 group-quantized weights + FP8 dynamic activations.; symbols: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability
  - `python/sglang/test/test_cutlass_w4a8_moe.py` modified +66/-23 (89 lines); hunks: # SPDX-License-Identifier: Apache-2.0; def test_cutlass_w4a8_moe(M, N, K, E, tp_size, use_ep_moe, topk, group_size, dty; symbols: _init_single_gpu_moe_parallel, pack_int4_values_to_int8, test_cutlass_w4a8_moe, test_cutlass_w4a8_moe
  - `python/sglang/jit_kernel/csrc/gemm/per_tensor_absmax_fp8.cuh` added +86/-0 (86 lines); hunks: +#include <sgl_kernel/tensor.h> // For TensorMatcher, SymbolicSize, SymbolicDevice; symbols: size_t, void, uint32_t, size_t
  - `python/sglang/jit_kernel/tests/test_per_tensor_absmax_fp8.py` added +81/-0 (81 lines); hunks: +import itertools; symbols: reference_absmax_scale, test_absmax_correctness, test_absmax_1d, test_absmax_3d
- Optimization/support interpretation: The concrete diff surface is `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`; keywords observed in patches: fp8, cuda, config, moe, quant, test. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22208 - [AMD] Optimize fused MoE kernel config for small-M decode on gfx950

- Link: https://github.com/sgl-project/sglang/pull/22208
- Status/date: `open`, created 2026-04-06; author `Arist12`.
- Diff scope read: `1` files, `+20/-6`; areas: MoE/router, kernel, docs/config; keywords: benchmark, config, marlin, moe, triton.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunks: def get_default_config(; symbols: get_default_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; keywords observed in patches: benchmark, config, marlin, moe, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22488 - Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation

- Link: https://github.com/sgl-project/sglang/pull/22488
- Status/date: `open`, created 2026-04-10; author `xu-yfei`.
- Diff scope read: `4` files, `+794/-53`; areas: MoE/router, kernel, tests/benchmarks; keywords: cuda, expert, moe, topk, cache, config, quant, spec, test.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunks: +/* Copyright 2025 SGLang Team. All Rights Reserved.; symbols: int, int, int, void
  - `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunks: +import sys; symbols: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts
  - `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunks: is_npu,; def fused_topk_deepseek(; symbols: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped
  - `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunks: +from __future__ import annotations; symbols: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py`; keywords observed in patches: cuda, expert, moe, topk, cache, config. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22496 - [Feature] kimi k25 w4a16 support deepep low latency

- Link: https://github.com/sgl-project/sglang/pull/22496
- Status/date: `open`, created 2026-04-10; author `zhangxiaolei123456`.
- Diff scope read: `11` files, `+4882/-25`; areas: MoE/router, quantization, kernel; keywords: cuda, expert, cache, moe, config, marlin, deepep, topk, triton, fp4.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0 (1948 lines); hunks: +/*; symbols: void, void, int, auto
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +1264/-6 (1270 lines); hunks: #pragma once; __global__ void permute_cols_kernel(; symbols: void, void, void, void
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunks: _is_hip = is_hip(); def create_moe_runner(; symbols: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_tma_utils.h` added +240/-0 (240 lines); hunks: +#pragma once; symbols: uint32_t, uint32_t, alignas, alignas
  - `python/sglang/jit_kernel/mask_silu_and_mul.py` added +229/-0 (229 lines); hunks: +from __future__ import annotations; symbols: MaskedSiluAndMulKernelConfig:, threads_n, _masked_silu_and_mul_triton_kernel, _validate_kernel_config
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`; keywords observed in patches: cuda, expert, cache, moe, config, marlin. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22806 - feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading

- Link: https://github.com/sgl-project/sglang/pull/22806
- Status/date: `open`, created 2026-04-14; author `MichaelPBX`.
- Diff scope read: `5` files, `+548/-9`; areas: model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: moe, config, expert, fp8, quant, spec, triton, fp4, cuda, kv.
- Code diff details:
  - `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0 (363 lines); hunks: +"""Unit tests for KimiW4AFp8Config and related functionality.; symbols: _make_kimi_quant_config, TestKimiW4AFp8ConfigFromConfig, method, test_basic_parsing
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunks: class W4AFp8Config(QuantizationConfig):; def get_config_filenames(cls) -> List[str]:; symbols: W4AFp8Config, for, for, __init__
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunks: def do_load_weights(; symbols: do_load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunks: def make_expert_params_mapping_fused_mxfp4(; symbols: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunks: def override_quantization_method(self, *args, **kwargs):; def override_quantization_method(self, *args, **kwargs):; symbols: override_quantization_method, override_quantization_method
- Optimization/support interpretation: The concrete diff surface is `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; keywords observed in patches: moe, config, expert, fp8, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22964 - [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output

- Link: https://github.com/sgl-project/sglang/pull/22964
- Status/date: `open`, created 2026-04-16; author `litmei`.
- Diff scope read: `1` files, `+6/-1`; areas: multimodal/processor; keywords: cuda, processor.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunks: def _cpu_call(self, text, images, **kwargs):; symbols: _cpu_call, _get_gpu_norm_tensors
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/kimi_k25.py`; keywords observed in patches: cuda, processor. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/kimi_k25.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23186 - [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4

- Link: https://github.com/sgl-project/sglang/pull/23186
- Status/date: `merged`, created 2026-04-19, merged 2026-04-21; author `akao-amd`.
- Diff scope read: `1` files, `+12/-0`; areas: model wrapper, attention/backend; keywords: attention, cache, fp8, kv, mla, quant, triton.
- Code diff details:
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunks: def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; def forward_absorb_prepare(; symbols: bmm_fp8, forward_absorb_prepare
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; keywords observed in patches: attention, cache, fp8, kv, mla, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 54; open PRs: 7.
- Open PRs to keep tracking: [#19703](https://github.com/sgl-project/sglang/pull/19703), [#21741](https://github.com/sgl-project/sglang/pull/21741), [#22208](https://github.com/sgl-project/sglang/pull/22208), [#22488](https://github.com/sgl-project/sglang/pull/22488), [#22496](https://github.com/sgl-project/sglang/pull/22496), [#22806](https://github.com/sgl-project/sglang/pull/22806), [#22964](https://github.com/sgl-project/sglang/pull/22964)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
