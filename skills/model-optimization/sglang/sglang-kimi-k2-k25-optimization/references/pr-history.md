# Kimi K2/K2.5 Optimization History

This reference was built from `git log --first-parent main`, local `git show`, merged PR pages, and the current open PR radar in `sgl-project/sglang`. It includes merged commits that directly changed Kimi K2 or Kimi K2.5 performance, kernel selection, quantized execution, or large-scale runtime plumbing.

When a PR included benchmark numbers, the tables below copy representative rows from the PR body instead of re-running the benchmark locally. For kernel PRs, the focus is on which hot path changed, why it changed, and which code pattern was introduced.

Excluded on purpose for the historical PR ladder:

- parser-only and tool-call formatting fixes
- CI-only and nightly-only changes
- docs-only changes
- platform bring-up commits that did not materially change the optimization playbook

The current-main snapshot below is an exception to that exclusion rule. It records active docs, parser tests, CI lanes, and backend-specific tests because those now define the validation surface for Kimi changes even when they are not themselves optimization PRs.

## Current Main Coverage Snapshot

Snapshot:
SGLang `origin/main` commit `c122d343a`, checked on `2026-04-21`.

Current Kimi-K2.5 serving contract:

- `docs_new/docs/basic_usage/kimi_k2_5.mdx` documents `moonshotai/Kimi-K2.5` as a 1T-parameter multimodal MoE with 32B active parameters, 256K context, MLA attention, MoonViT vision, thinking and instant modes, and image input support through the OpenAI-compatible vision API.
- The launch example uses both `--tool-call-parser kimi_k2` and `--reasoning-parser kimi_k2`; thinking is default, with instant mode controlled through `extra_body.chat_template_kwargs.thinking=false`.
- `test/registered/8-gpu-models/test_kimi_k25.py` runs Kimi-K2.5 with TP8 and TP8+DP8 variants, both with the `kimi_k2` tool and reasoning parsers.

Current parser and OpenAI-serving coverage:

- `python/sglang/srt/function_call/function_call_parser.py` maps `kimi_k2` to `KimiK2Detector`.
- `python/sglang/srt/parser/reasoning_parser.py` maps `kimi_k2` to the Kimi-K2 reasoning detector.
- `test/registered/function_call/test_kimik2_detector.py` covers non-streaming, streaming, structural tag, special-token leakage, and end-to-end reasoning plus function-call interactions.
- `test/registered/unit/function_call/test_function_call_parser.py`, `test/registered/unit/parser/test_reasoning_parser.py`, and `test/registered/unit/entrypoints/openai/test_serving_chat.py` add unit-level coverage for parser selection and `kimi_k2` OpenAI tool-call id formatting.

Current Kimi-K2.5 multimodal processor coverage:

- `python/sglang/srt/multimodal/processors/kimi_common.py` contains `KimiGridMMDataMixin`, shared by KimiVL and Kimi-K2.5 processors.
- `python/sglang/srt/multimodal/processors/kimi_k25.py` includes GPU image preprocessing utilities and returns `image_grid_thw` / `grid_thws` metadata used by the model.
- `python/sglang/srt/multimodal/processors/base_processor.py` maps `grid_thws` to image modality for Kimi-K2.5.

Current backend and adapter validation surface:

- `test/registered/lora/test_lora_kimi_k25_logprob_diff.py` validates Kimi-K2.5 LoRA logprobs against reference training data with TP8, Triton LoRA, `experts_shared_outer_loras=True`, FA4 prefill, and FlashInfer decode.
- `test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py` documents the native Kimi-K2.5 aiter MLA MI35x constraint: TP must be 4 because Kimi-K2.5 has 64 attention heads and the aiter ASM MLA kernel needs 16 heads per GPU.
- `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py` validates Kimi-K2.5-MXFP4 on MI35x at TP8, including default and FP8 KV-cache variants.
- `test/registered/amd/test_kimi_k25_mxfp4.py`, `test/registered/gb300/test_kimi_k25.py`, and `test/registered/gb300/test_kimi_k25_nvfp4.py` are the current hardware/quantization lanes to inspect before changing MXFP4, NVFP4, cache, or backend-specific behavior.
- `test/registered/stress/test_stress_kimi_k2.py` stress-tests `moonshotai/Kimi-K2-Thinking` with `--tool-call-parser kimi_k2` and `--reasoning-parser kimi_k2`, so parser changes should not be validated only by short unit tests.

Current open PR radar:

- [#22806](https://github.com/sgl-project/sglang/pull/22806): adds `KimiW4AFp8Config` and Kimi-K2.5 W4AFP8 loading tests, including expert input-scale mapping for gate, up, and down projections.
- [#22496](https://github.com/sgl-project/sglang/pull/22496): adds Kimi-K2.5 W4A16 DeepEP low-latency support through JIT Marlin/direct DeepEP MoE paths such as `deepep_moe_wna16_marlin_direct.py`.
- [#22964](https://github.com/sgl-project/sglang/pull/22964): fixes `KimiGPUProcessorWrapper._cpu_call` output after processor metadata changed around `grid_thws` and `image_grid_thw`.
- [#23186](https://github.com/sgl-project/sglang/pull/23186): adds an AMD/ROCm BF16 fused QK RMSNorm path for `Kimi-K2.5-MXFP4`; the PR reports GSM8K and throughput movement, so treat it as a backend optimization track.
- [#19703](https://github.com/sgl-project/sglang/pull/19703): migrates `kimi_k2_moe_fused_gate` from the AOT `sgl-kernel` path into `python/sglang/jit_kernel`.
- [#22488](https://github.com/sgl-project/sglang/pull/22488): generalizes the Kimi2 fused MoE gate JIT path to support GLM-5-style 256-expert shapes, which is relevant when reusing the K2 gate skill on nearby MoE families.
- [#22208](https://github.com/sgl-project/sglang/pull/22208): tunes AMD gfx950 small-M fused MoE behavior for Kimi-K2.5 `int4_w4a16`.
- [#21741](https://github.com/sgl-project/sglang/pull/21741): adds generic compressed-tensors W4AFP8 MoE support, a dependency-shaped track for the Kimi W4AFP8 work.

Known closed DeepEP plus int4/Marlin gap:

- [#13789](https://github.com/sgl-project/sglang/pull/13789) tried `moonshotai/Kimi-K2-Thinking --tp 8 --ep 4 --moe-a2a-backend deepep --deepep-mode auto` with `SGLANG_DEEPEP_BF16_DISPATCH=1`, but the PR was closed unmerged.
- The reported failure was an illegal memory access in the `fused_marlin_moe` path after `DeepEPMoE.forward_marlin_moe(...)` called the compressed-tensors MoE DeepEP-normal path.
- [#19181](https://github.com/sgl-project/sglang/pull/19181) later landed the generic JIT `moe_wna16_marlin` kernel, but that does not by itself prove Kimi-K2-Thinking DeepEP plus int4/Marlin is production-ready.
- [#22496](https://github.com/sgl-project/sglang/pull/22496) is the active related work, but it targets Kimi-K2.5 W4A16 DeepEP low-latency rather than Kimi-K2-Thinking.

## K2: Router, Gating, and MoE Kernel Path

### `6bdd27861` / [#8013](https://github.com/sgl-project/sglang/pull/8013) - `dsv3_router_gemm` supports `NUM_EXPERTS == 384`

- Expands `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` from a fixed `256`-expert assumption to `{256, 384}` while keeping hidden size `7168`.
- Adds bf16 and fp32 template instantiations for token counts `1..16` in both `dsv3_router_gemm_bf16_out.cu` and `dsv3_router_gemm_float_out.cu`.
- Parameterizes `sgl-kernel/tests/test_dsv3_router_gemm.py` and `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` across `256` and `384` experts.

Capability change:

| Aspect       | Before       | After              |
| ------------ | ------------ | ------------------ |
| Expert count | fixed `256`  | `256` or `384`     |
| Hidden dim   | fixed `7168` | still fixed `7168` |
| Output dtype | fp32 / bf16  | fp32 / bf16        |
| Token count  | `1..16`      | `1..16`            |

Code focus:

- The old `constexpr int num_experts = 256;` path becomes runtime-dispatched by `mat_b.size(0)`.
- The kernel still relies on template-specialized unrollers, but now instantiates separate `LoopUnroller<..., 256, 7168>` and `LoopUnroller<..., 384, 7168>` branches.
- This is the prerequisite that makes later K2-specific router and gating work possible without falling back to slower generic code.

Key code:

```cpp
const int num_experts = mat_b.size(0);
TORCH_CHECK(
    num_experts == DEFAULT_NUM_EXPERTS || num_experts == KIMI_K2_NUM_EXPERTS,
    "Expected num_experts=256 or num_experts=384");

if (num_experts == KIMI_K2_NUM_EXPERTS) {
  LoopUnroller<1, 16, KIMI_K2_NUM_EXPERTS, DEFAULT_HIDDEN_DIM>::unroll_float_output(...);
}
```

any K2 optimization that reuses DeepSeek router kernels must first remove the hidden `256`-expert assumption.

### `a1cb717d0` / [#13150](https://github.com/sgl-project/sglang/pull/13150) - optimized biased topk for K2 thinking

- Adds `kimi_k2_biased_topk_impl` to `python/sglang/srt/layers/moe/topk.py`.
- Special-cases `num_experts == 384` and `num_expert_group == 1`, so the code skips group masking, score masking, and the generic grouped-topk path.
- Keeps the existing post-processing behavior: renormalization, routed scaling, logical-to-physical expert remap, and padded-token masking.

PR benchmark highlights:

| Metric                              |      Main |        PR |
| ----------------------------------- | --------: | --------: |
| Profiler hotspot                    |   `33 us` |   `15 us` |
| Output throughput, concurrency `1`  |  `61.101` |  `64.239` |
| Mean TPOT ms, concurrency `1`       |   `8.424` |   `7.501` |
| Output throughput, concurrency `32` | `267.557` | `275.438` |
| Mean TPOT ms, concurrency `32`      |  `38.666` |  `38.297` |

Code focus:

- The optimization happens entirely in Python-level dispatch and `torch.compile`, not in a new CUDA kernel.
- The fast path directly computes `scores.sigmoid() + correction_bias`, then runs `torch.topk(...)` without grouped masking because K2 has `num_expert_group == 1`.
- The dispatch site in `biased_grouped_topk_gpu(...)` starts routing K2 shapes to this narrow implementation instead of the generic grouped version.

Key code:

```python
scores = gating_output.sigmoid()
tmp_scores = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
_, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
topk_weights = scores.gather(1, topk_ids)
```

for K2 thinking, first remove unnecessary grouped-topk work before trying to micro-optimize the generic path.

### `1d3d42bda` / [#13287](https://github.com/sgl-project/sglang/pull/13287) - adds `kimi_k2_moe_fused_gate`

- Introduces a new CUDA op, `sgl_kernel::kimi_k2_moe_fused_gate`, wired through `common_extension.cc` and `sgl_kernel_ops.h`.
- Adds `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, which fuses sigmoid, correction bias, top-k selection, and optional renormalization/scaling into one kernel.
- Adds both a benchmark and a dedicated unit test.

Representative PR benchmark rows:

| Seq length | Torch Compile us | Fused Kernel us |
| ---------- | ---------------: | --------------: |
| `1`        |         `10.687` |         `7.984` |
| `1024`     |         `30.023` |        `16.248` |
| `40000`    |        `775.371` |       `110.548` |

| Metric                              |       Main |         PR |
| ----------------------------------- | ---------: | ---------: |
| Profiler hotspot                    |    `14 us` |     `9 us` |
| Output throughput, concurrency `1`  |   `60.104` |   `65.973` |
| Mean TTFT ms, concurrency `1`       |  `193.579` |  `166.876` |
| Output throughput, concurrency `32` |  `271.123` |  `283.044` |
| Mean TTFT ms, concurrency `32`      | `1436.165` | `1371.391` |

CUDA kernel focus:

- Small-token path:
  one token per block, `12` warps collaborate on the same token, and all `384` experts are staged into shared memory.
- Large-token path:
  one CTA handles `6` warps (`WARPS_PER_CTA = 6`), each warp owns one row and performs in-warp top-k selection.
- Fused work:
  `sigmoid`, `+ bias`, top-k selection, writeback of selected weights and indices, and optional renormalize/scaling all stay inside the same kernel launch.
- Validation path:
  the PR adds `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` and `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, so later regressions have a dedicated harness.

Key code:

```cpp
static constexpr int NUM_EXPERTS = 384;
static constexpr int SMALL_TOKEN_THRESHOLD = 512;
static constexpr int WARPS_PER_TOKEN_SMALL = 12;

float sigmoid_val = 1.0f / (1.0f + expf(-static_cast<float>(input_val)));
float biased_val = sigmoid_val + static_cast<float>(bias_val);
shared_scores[tid] = biased_val;
shared_original_scores[tid] = sigmoid_val;
```

```cpp
if (lane_id == 0 && final_max_expert != -1) {
  output_ptr[output_idx] = shared_original_scores[final_max_expert];
  indices_ptr[output_idx] = final_max_expert;
  shared_scores[final_max_expert] = -FLT_MAX;
}
```

K2 got a model-specific fused router path because the `384`-expert decode case was valuable enough to justify a custom kernel.

### `50691d7b4` / [#13332](https://github.com/sgl-project/sglang/pull/13332) - applies the fused gate in runtime

- Imports `kimi_k2_moe_fused_gate` in `topk.py`.
- Switches the CUDA `384`-expert K2 path from `kimi_k2_biased_topk_impl` to the new fused CUDA kernel.
- Leaves the generic compiled implementation intact for non-K2 shapes and non-CUDA backends.

once the dedicated kernel exists, the runtime should dispatch to it by shape instead of relying on the general compiled implementation.

### `820e13c9c` / [#13374](https://github.com/sgl-project/sglang/pull/13374) - optimizes the K2 fused gate kernel

- Removes the original bf16/half templating and narrows the K2 fused gate kernel to float32 inputs.
- Reworks the small-token algorithm around iterative selection plus shared-memory staging.
- Vectorizes the large-token path with `float4` loads.

Representative PR benchmark rows:

This PR compares against the previous fused-kernel baseline from [#13287](https://github.com/sgl-project/sglang/pull/13287).

| Seq length | [#13287](https://github.com/sgl-project/sglang/pull/13287) fused kernel us | [#13374](https://github.com/sgl-project/sglang/pull/13374) fused kernel us |
| ---------- | -------------------------------------------------------------------------: | -------------------------------------------------------------------------: |
| `1`        |                                                                    `7.970` |                                                                    `6.391` |
| `1024`     |                                                                   `16.442` |                                                                   `13.550` |
| `40000`    |                                                                  `110.211` |                                                                   `93.820` |

| Metric                              | [#13287](https://github.com/sgl-project/sglang/pull/13287) baseline | [#13374](https://github.com/sgl-project/sglang/pull/13374) PR |
| ----------------------------------- | ------------------------------------------------------------------: | ------------------------------------------------------------: |
| Profiler hotspot                    |                                                            `9.1 us` |                                                      `6.4 us` |
| Input throughput, concurrency `1`   |                                                          `3958.378` |                                                    `4479.016` |
| Output throughput, concurrency `1`  |                                                            `65.973` |                                                      `74.650` |
| Output throughput, concurrency `32` |                                                           `283.044` |                                                     `285.589` |

CUDA kernel focus:

- Datatype simplification:
  templated bf16/half paths are removed, so the hot kernel only optimizes the dtype actually used by K2 here, namely fp32 router scores.
- Large-token vectorization:
  the kernel reinterpret-casts router inputs and bias to `float4*`, loading `4` experts at a time.
- Small-token rewrite:
  instead of storing full warp-local top-k state per stage, the kernel stores only `selected_experts`, `selected_vals`, `warp_maxs`, and `warp_experts`, which shrinks shared-memory pressure.
- Synchronization reduction:
  the PR body explicitly calls out dropping `__syncthreads()` from `2 * topk` to `topk + 1` per token.

Key code:

```cpp
static constexpr int VEC_SIZE = 4;
float4* input_vec = reinterpret_cast<float4*>(input + row_idx * NUM_EXPERTS);
float4* bias_vec = reinterpret_cast<float4*>(bias);
```

```cpp
__shared__ int selected_experts[8];
__shared__ float warp_maxs[WARPS_PER_TOKEN_SMALL];

float my_val = (tid < NUM_EXPERTS) ? shared_scores[tid] : -FLT_MAX;
int my_expert = tid;
```

once the specialized path exists, the next win came from simplifying the dtype story and vectorizing loads rather than adding more generic branching.

### `bfcf15a12` / [#13587](https://github.com/sgl-project/sglang/pull/13587) - deletes useless pad-kernel work

- Removes extra padding logic from `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`.
- The change is small, but it removes unnecessary work on the K2 path where block alignment had already been handled elsewhere.

after the big wins, small alignment-path cleanups can still matter on heavily exercised K2 MoE launches.

### `fb04d4342` / [#13596](https://github.com/sgl-project/sglang/pull/13596) - avoids useless `torch.zeros_`

- Extracts `fused_marlin_moe` from the old `sgl_kernel` Python wrapper into `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`.
- Changes block-size selection to use `try_get_optimal_moe_config(..., is_marlin=True)` instead of a small hardcoded heuristic loop.
- Reuses a shared temporary buffer for intermediate tensors.
- Only zeros `intermediate_cache3` when `expert_map is not None`, which avoids unnecessary zero fills on the non-EP path.
- Updates AWQ, GPTQ, and compressed-tensors MoE methods to import the new local helper instead of the old wrapper.

PR benchmark highlights:

| Metric           |     Main |     PR |
| ---------------- | -------: | -----: |
| Profiler hotspot | `3.5 us` | `2 us` |

Hot-path focus:

- The actual latency win is not a new CUDA kernel. It comes from removing an always-on zeroing path that was accidentally triggered by a fake `expert_map=torch.empty(1, device=x.device)` placeholder.
- Moving `fused_marlin_moe` into SGLang-side Python also decouples the hot path from an `sgl-kernel` release and avoids circular-import trouble by delaying imports locally.
- The helper now uses a single large `intermediate_cache13` slab and views it into `intermediate_cache1` and `intermediate_cache3`, reducing allocation churn.

Key code:

```python
config = try_get_optimal_moe_config(
    w1.shape,
    w2.shape,
    topk_ids.shape[1],
    None,
    is_marlin=True,
)
```

```python
if expert_map is not None:
    intermediate_cache3.zero_()
```

K2 thinking quantized MoE performance depended on trimming memory traffic and avoiding unconditional zeroing in the hot path.

### `85ffce30a` / [#13466](https://github.com/sgl-project/sglang/pull/13466) - piecewise CUDA graph support for K2

- Registers a fake implementation for `sgl_kernel::kimi_k2_moe_fused_gate` in `topk.py`.
- The fake op preserves output shapes and dtypes for graph capture and compile-time shape reasoning.

custom fast paths need PCG-friendly fake registrations or they become unusable in the very regime where K2 wants them.

### `ae6a6630e` / [#13725](https://github.com/sgl-project/sglang/pull/13725) - EP support for K2 thinking Marlin MoE

- In `compressed_tensors_moe.py`, fetches `layer.dispatcher.local_expert_mapping` when available.
- Passes both `expert_map` and `global_num_experts` into `fused_marlin_moe`.
- Leaves the non-EP path unchanged by using `None` and `-1` defaults when no local mapping exists.

Key code:

```python
expert_map = layer.dispatcher.local_expert_mapping
global_num_experts = self.moe_runner_config.num_experts
fused_marlin_moe(..., global_num_experts=global_num_experts, expert_map=expert_map)
```

K2 thinking EP support was mainly a plumbing change that let the quantized Marlin MoE path understand global-vs-local expert layout.

### `b399e3ac4` / [#15100](https://github.com/sgl-project/sglang/pull/15100) - piecewise CUDA graph support for fused Marlin MoE

- Updates `fused_marlin_moe.py` and related Marlin runner code so the quantized MoE path can run under piecewise CUDA graph.
- Treats Marlin MoE graph-safety as a serving requirement instead of a nice-to-have debug mode.

the K2 thinking Marlin path needs the same PCG hardening discipline as the router fast path.

### `56d12b4ae` / [#15306](https://github.com/sgl-project/sglang/pull/15306) - fixes warp illegal instruction in K2 PCG

- Handles the case where no valid expert is selected by writing zero outputs and zero indices instead of leaving invalid state behind.
- Applies the guard in both the small-token and large-token fused-gate kernels.

CUDA kernel focus:

- Small-token fix:
  after the iterative selection loop, thread `0` now explicitly writes zeros when `selected_experts[k]` is invalid.
- Large-token fix:
  lane `0` always writes an output slot; if `max_expert == -1`, it writes `0.0f` and index `0` instead of skipping the store.
- Why it matters:
  PCG exposes edge cases where partially initialized outputs can survive into later graph replay, leading to illegal instruction or undefined behavior.

Key code:

```cpp
if (expert_id >= 0 && expert_id < NUM_EXPERTS) {
  output_ptr[row_idx * topk + k] = shared_original_scores[expert_id];
  indices_ptr[row_idx * topk + k] = expert_id;
} else {
  output_ptr[row_idx * topk + k] = 0.0f;
  indices_ptr[row_idx * topk + k] = 0;
}
```

```cpp
if (max_expert != -1) {
  output_ptr[output_idx] = warp_original_scores[max_expert];
  indices_ptr[output_idx] = max_expert;
} else {
  output_ptr[output_idx] = 0.0f;
  indices_ptr[output_idx] = 0;
}
```

once K2 used the custom fused gate inside piecewise CUDA graph, correctness under edge cases mattered as much as raw latency.

### `84c839051` / [#15347](https://github.com/sgl-project/sglang/pull/15347) - runtime prefers `fused_topk_deepseek` for the supported K2 shape

- Changes `biased_grouped_topk_gpu(...)` so supported CUDA grouped-topk shapes can go through flashinfer's `fused_topk_deepseek` before falling back to older paths.
- For K2-style `384`-expert routing, this means the best available fast path may be the maintained DSV3 routing kernel instead of the older dedicated K2 gate kernel.
- Later [#17325](https://github.com/sgl-project/sglang/pull/17325) fixes kernel selection in `biased_grouped_topk_gpu`, reinforcing that dispatch order is part of the optimization contract.

Code focus:

- The runtime first checks whether the current grouped-topk shape satisfies the fused-topk-deepseek constraints.
- Only if those constraints fail does it fall through to `moe_fused_gate`, `aiter`, or the K2-specific fused gate fallback.
- In practice, "the K2 fast path" on current main means "the fastest matching specialized router op", not automatically "always `kimi_k2_moe_fused_gate`".

when optimizing K2 on current main, verify which specialized router path is actually selected before micro-optimizing the older dedicated gate kernel.

### `beabaa8d3` / [#19181](https://github.com/sgl-project/sglang/pull/19181) - migrates Marlin MoE kernel implementation to JIT

- Introduces `python/sglang/jit_kernel/moe_wna16_marlin.py` plus dedicated tests and benchmarks for the Marlin MoE GEMM path.
- Leaves `fused_marlin_moe.py` as the higher-level launcher and orchestrator, but moves the active kernel implementation into the JIT kernel stack.
- Changes where future optimization work should land: kernel tuning now requires reading the JIT kernel path, not only the old wrapper logic.

on current main, quantized K2 thinking optimization is partly a launcher problem and partly a JIT-kernel problem; editing only the wrapper is no longer enough.

## K2: Hardware-Specific Fused MoE Tuning Files

### `14f1f1514` / [#8047](https://github.com/sgl-project/sglang/pull/8047) - H20 config

- Adds `triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- The file supplies per-batch `BLOCK_SIZE_*`, `GROUP_SIZE_M`, `num_warps`, and `num_stages`.

K2 needed its own device-specific MoE table instead of borrowing generic DeepSeek numbers.

### `c07f647c9` / [#8021](https://github.com/sgl-project/sglang/pull/8021) - H20-3e config

- Adds `triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- Compared with the H20 config, the filename itself shows a different `N` value, which means the tuning target is not identical.

treat hardware and effective MoE shape as a pair. A K2 tuning file is only valid for the encoded `E`, `N`, dtype, Triton version, and device.

### `0f9b11e31` / [#8176](https://github.com/sgl-project/sglang/pull/8176) - H200 config, `E=385`

- Adds `triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- The extra expert count compared with pure `384` is visible in the filename.
- Inference: this likely targets a configuration where an additional expert-like slot matters, possibly a fused shared-expert path. This is inferred from the filename, not explicitly documented in the diff.

do not assume every K2 tuning file uses `E=384`; inspect the exact filename before reusing it.

### `f62d75b6a` / [#8178](https://github.com/sgl-project/sglang/pull/8178) - B200 config, `E=385`

- Adds `triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- The config includes several B200-specific choices such as larger `BLOCK_SIZE_N` at higher batch sizes and an `8`-warp entry at batch `2048`.

Blackwell-specific K2 tuning should start from the B200 table instead of extrapolating from Hopper.

### `bbcfbc1a0` / [#8183](https://github.com/sgl-project/sglang/pull/8183) - H200 config, `E=384`

- Adds `triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- This sits alongside the `E=385` H200 file.

the repository preserved both shapes, so reuse requires checking whether the current runtime shape matches the file naming exactly.

### `20cfc5a25` / [#9010](https://github.com/sgl-project/sglang/pull/9010) - B200 config, `E=384,N=256`

- Adds `triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`.
- The move from `triton_3_3_1` to `triton_3_4_0` is part of the encoded compatibility surface.

for B200 K2, both Triton version and shape-specific config naming are part of the optimization contract.

## K2.5: Wrapper, Quantization, and Runtime Plumbing

### `479ab7a4e` / [#17789](https://github.com/sgl-project/sglang/pull/17789) - initial K2.5 model support

- Adds `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/models/kimi_k25.py`, and the multimodal processor.
- Wraps a `DeepseekV3ForCausalLM` language model with K2.5-specific vision tower and projector plumbing.
- Extends reasoning-parser handling so `kimi_k2` can be forced from request template kwargs.

most later K2.5 optimization PRs patch the wrapper around the language model rather than a new standalone language core.

### `d11ccc0a0` / [#17991](https://github.com/sgl-project/sglang/pull/17991) - avoids double reduce in the K2.5 VLM DP-attention path

- Touches both `python/sglang/srt/layers/attention/vision.py` and `python/sglang/srt/models/kimi_k25.py`.
- Fixes a DP-attention-side issue that could introduce an extra reduction in the VLM path.

K2.5 multimodal scaling is not only about DP-sharding the vision encoder; the DP-attention path must also stay reduction-correct.

### `599c5f492` / [#18064](https://github.com/sgl-project/sglang/pull/18064) - fixes K2.5 MoE config initialization

- Changes `Scheduler.init_moe_gemm_config()` to inspect `hf_config.text_config` when present.
- The original logic only checked the top-level `hf_config`, which misses MoE fields for multimodal wrappers.

for K2.5, MoE init logic must often look through the multimodal wrapper into the text config.

### `7b8365931` / [#18370](https://github.com/sgl-project/sglang/pull/18370) - fixes NVFP4 weight mapping and exclude list

- Adds `hf_to_sglang_mapper` to `KimiK25ForConditionalGeneration`, remapping `language_model.layers.` to `language_model.model.layers.`.
- Adds `ModelOptQuantConfig.apply_weight_name_mapper()` so excluded module patterns are remapped into the SGLang layout.
- Expands mapped excludes with and without the `language_model.` prefix and deduplicates them.

K2.5 quantized checkpoints needed explicit weight-layout normalization before performance features could even load correctly.

### `071bf2ce0` / [#18440](https://github.com/sgl-project/sglang/pull/18440) - stores `quant_config` on the wrapper

- Adds `self.quant_config = quant_config` inside `KimiK25ForConditionalGeneration.__init__`.
- Tiny change, but it preserves quantization metadata on the wrapper model.

some K2.5 performance or load paths depend on the wrapper carrying quantization context, not only the language submodule.

## K2.5: Parallelism, Multimodal Scaling, and Speculative Decoding

### `4a3a787f1` / [#18434](https://github.com/sgl-project/sglang/pull/18434) - K2.5 PP support

- Threads `pp_proxy_tensors` through `KimiK25ForConditionalGeneration.forward()`.
- In `DeepseekV2Model.forward()`, moves hidden-state initialization before device selection so `device` comes from the active hidden state, not only optional `input_embeds`.
- Keeps PP first-rank and non-first-rank logic explicit.

K2.5 PP support was mostly about making the wrapper and DeepSeek core behave correctly when the wrapper sits on top of pipeline stages.

### `5a7ae059e` / [#18689](https://github.com/sgl-project/sglang/pull/18689) - DP ViT support for K2.5

- Adds `self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder`.
- Passes `use_data_parallel` into the vision tower constructor and then down into `MoonViTEncoderLayer`.
- In `get_image_feature()`, adds a DP encoder path using `run_dp_sharded_mrope_vision_model(...)`.

for K2.5 multimodal performance, the encoder path needed its own DP-aware fast path instead of reusing only the auto-batched local encode.

### `a1ef8e2cc` / [#19228](https://github.com/sgl-project/sglang/pull/19228) - AMD K2.5 fused MoE tuning

- Extends the fused-MoE tuning utilities to handle encoder-decoder style wrappers by looking through `text_config`.
- Learns block shape from `quantization_config["config_groups"]` when present.
- Adds `use_int4_w4a16` throughout the tuning and benchmarking pipeline.
- Injects `ServerArgs` into benchmark workers so runtime-dependent config selection matches real serving.
- Adds two K2.5 config files:
  `triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` and the `_down` companion file.

PR benchmark highlights:

Prefill and decode hotspot data from the PR body:

| Stage              |     Before |      After |
| ------------------ | ---------: | ---------: |
| Prefill first MoE  |  `9.11 ms` | `2.881 ms` |
| Prefill second MoE | `4.284 ms` | `1.461 ms` |
| Decode first MoE   |   `501 us` |   `276 us` |
| Decode second MoE  |   `180 us` |    `82 us` |

End-to-end throughput data from the PR body:

| Concurrency | Ori tok/s/user | Opt tok/s/user | Ori tput/GPU | Opt tput/GPU |
| ----------- | -------------: | -------------: | -----------: | -----------: |
| `4`         |    `45.097500` |    `50.610000` |   `22.54875` |   `25.30500` |
| `8`         |    `30.676250` |    `44.176250` |   `30.67625` |   `44.17625` |
| `16`        |    `20.511250` |    `32.547500` |   `41.02250` |   `65.09500` |
| `32`        |    `13.784688` |    `22.509687` |   `55.13875` |   `90.03875` |

Tooling and tuning focus:

- Wrapper-aware config readout:
  `get_model_config(...)` now switches to `config.get_text_config()` before reading MoE fields, which is required for K2.5's multimodal wrapper.
- Int4 naming and shape fix:
  config filename generation changes `N` for `int4_w4a16`, because the packed weight layout changes the effective kernel shape.
- Synthetic benchmark tensors:
  tuning code adds `torch.uint8` packed weights and bfloat16 scale tensors for `int4_w4a16`.
- Runtime parity during tuning:
  each worker receives `ServerArgs` and calls `set_global_server_args_for_scheduler(server_args)`, preventing the tuner from using a different runtime shape than serving.

Key code:

```python
if hasattr(config, "text_config"):
    config = config.get_text_config()

config_groups = config.quantization_config["config_groups"]
group_size = first_group.get("weights", {}).get("group_size")
block_shape = [0, group_size]
```

```python
N = shard_intermediate_size // 2
if use_int4_w4a16:
    N = N // 2
```

```python
server_args = ServerArgs(model_path=args.model, tp_size=args.tp_size, ep_size=args.ep_size)
workers = [BenchmarkWorker.remote(args.seed, server_args) for _ in range(num_gpus)]
```

AMD K2.5 tuning required both new dtype support and wrapper-aware model introspection; only adding a JSON config file would not have been enough.

### `85f7a0aa3` / [#19689](https://github.com/sgl-project/sglang/pull/19689) - K2.5 Eagle3 support

- Adds wrapper methods that delegate to the language model:
  `set_eagle3_layers_to_capture`, `get_embed_and_head`, and `set_embed_and_head`.
- Does not change the language core itself; it exposes the required hooks through the K2.5 wrapper.

K2.5 speculative decoding support was largely a wrapper-surface problem.

### `069d4c577` / [#19959](https://github.com/sgl-project/sglang/pull/19959) - PP layer range exposure for PD disaggregation

- Adds `start_layer` and `end_layer` properties to the K2.5 wrapper.
- Exposes the language-model stage boundaries directly on `KimiK25ForConditionalGeneration`.

PD and PP features expect the wrapper to surface layer-range metadata instead of hiding it one level down.

### `24a27d532` / [#20747](https://github.com/sgl-project/sglang/pull/20747) - piecewise CUDA graph support for K2.5 VLM

- Adds `self.model = self.language_model.model` to the wrapper.
- Small change, but it exposes the model object the way other runtime utilities expect.

K2.5 VLM PCG support depended on wrapper compatibility with generic runtime introspection.

### `01ccdb91b` / [#21004](https://github.com/sgl-project/sglang/pull/21004) - EPLB rebalance support for K2.5

- Adds `routed_experts_weights_of_layer` as a wrapper property that forwards to `self.language_model._routed_experts_weights_of_layer.value`.
- This exposes routed-expert weights to rebalancing logic without unwrapping the language model manually.

EPLB support on wrapped multimodal models often reduces to exposing the right language-model property at the wrapper boundary.

### `8c3ccef2d` / [#21391](https://github.com/sgl-project/sglang/pull/21391) - fixes DP-attention plus speculative-decoding launch crash

- In `llama_eagle3.py`, when extending a multimodal batch, uses `forward_batch.mm_input_embeds` and only appends the final token embedding from `embed_tokens(...)`.
- Avoids re-embedding the already-prepared multimodal prefix during extend mode.
- Updates `test/registered/8-gpu-models/test_kimi_k25.py` to add a `TP8+DP8+MTP` variant and drops a fixed `--mem-frac=0.85`.

once K2.5 combines multimodal inputs, DP attention, and Eagle3, launch correctness depends on using the prebuilt multimodal embeddings instead of reconstructing them naively.

## Practical Exclusions

These commits are Kimi-related but not part of the optimization playbook:

- parser and tool-call format fixes such as [#8043](https://github.com/sgl-project/sglang/pull/8043), [#9606](https://github.com/sgl-project/sglang/pull/9606), [#10612](https://github.com/sgl-project/sglang/pull/10612), [#10972](https://github.com/sgl-project/sglang/pull/10972), [#19120](https://github.com/sgl-project/sglang/pull/19120), [#19552](https://github.com/sgl-project/sglang/pull/19552)
- nightly or CI additions such as [#17523](https://github.com/sgl-project/sglang/pull/17523), [#17656](https://github.com/sgl-project/sglang/pull/17656), [#18269](https://github.com/sgl-project/sglang/pull/18269), [#19802](https://github.com/sgl-project/sglang/pull/19802)
- pure platform bring-up such as [#12759](https://github.com/sgl-project/sglang/pull/12759), [#19331](https://github.com/sgl-project/sglang/pull/19331), or the later Ascend `w4a8` support commit

Use those only if the problem is parser behavior, CI registration, or backend bring-up rather than optimization.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Kimi K2 / K2.5 PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

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

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
