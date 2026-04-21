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
