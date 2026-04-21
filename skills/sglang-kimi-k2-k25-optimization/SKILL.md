---
name: sglang-kimi-k2-k25-optimization
description: PR-backed and current-main optimization manual for `moonshotai/Kimi-K2*` and `moonshotai/Kimi-K2.5*` in SGLang. Use when Codex needs to recover, extend, or audit Kimi optimizations, including K2 router/MoE fast paths, K2 thinking Marlin paths, K2.5 wrapper/multimodal/runtime plumbing, W4AFP8/W4A16 quant tracks, parser contracts, LoRA coverage, and backend-specific validation.
---

# SGLang Kimi K2/K2.5 Optimization

## Overview

The skill is an optimization ladder. Identify which stage the current code is at, apply the next missing optimization, and only move deeper after the earlier stage is satisfied.

Current-main snapshot:
This skill was refreshed against SGLang `origin/main` commit `c122d343a` on `2026-04-21`. Since the older PR ladder was written, current main has added a Kimi-K2.5 usage doc, parser and OpenAI-serving tests for `kimi_k2`, Kimi-K2.5 LoRA regression coverage, AMD/GB300 validation lanes, and a Kimi-K2-Thinking stress test. Treat those as part of the active validation surface, not as optional CI trivia.
Active open PRs now also define several next likely skill updates: W4AFP8 loading, W4A16 DeepEP low-latency, Kimi-K2.5 multimodal processor fixes, ROCm fused QK RMSNorm, and JIT migration of the older K2 fused gate path.
One important non-open gap is Kimi-K2-Thinking DeepEP plus int4/Marlin: [#13789](https://github.com/sgl-project/sglang/pull/13789) tried to support it but was closed unmerged after hitting an illegal memory access in the `fused_marlin_moe` path. Do not mark that combination as mainline-supported just because the generic Marlin JIT work in [#19181](https://github.com/sgl-project/sglang/pull/19181) landed.

The historical evidence for every stage lives in:

- [references/pr-history.md](references/pr-history.md): merged PR evidence, benchmark tables, key code
- [references/playbook.md](references/playbook.md): symptom mapping, commands, validation order

## Before You Change Anything

Record the exact serving shape first:

- K2 or K2.5
- thinking or instruct
- text-only or multimodal
- native or quantized weights
- TP / DP / EP / PP / PD / EPLB topology
- speculative decoding enabled or not
- NVIDIA / AMD / other backend
- launch parser pair: `--tool-call-parser kimi_k2` and `--reasoning-parser kimi_k2`
- LoRA enabled or not
- current SGLang commit and the matching registered-test lane

## Core Principle

Do not treat K2 and K2.5 as one optimization problem.

- K2 is mainly a `384 experts` router and MoE hot-path story.
- K2 thinking adds a separate quantized Marlin MoE story.
- K2.5 is much more wrapper-heavy: `text_config`, quant mapping, PP/PD/EPLB, multimodal DP encoder, Eagle3, and PCG compatibility all matter.
- Current K2/K2.5 serving also has a parser contract: tool calls and thinking output are expected to go through `kimi_k2`.
- Current open PRs split into three useful future tracks: quantized K2.5 loading/execution, multimodal wrapper correctness, and backend-specific fused norm or MoE kernel work.

For non-Kimi models, first decide which family they resemble more:

- a K2-like router and MoE hot-path problem
- a K2 thinking-like quantized Marlin MoE problem
- or a K2.5-like wrapper, multimodal, and runtime-plumbing problem

The optimization order also matters:

1. enable the right shape or wrapper contract
2. remove generic overhead
3. add or select the Kimi-specific fast path
4. micro-optimize the hot kernel
5. harden PCG or distributed correctness
6. tune per-device fused MoE configs

## What Transfers To Similar Models

Reuse this skill on a non-Kimi model when it shares one or more of these traits:

- a sparse MoE router whose real expert count or top-k shape breaks generic DeepSeek-era assumptions
- a wrapper that hides text-model metadata inside `text_config` or wrapper-local quant fields
- a multimodal tower whose throughput depends on DP encoder or DP attention execution details
- speculative decoding, PP, PD, EPLB, or PCG features that depend on wrapper surface area rather than only kernel speed

Reuse the optimization order and validation discipline, not the literal Kimi constants or filenames.

## Open PR Radar

Check these active upstream tracks before designing a new Kimi skill or declaring a gap:

- [#22806](https://github.com/sgl-project/sglang/pull/22806): `KimiW4AFp8Config` and W4AFP8 model-loading support for Kimi-K2.5.
- [#22496](https://github.com/sgl-project/sglang/pull/22496): Kimi-K2.5 W4A16 DeepEP low-latency path with JIT Marlin and direct DeepEP MoE work.
- [#22964](https://github.com/sgl-project/sglang/pull/22964): `KimiGPUProcessorWrapper._cpu_call` output fix after grid metadata changed from `grid_thws` toward `image_grid_thw`.
- [#23186](https://github.com/sgl-project/sglang/pull/23186): AMD/ROCm BF16 fused QK RMSNorm path for `Kimi-K2.5-MXFP4`.
- [#19703](https://github.com/sgl-project/sglang/pull/19703): migrate `kimi_k2_moe_fused_gate` from AOT `sgl-kernel` into the JIT kernel module.
- [#22488](https://github.com/sgl-project/sglang/pull/22488): generalize the Kimi2 fused MoE gate JIT path to GLM-5-style `256`-expert shapes.
- [#22208](https://github.com/sgl-project/sglang/pull/22208): AMD gfx950 small-M fused MoE config tuning for Kimi-K2.5 `int4_w4a16`.
- [#21741](https://github.com/sgl-project/sglang/pull/21741): generic compressed-tensors W4AFP8 MoE support that underpins some Kimi-K2.5 quantized loading work.

Known closed gap to remember:

- [#13789](https://github.com/sgl-project/sglang/pull/13789): attempted Kimi-K2-Thinking DeepEP support with int4/Marlin, but remained unmerged after an illegal memory access in `fused_marlin_moe`.

## K2 Evolution Path

Use this path when the target is `moonshotai/Kimi-K2*`.

### Stage K2-0: Basic support but not yet K2-shaped

The code may "support K2" in a broad sense, but still inherit DeepSeek assumptions that silently cap performance or break specialized kernels.

- hidden assumptions that `num_experts == 256`
- generic grouped-topk path still used for K2
- no dedicated K2 router or gate dispatch

- [#8013](https://github.com/sgl-project/sglang/pull/8013)

- router-side code can legally run with `384` experts
- tests and benchmarks no longer hardcode `256` for the K2 path

### Stage K2-1: Remove the `256 experts` router assumption

Before K2 can benefit from any later hot-path work, the router GEMM path has to accept `384` experts.

- make `dsv3_router_gemm` dispatch by runtime expert count
- keep the specialized unrolled kernel structure instead of falling back to a generic slow path
- preserve the existing hidden-size and output-dtype contract

- `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`

- [#8013](https://github.com/sgl-project/sglang/pull/8013)

- the K2 path uses the specialized router kernel with `384` experts
- there is no hidden fallback caused by a hardcoded `256`

### Stage K2-2: Remove generic grouped-topk overhead

K2 thinking has `384` experts and `num_expert_group == 1`. The generic grouped-topk path wastes time on masking and grouping logic that K2 does not need.

- add a narrow K2-specific topk implementation
- keep semantics identical: renormalization, routed scaling, expert remap, padded-token masking
- optimize the dispatch condition before touching CUDA

- `python/sglang/srt/layers/moe/topk.py`

- [#13150](https://github.com/sgl-project/sglang/pull/13150)

- K2 uses a dedicated biased-topk path instead of the generic grouped implementation
- the router hotspot is smaller even before adding a fused CUDA op

### Stage K2-3: Fuse sigmoid, bias, top-k, and renorm into one CUDA op

Once the Python or `torch.compile` topk path is still hot, K2 needs a model-specific fused gate kernel to cut launch overhead and reduce temporary traffic.

- build a dedicated `kimi_k2_moe_fused_gate` op for `384 experts`
- separate small-token and large-token execution strategies
- fuse `sigmoid`, `+ bias`, top-k selection, writeback, and optional renormalize/scaling
- add a dedicated benchmark and unit test at the same time

- `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`
- `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`
- `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`

- [#13287](https://github.com/sgl-project/sglang/pull/13287)

- the K2 router hotspot moves from generic topk to the dedicated fused gate
- benchmark coverage exists for the exact K2 shape

### Stage K2-4: Wire runtime dispatch to the best available K2 fast path

Adding a fast kernel is not enough; the runtime must actually choose the best maintained path for the K2 shape.

- dispatch by K2 shape in `topk.py`
- use `kimi_k2_moe_fused_gate` when it is the active best path for the current backend
- prefer `fused_topk_deepseek` when backend constraints are satisfied and it supersedes the older gate path
- keep the generic implementation alive for non-K2 shapes and non-CUDA backends
- treat dispatch order as part of the optimization contract, not as an incidental detail

- `python/sglang/srt/layers/moe/topk.py`

- [#13332](https://github.com/sgl-project/sglang/pull/13332)
- [#15347](https://github.com/sgl-project/sglang/pull/15347)
- [#17325](https://github.com/sgl-project/sglang/pull/17325)

- CUDA K2 requests no longer route through the old compiled topk path
- kernel selection is deterministic for the active backend and shape

### Stage K2-5: Micro-optimize the K2 fused gate kernel when it is still the hot path

After the fused gate exists, and when it is still the active router fast path, the next gains come from simplifying the kernel and improving memory behavior rather than inventing another algorithm.

- simplify dtype support to the dtype the hot path actually uses
- vectorize large-token loads with `float4`
- reduce `__syncthreads()` in the small-token path
- shrink shared-memory footprint by storing only selected top-k state
- preserve deterministic tie-breaking in reductions

- `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`

- [#13374](https://github.com/sgl-project/sglang/pull/13374)

- profiler shows the fused gate kernel materially smaller than the first fused version
- kernel correctness is unchanged

### Stage K2-6: Harden piecewise CUDA graph correctness

Fast kernels that are not PCG-safe become unusable in exactly the high-performance serving regime K2 cares about.

- register fake ops for shape and dtype propagation
- ensure invalid expert selections write deterministic zero outputs and zero indices
- treat PCG correctness bugs as first-class regressions, not edge cases

- `python/sglang/srt/layers/moe/topk.py`
- `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`

- [#13466](https://github.com/sgl-project/sglang/pull/13466)
- [#15306](https://github.com/sgl-project/sglang/pull/15306)

- PCG capture and replay work without launch crashes or illegal instructions

### Stage K2-7: Clean up and harden the quantized K2 thinking Marlin MoE path

For K2 thinking, a lot of remaining latency and correctness risk lives in memory traffic, scratch-buffer handling, EP plumbing, PCG compatibility, and the active Marlin implementation boundary rather than in the router itself.

- move `fused_marlin_moe` into SGLang-side code so it can evolve without `sgl-kernel` release friction
- choose config via `try_get_optimal_moe_config(..., is_marlin=True)`
- reuse shared temporary buffers
- do not zero scratch buffers unless EP actually requires it
- only pass a real `expert_map` when dispatcher metadata exists
- keep the Marlin path PCG-safe
- optimize the active JIT-backed Marlin kernel implementation, not only the wrapper around it

- `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`
- `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `python/sglang/jit_kernel/moe_wna16_marlin.py`

- [#13596](https://github.com/sgl-project/sglang/pull/13596)
- [#13725](https://github.com/sgl-project/sglang/pull/13725)
- [#15100](https://github.com/sgl-project/sglang/pull/15100)
- [#19181](https://github.com/sgl-project/sglang/pull/19181)

- tp-only K2 thinking no longer pays the fake-EP zeroing penalty
- EP-aware paths pass true expert metadata instead of placeholders
- PCG capture works for the quantized Marlin MoE path
- future kernel changes land in the active JIT-backed implementation instead of only in stale wrapper code

### Stage K2-8: Add hardware-specific fused MoE tuning files

Even with the correct kernel path, K2 MoE throughput is device-sensitive. H20, H200, and B200 needed separate config tables.

- treat the tuning filename as part of the optimization contract
- match exact `Triton version`, `E`, `N`, `dtype`, `block_shape`, and `device_name`
- do not reuse a config file just because the model name is also K2

- `python/sglang/srt/layers/moe/fused_moe_triton/configs/`

- [#8047](https://github.com/sgl-project/sglang/pull/8047)
- [#8021](https://github.com/sgl-project/sglang/pull/8021)
- [#8176](https://github.com/sgl-project/sglang/pull/8176)
- [#8178](https://github.com/sgl-project/sglang/pull/8178)
- [#8183](https://github.com/sgl-project/sglang/pull/8183)
- [#9010](https://github.com/sgl-project/sglang/pull/9010)

- the serving shape resolves to the exact per-device tuning file instead of a generic default

## K2.5 Evolution Path

Use this path when the target is `moonshotai/Kimi-K2.5*`.

### Stage K25-0: Wrapper bring-up

K2.5 is not "just another DeepSeek model". It is a wrapped multimodal model, so optimization begins with the wrapper contract.

- expose the language model through a K2.5 wrapper
- keep multimodal tower and projector plumbing explicit
- do not assume later runtime features can bypass the wrapper

- [#17789](https://github.com/sgl-project/sglang/pull/17789)

- K2.5 can launch with the wrapper and basic multimodal plumbing intact

### Stage K25-1: Make MoE config initialization wrapper-aware

The first K2.5 bottleneck was often not kernel speed, but the wrong config being read because MoE metadata lived in `text_config`.

- inspect `hf_config.text_config` when present
- do not assume top-level `hf_config` contains the text-model MoE fields

- `python/sglang/srt/managers/scheduler.py`

- [#18064](https://github.com/sgl-project/sglang/pull/18064)

- fused MoE config init sees the real text-model expert metadata

### Stage K25-2: Make quantized checkpoint loading robust

Before K2.5 can be optimized, quantized checkpoints must load with the right name mapping and quant metadata propagation.

- remap HF weight names into the SGLang wrapper layout
- remap excluded-module patterns too, not only main weights
- preserve `quant_config` on the wrapper itself

- `python/sglang/srt/models/kimi_k25.py`
- `python/sglang/srt/layers/quantization/modelopt_quant.py`

- [#18370](https://github.com/sgl-project/sglang/pull/18370)
- [#18440](https://github.com/sgl-project/sglang/pull/18440)

- NVFP4 or related quantized K2.5 checkpoints load without wrapper-name mismatch

### Stage K25-3: Add parallel runtime plumbing through the wrapper

K2.5 performance features often fail because the wrapper does not expose the metadata that PP, PD, EPLB, or PCG expect.

- forward `pp_proxy_tensors`
- expose `start_layer` and `end_layer`
- expose `self.model = self.language_model.model`
- expose `routed_experts_weights_of_layer`

- `python/sglang/srt/models/kimi_k25.py`
- `python/sglang/srt/models/deepseek_v2.py`

- [#18434](https://github.com/sgl-project/sglang/pull/18434)
- [#19959](https://github.com/sgl-project/sglang/pull/19959)
- [#20747](https://github.com/sgl-project/sglang/pull/20747)
- [#21004](https://github.com/sgl-project/sglang/pull/21004)

- wrapper-level runtime features no longer need to special-case K2.5 manually

### Stage K25-4: Scale and stabilize the multimodal DP path

For multimodal K2.5, the vision path becomes part of the performance story. Local auto-batching alone was not enough, and DP-attention correctness issues could erase the gains.

- gate the vision tower by `mm_enable_dp_encoder`
- pass `use_data_parallel` through the wrapper and encoder stack
- use the DP-sharded vision execution path when enabled
- avoid extra reduction or mismatched behavior in the VLM DP-attention path

- `python/sglang/srt/models/kimi_k25.py`
- `python/sglang/srt/layers/attention/vision.py`

- [#17991](https://github.com/sgl-project/sglang/pull/17991)
- [#18689](https://github.com/sgl-project/sglang/pull/18689)

- multimodal K2.5 can use the DP encoder path instead of only local vision execution
- enabling DP attention does not introduce extra reduction or launch mismatch

### Stage K25-5: Tune fused MoE for the real K2.5 quant and hardware shape

K2.5 fused MoE can look supported while still using a terrible default config. On AMD, the big gain came from teaching the tuning tools the real K2.5 wrapper and int4 shape.

- look through `text_config` in tuning utilities
- derive `block_shape` from `quantization_config["config_groups"]` when needed
- add full `int4_w4a16` support to benchmark and tuning scripts
- pass `ServerArgs` into worker processes so tuning matches real serving
- save the final per-shape configs under the exact filename contract

- `benchmark/kernels/fused_moe_triton/common_utils.py`
- `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/`

- [#19228](https://github.com/sgl-project/sglang/pull/19228)

- K2.5 no longer falls back to a default fused-MoE config for the active quant and device

### Stage K25-6: Make speculative decoding wrapper-compatible

For K2.5, speculative decoding support was mostly blocked by missing wrapper hooks, not by the draft model core itself.

- forward Eagle3 helper methods through the K2.5 wrapper
- keep embed/head exposure consistent with other optimized model wrappers

- `python/sglang/srt/models/kimi_k25.py`

- [#19689](https://github.com/sgl-project/sglang/pull/19689)

- Eagle3 runtime code can call the wrapper the same way it calls direct language-model wrappers

### Stage K25-7: Harden multimodal + DP attention + speculative decode correctness

Once K2.5 mixes multimodal embeddings, DP attention, and speculative decoding, correctness bugs show up in extend mode and launch paths.

- respect `forward_batch.mm_input_embeds`
- append only the new tail token embedding in extend mode
- validate on the exact combined launch shape, not just TP-only text mode

- `python/sglang/srt/models/llama_eagle3.py`
- `test/registered/8-gpu-models/test_kimi_k25.py`

- [#21391](https://github.com/sgl-project/sglang/pull/21391)

- the combined multimodal + DP attention + MTP or Eagle3 path runs without launch crashes

### Stage K25-8: Keep current-main serving and validation surfaces aligned

Current main has enough Kimi-K2.5 coverage that optimization work should preserve the launch, parser, multimodal-grid, LoRA, and backend validation contracts together.

- keep the documented Kimi-K2.5 launch contract in sync with tests: `--tool-call-parser kimi_k2` and `--reasoning-parser kimi_k2`
- preserve Kimi grid metadata flow: `grid_thws`, `KimiGridMMDataMixin`, and the GPU image preprocessing path with CPU-compatible inputs
- if LoRA, MoE LoRA sharing, attention backend selection, or logprob paths are touched, include the Kimi-K2.5 LoRA regression
- choose AMD validation by backend and quant shape:
  - native Kimi-K2.5 with `aiter` MLA uses TP4 on MI35x because 64 heads at TP8 gives only 8 heads per GPU
  - Kimi-K2.5-MXFP4 MI35x coverage uses TP8 and validates default plus FP8 KV-cache variants
- use GB300/NVFP4 lanes when changing Blackwell-specific quant, cache, or kernel behavior
- use the Kimi-K2-Thinking stress test when parser, long-run stability, or K2 thinking serving paths are involved

- `docs_new/docs/basic_usage/kimi_k2_5.mdx`
- `python/sglang/srt/function_call/kimik2_detector.py`
- `python/sglang/srt/parser/reasoning_parser.py`
- `python/sglang/srt/multimodal/processors/kimi_common.py`
- `python/sglang/srt/multimodal/processors/kimi_k25.py`
- `test/registered/function_call/test_kimik2_detector.py`
- `test/registered/lora/test_lora_kimi_k25_logprob_diff.py`
- `test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py`
- `test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py`
- `test/registered/gb300/test_kimi_k25.py`
- `test/registered/gb300/test_kimi_k25_nvfp4.py`
- `test/registered/stress/test_stress_kimi_k2.py`

- docs, registered launches, and stress tests agree on the parser flags
- image token expansion still derives from grid metadata instead of placeholder counts alone
- backend-specific tests run on the topology they were written for
- LoRA logprob coverage is not skipped when changing K2.5 adapter or shared-expert LoRA behavior

## Usage

When asked to optimize Kimi, or a structurally similar model, do not start from whatever file looks hottest. First place the current code on the ladder above.

Use this decision process:

1. Determine whether the current problem is K2, K2.5, or which family it most closely resembles.
2. Find the highest stage that is already satisfied.
3. Work on the next missing stage, not a later one.
4. Validate narrowly on the exact serving shape.
5. Only after that, widen to more topology combinations.

Examples:

- K2 decode is still spending time in generic grouped-topk:
  you are missing `K2-2`, so do not jump straight to fused-MoE tuning files.
- K2 fused gate exists but PCG crashes:
  you are at `K2-6`, so focus on fake op registration or invalid-selection guards.
- K2.5 int4 on AMD uses a default config:
  you are at `K25-5`, so tune the wrapper-aware fused-MoE path before editing the model wrapper again.
- K2.5 PP works but Eagle3 multimodal crashes:
  you are at `K25-6` or `K25-7`, not at the early support stages.
- K2.5 tool calls, thinking output, image requests, or LoRA logprobs regress after a runtime change:
  you are at `K25-8`; check launch parser flags, grid metadata, OpenAI-serving parser tests, and the targeted backend lane before blaming the MoE kernel.

For non-Kimi models, map them by structure:

- a new sparse MoE model with non-generic expert count or router shape:
  treat it like the K2 path first
- a quantized sparse MoE model whose hot path runs through Marlin:
  treat it like the K2 thinking Marlin path
- a wrapped multimodal model whose PP, PD, DP encoder, or speculative decode support keeps breaking:
  treat it like the K2.5 path first

## Guardrails

- Do not hardcode `256` experts anywhere on the K2 fast path.
- Do not collapse K2 and K2.5 into one generic DeepSeek optimization.
- Do not optimize generic kernels first if K2 already has a dedicated specialization.
- Do not bypass the K2.5 wrapper to make one feature work; later PP, PD, EPLB, or PCG features often depend on that wrapper surface.
- Do not trust a tuning file unless `Triton version`, `E`, `N`, `dtype`, `block_shape`, and `device` all match the current shape.
- Do not validate only TP-only text mode if the real bug involves PP, DP encoder, multimodal inputs, EP, or speculative decoding.
- Do not omit `--tool-call-parser kimi_k2` or `--reasoning-parser kimi_k2` when validating current K2/K2.5 tool or thinking behavior.
- Do not run native Kimi-K2.5 aiter MLA on MI35x at TP8 and treat the failure as a model regression; the registered native aiter MLA lane documents TP4 as the supported shape.
- Do not copy Kimi-specific constants, dispatch predicates, or tuning filenames into a different model unless its serving shape actually matches.

## Validation

For K2 kernel work:

```bash
pytest -q sgl-kernel/tests/test_dsv3_router_gemm.py
pytest -q sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py
pytest -q test/registered/kernels/test_fused_topk_deepseek.py
python benchmark/bench_kimi_k2_moe_fused_gate.py
```

For K2 quantized Marlin MoE work:

```bash
pytest -q test/registered/quant/test_marlin_moe.py
pytest -q python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py
```

For K2.5 wrapper or combined runtime work:

```bash
pytest -q test/registered/8-gpu-models/test_kimi_k25.py
```

For current parser and OpenAI-serving behavior:

```bash
pytest -q test/registered/function_call/test_kimik2_detector.py
pytest -q test/registered/unit/parser/test_reasoning_parser.py -k KimiK2
pytest -q test/registered/unit/function_call/test_function_call_parser.py -k KimiK2
pytest -q test/registered/unit/entrypoints/openai/test_serving_chat.py -k kimi_k2
```

For backend-specific or adapter-sensitive K2.5 work, run only on matching hardware:

```bash
pytest -q test/registered/lora/test_lora_kimi_k25_logprob_diff.py
pytest -q test/registered/amd/accuracy/mi35x/test_kimi_k25_aiter_mla_eval_mi35x.py
pytest -q test/registered/amd/accuracy/mi35x/test_kimi_k25_mxfp4_eval_mi35x.py
pytest -q test/registered/gb300/test_kimi_k25.py
pytest -q test/registered/gb300/test_kimi_k25_nvfp4.py
pytest -q test/registered/stress/test_stress_kimi_k2.py
```

For tuning work:

- rerun only the relevant tuning script
- keep the real model, quant, TP, EP, and backend
- save output under the exact config filename contract

## References

- Historical evidence and benchmark tables: [references/pr-history.md](references/pr-history.md)
- Symptom mapping and validation order: [references/playbook.md](references/playbook.md)
