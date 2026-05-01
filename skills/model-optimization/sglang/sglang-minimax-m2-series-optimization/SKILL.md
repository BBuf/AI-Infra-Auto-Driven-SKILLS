---
name: sglang-minimax-m2-series-optimization
description: "PR-backed and current-main optimization manual for the `MiniMaxAI/MiniMax-M2` series, including M2, M2.1, M2.5, M2.7, and M2.7-highspeed. Use when an engineer needs to recover, extend, or audit MiniMax-specific optimizations, TP QK norm/all-reduce behavior, parser contracts, distributed runtime behavior, quantized loading, or backend-specific validation."
---

# SGLang MiniMax M2 Series Optimization

## Overview

The skill covers the full MiniMax optimization ladder: mainline history, the remaining still-open upstream PR track, and current-main validation lanes. Use it to recover, extend, or audit MiniMax-specific optimizations, or to reuse the patterns on a structurally similar MoE model.

As of `2026-04-21`, refreshed against SGLang `origin/main` commit `c122d343a`, the MiniMax story is split across three sources of truth:

- mainline history already present in `main`
- still-open upstream PRs that are important for MiniMax-M2.5, but not fully landed in `main` yet
- current registered docs/tests/workflows, especially the MiniMax-M2.7 AMD accuracy and performance lanes

This skill tracks all three, but it labels them clearly. Do not assume an optimization from a PR page is already in your local tree, and do not assume MiniMax-M2.7 or M2.7-highspeed is covered by MiniMax-M2.5 validation just because the same model file is used.

The historical evidence for every stage lives in:

- [references/pr-history.md](references/pr-history.md): mainline and still-open PR evidence, benchmark notes, key code patterns
- [references/playbook.md](references/playbook.md): symptom mapping, commands, validation order

## Non-Negotiable Evidence Rule

Use `skills/model-optimization/model-pr-diff-dossier/SKILL.md` as the production bar.
Every PR cited for this family must be based on diff reading, not only PR titles.

## Before You Change Anything

Record the exact serving shape first:

- M2, M2.1, M2.5, or M2.7
- instruct or reasoning-style launch
- native, AWQ, FP8, FP4, ModelSlim, or other quant format
- TP / DP / EP / PP topology
- DP attention enabled or not
- DeepEP, FlashInfer, Triton, or other MoE / attention backend
- piecewise CUDA graph enabled or not
- speculative decoding or Eagle3 enabled or not
- NVIDIA, AMD, NPU, or other backend
- launch parser pair: `--tool-call-parser minimax-m2` and `--reasoning-parser minimax-append-think` when tool/thinking behavior matters
- exact registered suite, workflow job, or hardware lane used for validation

- QK normalization depends on how heads are partitioned or replicated
- M2.5 scale-out performance depends on communication strategy, not only kernels
- quantized checkpoints depend on exact loader conventions

## Core Principle

Do not treat MiniMax as a generic DeepSeek-like MoE.

- MiniMax-M2 is a QK-normalized attention plus sparse-MoE story.
- MiniMax-M2.5 adds a much heavier distributed and quantized runtime story.
- MiniMax-M2.7 currently follows the MiniMax-M2-family model path but has its own AMD accuracy/performance lanes; treat it as a separate validation target.
- MiniMax-M2.7-highspeed is currently visible through upstream docs work, not a separate current-main runtime path.
- TP QK norm/all-reduce is already a two-generation mainline optimization: [#16483](https://github.com/sgl-project/sglang/pull/16483) keeps the older RMSNormTP all-reduce aligned, and [#20673](https://github.com/sgl-project/sglang/pull/20673) adds the fused JIT TP QK norm path behind `SGLANG_USE_FUSED_PARALLEL_QKNORM`.
- The most important distinctions are often not "model size" but:
  - whether attention TP equals model TP
  - whether KV heads are partitioned or replicated
  - whether MoE output should all-reduce, reduce-scatter, or stay fused for the next layer

The optimization order matters:

1. confirm the loader and topology contract
2. fix correctness in the MiniMax-specific path
3. remove generic overhead in the hot path
4. only then add deeper kernel or communication specialization
5. validate on the exact topology that triggered the issue

## What Transfers To Similar Models

Reuse this skill on a non-MiniMax model when it shares one or more of these traits:

- QK normalization whose cost is dominated by TP communication
- KV-head replication when `num_key_value_heads < tp_size`
- sparse MoE without DeepSeek-style shared experts
- topology-dependent attention groups where attention TP is not the same as model TP
- quantized MoE checkpoints that rely on packed or fused module mappings

Reuse the order of investigation and validation discipline, not the MiniMax-specific constants.

## M2 Core Evolution Path

Use this path when the target is `MiniMaxAI/MiniMax-M2*` and the problem is mostly inside the core model path already on `main`.

### Stage M2-0: Basic support exists, but the path is still naive

The model can launch, but the earliest support path is not yet optimized and may still miss MiniMax-specific surfaces.

- basic model registration and weight loading
- MiniMax-specific MoE, QK norm, and tool-call integration exist
- do not confuse "supported" with "optimized"

- [#12129](https://github.com/sgl-project/sglang/pull/12129)

- `python/sglang/srt/models/minimax_m2.py` exists and is the active runtime path
- later performance or correctness work has a stable MiniMax-specific home

### Stage M2-1: Fix RMSNorm precision before chasing speed

MiniMax QK normalization is numerically sensitive. Before deeper optimization, the norm path must accumulate safely.

- prefer fp32 accumulation in the norm path
- treat QK norm correctness as a prerequisite for later TP work

- [#12186](https://github.com/sgl-project/sglang/pull/12186)

- the norm path no longer relies on lower-precision accumulation where MiniMax accuracy is sensitive

### Stage M2-2: Expose Eagle3 capture and embedding surfaces

MiniMax needs to expose the same capture surfaces as other spec-decoding-capable models. Without them, speculative or auxiliary-hidden-state features fail even if base generation works.

- capture intermediate hidden states for selected layers
- expose `get_embed_and_head`
- keep the speculative-decoding surface area on the MiniMax model, not on ad hoc wrappers

- [#12798](https://github.com/sgl-project/sglang/pull/12798)
- [#13297](https://github.com/sgl-project/sglang/pull/13297)

- `set_eagle3_layers_to_capture(...)` works
- `get_embed_and_head()` exists and downstream speculative code can call it

### Stage M2-3: Make the MoE path correct before making it faster

Before tuning kernels, MiniMax needs the right MoE contract. This includes correct DeepEP forward usage and removing unnecessary router-side work.

- keep the DeepEP forward path aligned with MiniMax's expert layout
- do not add shared-expert logic that MiniMax does not use
- remove unnecessary router work by specializing the top-k sigmoid path

- [#13892](https://github.com/sgl-project/sglang/pull/13892)
- [#14047](https://github.com/sgl-project/sglang/pull/14047)

- the DeepEP MiniMax MoE path is functionally correct
- the router no longer spends time on generic work MiniMax does not need

### Stage M2-4: Specialize the QK norm hot path

For MiniMax, QK normalization is a real decode hotspot. Once correctness is solid, the next gains come from fusing the TP-aware norm path instead of doing separate generic operations.

- compute Q and K norm together
- keep TP-aware reduction in the same specialized path
- preserve the custom all-reduce fast path by keeping reduction buffers aligned
- prefer the fused JIT TP QK norm path on supported CUDA launches instead of stopping at the older in-model RMSNormTP path
- check `SGLANG_USE_FUSED_PARALLEL_QKNORM`, `fused_parallel_qknorm(...)`, and `CustomAllReduceV2` before claiming a missing all-reduce optimization

- [#14416](https://github.com/sgl-project/sglang/pull/14416)
- [#16483](https://github.com/sgl-project/sglang/pull/16483)
- [#20673](https://github.com/sgl-project/sglang/pull/20673)

- `MiniMaxM2RMSNormTP` is the active per-layer QK norm implementation
- the reduction path consistently selects the fast aligned all-reduce path
- `MiniMaxM2QKRMSNorm` can use the fused TP QK norm custom op when the JIT path is enabled and supported
- focused validation exists in `python/sglang/jit_kernel/tests/test_tp_qknorm.py` and `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`

### Stage M2-5: Harden for piecewise CUDA graph and pipeline parallelism

Once the core hot paths are in place, MiniMax needs to remain usable under graph capture and PP partitioning.

- keep piecewise CUDA graph contexts correct around MoE expert-distribution recording
- propagate `pp_proxy_tensors`
- make weight loading layer-range aware under PP

- [#18217](https://github.com/sgl-project/sglang/pull/18217)
- [#19577](https://github.com/sgl-project/sglang/pull/19577)

Family-adjacent caveat:

- [#18310](https://github.com/sgl-project/sglang/pull/18310) is for MiniMax-M2.1 and focuses on a `torch.compile` plus CUDA-graph crash. It is not the core M2 mainline optimization ladder, but it is worth borrowing if graph tracing regresses on a MiniMax-family branch.

- MiniMax can run under PP without wrapper gaps
- piecewise CUDA graph support does not regress the MiniMax-specific path

## M2.5 Extension Path

Use this path when the target is `MiniMaxAI/MiniMax-M2.5` or another later MiniMax-family checkpoint. Start from the M2 core stages above, then continue here.

### Stage M25-0: Audit the loader contract already on `main`

M2.5 stresses loading and quantized checkpoint conventions much harder than the early M2 path.

- preserve `packed_modules_mapping`
- preserve KV-cache scale remapping
- keep ModelSlim-specific layer assumptions consistent with MiniMax layout

- [#19995](https://github.com/sgl-project/sglang/pull/19995)
- [#20870](https://github.com/sgl-project/sglang/pull/20870)
- [#20905](https://github.com/sgl-project/sglang/pull/20905)

- packed qkv and gate-up modules load correctly
- KV cache scales are not silently dropped
- ModelSlim quant layers do not assume a different MoE layout

### Stage M25-1: Fill the remaining quantized-loader gaps not yet in `main`

Status:
Tracked upstream PR work; not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

Some M2.5 quantized checkpoints use fused expert naming that the current mainline loader still does not fully cover.

- support fused expert mappings such as `w13`
- prefer explicit fused mapping before falling back to older `w1/w2/w3` logic
- add a focused weight-loading test when you port this work

- [#20031](https://github.com/sgl-project/sglang/pull/20031)

- AWQ or similar M2.5 checkpoints with fused expert weights load without local remapping hacks

### Stage M25-2: Make the scale-out runtime contract explicit

Status:
Partly on `main`, partly still tracked from upstream PR work as of `origin/main` commit `c122d343a` on `2026-04-21`.

For M2.5, the next bottleneck is often not a single kernel. It is the distributed contract across PP, EP, DP, and DeepEP.

- keep PP support from the mainline path
- make DeepEP runtime requirements explicit, especially hidden-size and dtype expectations
- treat DP support and DP-attention support as separate stages

- [#19577](https://github.com/sgl-project/sglang/pull/19577)
- [#17826](https://github.com/sgl-project/sglang/pull/17826)
- [#19468](https://github.com/sgl-project/sglang/pull/19468)

- PP launches correctly
- DeepEP no longer fails due to unsupported hidden size or dtype mismatch
- the runtime contract is written down for the exact TP / DP / EP / PP shape you care about

### Stage M25-3: Add the DP-attention and DEP communication optimizations

Status:
Mixed: [#20067](https://github.com/sgl-project/sglang/pull/20067) is part of `main`; [#20489](https://github.com/sgl-project/sglang/pull/20489) and [#20975](https://github.com/sgl-project/sglang/pull/20975) remain tracked as upstream PR work not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

This is the biggest M2.5 scale-out gap. Performance and correctness both depend on using the attention-TP group rather than blindly reusing the model-TP group.

- use attention TP group and rank instead of global TP group in MiniMax attention
- allow reduce-scatter after MoE when padding or DEP makes it profitable
- support FP4 all-gather when the communication path can quantize before transport
- allow all-reduce fusion between the MoE output and the next attention preparation
- guard zero-token and empty-batch paths

- [#20067](https://github.com/sgl-project/sglang/pull/20067)
- [#20489](https://github.com/sgl-project/sglang/pull/20489)
- [#20975](https://github.com/sgl-project/sglang/pull/20975)

- DP-attention MiniMax uses attention-TP metadata consistently
- DEP no longer performs an unnecessary all-reduce plus slice
- empty-batch or high-rank edge cases no longer crash

### Stage M25-4: Replace the older TP QK norm path with the fused JIT version

Status:
Mainline as [#20673](https://github.com/sgl-project/sglang/pull/20673) by `origin/main` commit `c122d343a` on `2026-04-21`.

The older QK norm path was already specialized; the newer mainline path pushes it further by moving to a fused JIT kernel that reuses custom all-reduce v2 more efficiently.

- fuse TP Q and K norm into one custom op
- keep a fallback path for unsupported environments
- add a dedicated benchmark and unit test with the PR

- [#20673](https://github.com/sgl-project/sglang/pull/20673)

- the MiniMax path can use the fused TP QK norm custom op
- the fallback path is still available when the JIT kernel cannot run

### Stage M25-5: Fix TP16 and replicated-KV-head correctness

Status:
Mainline as [#20967](https://github.com/sgl-project/sglang/pull/20967) by `origin/main` commit `c122d343a` on `2026-04-21`.

When `num_key_value_heads < tp_size`, multiple TP ranks can share the same KV head. That means the K norm weights and reductions must follow the replica layout, not a naive full-TP assumption.

- shard norm weights by logical head replica
- reduce only across ranks that share the same head
- do not assume the full TP group is the correct reduction group

- [#20967](https://github.com/sgl-project/sglang/pull/20967)

- high-TP MiniMax-M2.5 runs do not produce repeated or garbled output caused by incorrect K norm sharding

### Stage M25-6: Optional NVFP4 fallback for non-Blackwell GPUs

Status:
Mainline as [#19652](https://github.com/sgl-project/sglang/pull/19652) by `origin/main` commit `c122d343a` on `2026-04-21`; not MiniMax-specific, but directly relevant to some MiniMax-M2.5 deployments.

If the target checkpoint is an NVFP4 MiniMax variant on A100, H100, A40, or another non-Blackwell GPU, the real blocker may be the generic FP4 Marlin fallback rather than MiniMax model code.

- keep weights compressed in FP4
- route unsupported native FP4 cases to Marlin fallback
- preserve both linear and MoE fallback paths

- [#19652](https://github.com/sgl-project/sglang/pull/19652)

- NVFP4 MiniMax-family checkpoints can run coherently on non-Blackwell GPUs without decompression hacks

## M2.7 Current-Main Validation Path

Use this path when the target is `MiniMaxAI/MiniMax-M2.7`, `MiniMaxAI/MiniMax-M2.7-highspeed`, or when an AMD MiniMax change might affect the currently registered M2.7 lanes. Current main has explicit AMD accuracy and performance coverage for M2.7, while first-class M2.7 and M2.7-highspeed docs are tracked by upstream PR [#20873](https://github.com/sgl-project/sglang/pull/20873).

### Stage M27-0: Treat M2.7 as a separate later-family validation target

M2.7 currently reuses the MiniMax-M2-family runtime code, but the active registered tests are not just copies of M2.5. They launch `MiniMaxAI/MiniMax-M2.7` on AMD with TP8+EP8 and the aiter attention backend.

- keep the model-file assumptions from the M2/M2.5 ladder unless current code proves M2.7 has a new runtime path
- validate M2.7 separately from M2.5 on AMD when changing attention, MoE communication, loader, or aiter-related behavior
- use the registered M2.7 model path override `MINIMAX_M27_MODEL_PATH` for local mirrors
- preserve launch details from the current tests: `--tp 8`, `--ep-size 8`, `--attention-backend aiter`, `SGLANG_USE_AITER=1`, `--mem-fraction-static 0.85`, multithread loading, and a long watchdog timeout
- inspect both MI30x/MI325 and MI35x lanes because they use distinct registered suites

- `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py`
- `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py`
- `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py`
- `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py`
- `.github/workflows/nightly-test-amd.yml`
- `.github/workflows/nightly-test-amd-rocm720.yml`

- M2.7 accuracy and performance suites pass on the target AMD lane
- M2.5 and M2.7 failures are triaged independently
- docs do not become the only source of truth for M2.7 until a first-class M2.7 usage doc exists

## Open Optimization Items

Check these active upstream tracks before designing a new MiniMax skill or declaring a gap:

- [#22934](https://github.com/sgl-project/sglang/pull/22934): M2.5 EPLB bugfix for missing `routed_experts_weights_of_layer` on `MiniMaxM2ForCausalLM`.
- [#22744](https://github.com/sgl-project/sglang/pull/22744): NVIDIA `--enable-tf32-matmul` support aimed at reducing FP32 gate GEMM cost for MiniMax-M2.5 decode.
- [#22300](https://github.com/sgl-project/sglang/pull/22300): FP8 GEMM/deepgemm scale-format fix for fp16 MiniMax-M2.5 models.
- [#23301](https://github.com/sgl-project/sglang/pull/23301): token-by-token streaming of MiniMax-M2 string tool-call parameters.
- [#20873](https://github.com/sgl-project/sglang/pull/20873): docs-only M2.7 and M2.7-highspeed support.
- [#22432](https://github.com/sgl-project/sglang/pull/22432) closed unmerged on 2026-04-27 after exploring NPU split-QKV / TP RMSNorm / RoPE wiring for MiniMax2; [#23190](https://github.com/sgl-project/sglang/pull/23190) remains the follow-up to check for NPU Eagle3 and DP-attention fixes.
- [#17826](https://github.com/sgl-project/sglang/pull/17826), [#19468](https://github.com/sgl-project/sglang/pull/19468), [#20031](https://github.com/sgl-project/sglang/pull/20031), [#20489](https://github.com/sgl-project/sglang/pull/20489), and [#20975](https://github.com/sgl-project/sglang/pull/20975): remaining distributed, DeepEP, quant-loader, and DP-attention cleanup tracks carried from the earlier M2.5 radar.

## Investigation Order

When debugging a MiniMax issue, prefer this order:

1. classify the exact runtime shape
2. check whether the relevant optimization already exists on `main`
3. if not, check whether it lives in a still-open upstream PR
4. only then decide whether to port, reimplement, or defer it

For the supporting evidence and commands, use:

- [references/playbook.md](references/playbook.md)
- [references/pr-history.md](references/pr-history.md)

## Anti-Patterns

- Do not treat attention TP and model TP as automatically identical for MiniMax-M2.5.
- Do not optimize a generic MoE kernel first if the real problem is MiniMax loader or topology plumbing.
- Do not assume a TP-only text launch proves PP, DP, DP-attention, or DeepEP correctness.
- Do not bypass `packed_modules_mapping` or KV-scale remapping just to make one checkpoint load.
- Do not copy still-open upstream PR behavior into production without noting that it is not on `main` yet.
- Do not assume MiniMax-M2.7 is validated by passing only MiniMax-M2.5 tests; use the M2.7 AMD lanes when the change can affect that checkpoint.
- Do not omit `--tool-call-parser minimax-m2` or `--reasoning-parser minimax-append-think` when validating tool or reasoning behavior from the serving docs.
- Do not miss `SGLANG_USE_FUSED_PARALLEL_QKNORM` when benchmarking or diagnosing the current TP QK norm path.
