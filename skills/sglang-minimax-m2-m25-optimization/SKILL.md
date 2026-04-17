---
name: sglang-minimax-m2-m25-optimization
description: "PR-backed optimization manual for `MiniMaxAI/MiniMax-M2*` and `MiniMaxAI/MiniMax-M2.5*` in SGLang. Use when Codex needs to recover, extend, or audit MiniMax-specific optimizations, or when a structurally similar MoE model can reuse the same workflow: classify the serving shape, identify whether the code matches mainline or still-open upstream PR stages, apply the next missing optimization family, and validate on the exact topology, quant format, and backend."
---

# SGLang MiniMax M2/M2.5 Optimization

## Overview

The skill covers the full MiniMax optimization ladder — mainline history plus the remaining still-open upstream PR track. Use it to recover, extend, or audit MiniMax-specific optimizations, or to reuse the patterns on a structurally similar MoE model.

As of `2026-04-17`, the MiniMax story is split across two sources of truth:

- mainline history already present in `main`
- still-open upstream PRs that are important for MiniMax-M2.5, but not fully landed in `main` yet

This skill tracks both, but it labels them clearly. Do not assume an optimization from a PR page is already in your local tree.

The historical evidence for every stage lives in:

- [references/pr-history.md](references/pr-history.md): mainline and still-open PR evidence, benchmark notes, key code patterns
- [references/playbook.md](references/playbook.md): symptom mapping, commands, validation order

## Before You Change Anything

Record the exact serving shape first:

- M2, M2.1, or M2.5
- instruct or reasoning-style launch
- native, AWQ, FP8, FP4, ModelSlim, or other quant format
- TP / DP / EP / PP topology
- DP attention enabled or not
- DeepEP, FlashInfer, Triton, or other MoE / attention backend
- piecewise CUDA graph enabled or not
- speculative decoding or Eagle3 enabled or not
- NVIDIA, AMD, NPU, or other backend


- QK normalization depends on how heads are partitioned or replicated
- M2.5 scale-out performance depends on communication strategy, not only kernels
- quantized checkpoints depend on exact loader conventions

## Core Principle

Do not treat MiniMax as a generic DeepSeek-like MoE.

- MiniMax-M2 is a QK-normalized attention plus sparse-MoE story.
- MiniMax-M2.5 adds a much heavier distributed and quantized runtime story.
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


- [#14416](https://github.com/sgl-project/sglang/pull/14416)
- [#16483](https://github.com/sgl-project/sglang/pull/16483)


- `MiniMaxM2RMSNormTP` is the active per-layer QK norm implementation
- the reduction path consistently selects the fast aligned all-reduce path

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
Still-open upstream PR, not fully on `main` as of `2026-04-17`.

Some M2.5 quantized checkpoints use fused expert naming that the current mainline loader still does not fully cover.


- support fused expert mappings such as `w13`
- prefer explicit fused mapping before falling back to older `w1/w2/w3` logic
- add a focused weight-loading test when you port this work


- [#20031](https://github.com/sgl-project/sglang/pull/20031)


- AWQ or similar M2.5 checkpoints with fused expert weights load without local remapping hacks

### Stage M25-2: Make the scale-out runtime contract explicit

Status:
Partly on `main`, partly still-open upstream as of `2026-04-17`.

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
Mixed: [#20067](https://github.com/sgl-project/sglang/pull/20067) is part of `main`; [#20489](https://github.com/sgl-project/sglang/pull/20489) and [#20975](https://github.com/sgl-project/sglang/pull/20975) remain open as of `2026-04-17`.

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
Mainline as [#20673](https://github.com/sgl-project/sglang/pull/20673) by `2026-04-17`.

The older QK norm path was already specialized; the newer mainline path pushes it further by moving to a fused JIT kernel that reuses custom all-reduce v2 more efficiently.


- fuse TP Q and K norm into one custom op
- keep a fallback path for unsupported environments
- add a dedicated benchmark and unit test with the PR


- [#20673](https://github.com/sgl-project/sglang/pull/20673)


- the MiniMax path can use the fused TP QK norm custom op
- the fallback path is still available when the JIT kernel cannot run

### Stage M25-5: Fix TP16 and replicated-KV-head correctness

Status:
Mainline as [#20967](https://github.com/sgl-project/sglang/pull/20967) by `2026-04-17`.

When `num_key_value_heads < tp_size`, multiple TP ranks can share the same KV head. That means the K norm weights and reductions must follow the replica layout, not a naive full-TP assumption.


- shard norm weights by logical head replica
- reduce only across ranks that share the same head
- do not assume the full TP group is the correct reduction group


- [#20967](https://github.com/sgl-project/sglang/pull/20967)


- high-TP MiniMax-M2.5 runs do not produce repeated or garbled output caused by incorrect K norm sharding

### Stage M25-6: Optional NVFP4 fallback for non-Blackwell GPUs

Status:
Mainline as [#19652](https://github.com/sgl-project/sglang/pull/19652) by `2026-04-17`; not MiniMax-specific, but directly relevant to some MiniMax-M2.5 deployments.

If the target checkpoint is an NVFP4 MiniMax variant on A100, H100, A40, or another non-Blackwell GPU, the real blocker may be the generic FP4 Marlin fallback rather than MiniMax model code.


- keep weights compressed in FP4
- route unsupported native FP4 cases to Marlin fallback
- preserve both linear and MoE fallback paths


- [#19652](https://github.com/sgl-project/sglang/pull/19652)


- NVFP4 MiniMax-family checkpoints can run coherently on non-Blackwell GPUs without decompression hacks

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
