# MiniMax M2/M2.5 Optimization History

This reference was built from local `git log --first-parent main`, local `git show`, and the PR pages in `sgl-project/sglang`.

As of `2026-04-17`, MiniMax optimization evidence falls into two buckets:

- mainline history already present in `main`
- still-open upstream PRs that are highly relevant to MiniMax-M2.5, but not fully landed in `main`

This split matters. The still-open PRs are useful as an optimization manual, but they are not yet the same thing as shipped behavior. PRs that moved from active tracking into mainline are called out below so older porting notes do not make already-shipped features look missing.

Excluded on purpose:

- parser-only and tool-call formatting fixes
- CI-only and docs-only changes
- cookbook redirects and other documentation reshuffles
- generic backend work that only happens to benchmark MiniMax unless it materially changes a MiniMax path

## Mainline History

### `7ebc28f5d` / [#12129](https://github.com/sgl-project/sglang/pull/12129) - first MiniMax-M2 support

- Adds `python/sglang/srt/models/minimax_m2.py`.
- Adds MiniMax-specific tool-call plumbing in `python/sglang/srt/function_call/minimax_m2.py`.
- Establishes the base MiniMax model, MoE, attention, and loader structure that every later optimization builds on.

This is the bring-up point, not the optimized endpoint.

### `a8b91f6b2` / [#12186](https://github.com/sgl-project/sglang/pull/12186) - improve MiniMax RMSNorm precision

- Tightens the MiniMax norm path before deeper performance work.
- The important lesson is ordering: fix QK norm correctness before trying to accelerate it.

MiniMax QK norm is accuracy-sensitive enough that precision fixes belong near the start of the ladder.

### `f1a9c72de` / [#12798](https://github.com/sgl-project/sglang/pull/12798) - capture auxiliary hidden states for MiniMax

- Adds MiniMax support for capturing intermediate hidden states used by speculative decoding flows.
- Extends the model surface instead of making speculative code special-case MiniMax elsewhere.

Code focus:

- `self.layers_to_capture`
- `aux_hidden_states`
- capture-aware forward return path

For MiniMax, speculative or auxiliary-state support is part of the model contract, not an afterthought.

### `b051d76da` / [#13297](https://github.com/sgl-project/sglang/pull/13297) - add missing `get_embed_and_head`

- Exposes `get_embed_and_head()` on `MiniMaxM2ForCausalLM`.
- Closes a missing interface gap for Eagle3-style speculative logic.

MiniMax needs the same embedding and LM-head surface area as other spec-capable models.

### `e0e8a9963` / [#13892](https://github.com/sgl-project/sglang/pull/13892) - correct MiniMax DeepEP MoE forward usage

- Fixes the MiniMax DeepEP MoE forward path in `minimax_m2.py`.
- The PR is about correctness, not a new kernel.

Do not tune MiniMax DeepEP until the forward contract is correct.

### `3dabd609f` / [#14047](https://github.com/sgl-project/sglang/pull/14047) - optimize MiniMax top-k sigmoid

- Moves MiniMax away from a more generic top-k path.
- Reduces router-side overhead without inventing a MiniMax-specific CUDA op.
- Removes unnecessary work from the hottest router-side step first.

Files changed:

- `python/sglang/srt/layers/moe/topk.py`
- `python/sglang/srt/models/minimax_m2.py`

MiniMax followed the same pattern as other high-value MoE models: remove generic router work before deeper kernel work.

### `d17b9e639` / [#14416](https://github.com/sgl-project/sglang/pull/14416) - fuse MiniMax RMSNormTP

- Adds the fused TP-aware QK normalization structure inside `minimax_m2.py`.
- Introduces paired sum-of-squares and apply kernels for Q and K.
- Makes MiniMax QK norm a model-specific optimized path rather than a stack of generic ops.

Code focus:

- `rms_sumsq_serial(...)`
- `rms_apply_serial(...)`
- `MiniMaxM2RMSNormTP.forward_qk(...)`

MiniMax decode performance depends enough on QK norm that it justified a specialized TP-aware implementation.

### `486c7de39` / [#16483](https://github.com/sgl-project/sglang/pull/16483) - keep the RMSNormTP all-reduce on the fast path

- Pads the reduction buffer so `sglang::cross_device_reduce_1stage` consistently satisfies its alignment requirement.
- The PR body reports about a `6%` overall throughput improvement on MiniMax-M2.1.

Representative benchmark direction from the PR body:

- concurrency `1`: output throughput `116.87 -> 124.06`
- concurrency `16`: output throughput `640.00 -> 676.75`
- concurrency `64`: output throughput `1118.99 -> 1188.19`

The QK norm optimization story is not just arithmetic. Communication alignment is part of the hot path.

### `079fc8f3c` / [#18217](https://github.com/sgl-project/sglang/pull/18217) - piecewise CUDA graph support for MiniMax-M2

- Threads piecewise-graph-safe behavior through `minimax_m2.py`.
- Updates the fp8 kernel path at the same time.
- Makes graph capture a first-class consideration for MiniMax rather than a later patch.

For MiniMax, graph safety must be maintained alongside performance work.

### `2d183c4e6` / [#19577](https://github.com/sgl-project/sglang/pull/19577) - PP support for the MiniMax-M2 series

- Adds PP support to `minimax_m2.py`.
- Extends layer partitioning, PP proxy tensors, and missing-layer handling across MiniMax-M2, M2.1, and M2.5.

Representative accuracy note from the PR body on `MiniMax-M2.5`:

- PP-only run: accuracy `0.945`
- TP-only run: accuracy `0.940`

Pipeline support is part of the MiniMax runtime contract, not a wrapper-only concern.

### `df1d046de` / [#19995](https://github.com/sgl-project/sglang/pull/19995) - add `packed_modules_mapping`

- Adds `packed_modules_mapping` to `MiniMaxM2ForCausalLM`.
- Makes packed qkv and gate-up layouts explicit in the model definition.

Later quantized or packed checkpoints rely on this mapping being stable.

### `a3196d08b` / [#20870](https://github.com/sgl-project/sglang/pull/20870) - fix KV cache scale loading

- Extends the MiniMax loader path so KV cache scales are not silently lost.
- Works with the existing `maybe_remap_kv_scale_name(...)` pattern.

For quantized MiniMax-family checkpoints, loader details are part of optimization because a wrong scale load becomes a silent runtime regression.

### `1b4933d45` / [#20905](https://github.com/sgl-project/sglang/pull/20905) - adapt ModelSlim `w2` quant layer for MiniMax-M2.5

- Adjusts ModelSlim quant-layer handling for the MiniMax-M2.5 layout.
- Changes both `modelslim.py` and `minimax_m2.py`.

MiniMax-M2.5 quant support increasingly depends on model-specific loader assumptions, not only generic quant infrastructure.

## Mixed Mainline And Still-Open Upstream Track

This section was originally the open-upstream track. As of `2026-04-17`, several entries are now part of upstream `main`; the remaining open entries are still useful as porting or gap-analysis references.

### [#17826](https://github.com/sgl-project/sglang/pull/17826) - PP and DP for MiniMax-M2

Status:
Still open upstream as of `2026-04-17`.

- Extends `minimax_m2.py` for PP and DP rather than PP alone.
- Adds attention-group-aware embedding and layer behavior.
- The PR body validates `TP2 + PP2 + DP2` on MiniMax-M2.1.

This is the upstream bridge from PP support toward a fuller MiniMax distributed contract.

### [#19468](https://github.com/sgl-project/sglang/pull/19468) - DeepEP support for MiniMax models

Status:
Still open upstream as of `2026-04-17`.

- Updates the DeepEP environment for MiniMax hidden size `3072`.
- Forces bf16 to satisfy DeepEP expectations.
- The key failures in the PR body are:
  - `Unsupported hidden`
  - DeepEP bf16 assertion failure

Some MiniMax-M2.5 DeepEP blockers are runtime-contract issues, not model-code issues.

### [#20031](https://github.com/sgl-project/sglang/pull/20031) - load fused expert weights such as `w13` for AWQ

Status:
Still open upstream as of `2026-04-17`.

- Extends `load_weights(...)` with fused expert mapping before the older mapping.
- Adds dedicated weight-loading test coverage in the PR branch.

Code focus:

- `FusedMoE.make_expert_params_mapping_fused(...)`
- try fused mapping first, then fall back to older expert mapping

M2.5 loader evolution is moving toward explicit fused-expert handling, not only generic expert-name remapping.

### [#20067](https://github.com/sgl-project/sglang/pull/20067) - DP attention, reduce-scatter, FP4 all-gather, and all-reduce fusion for MiniMax-M2.5

Status:
Mainline in upstream `main` as commit `7dbd0dd9f` by `2026-04-17`.

- Switches MiniMax attention and norms to attention-TP-aware metadata.
- Allows post-MoE communication to avoid unconditional all-reduce.
- Adds hooks for:
  - DP attention
  - reduce-scatter after MoE
  - FP4 all-gather when supported
  - all-reduce fusion into the next layer

Representative accuracy and throughput notes from the PR body:

- FP4 `DEP4` with FP4 all-gather: accuracy `0.959`, output throughput `6245.561 token/s`
- FP4 `DEP4` with bf16 all-gather: accuracy `0.948`, output throughput `5914.209 token/s`
- FP4 `TP4` with all-reduce fusion: accuracy `0.948`, output throughput `3559.490 token/s`

This is now the baseline M2.5 scale-out optimization track to verify before porting newer DP-attention cleanup PRs.

### [#20489](https://github.com/sgl-project/sglang/pull/20489) and [#20975](https://github.com/sgl-project/sglang/pull/20975) - DP-attention fixes for MiniMax M2

Status:
Still open upstream as of `2026-04-17`.

- Fix attention TP group usage inside MiniMax attention.
- Fix model-runner and memory-pool assumptions that break at higher DP-attention ranks.
- Add empty-batch-safe rotary behavior.

Files touched across the two PRs:

- `python/sglang/srt/models/minimax_m2.py`
- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/layers/rotary_embedding/base.py`
- `python/sglang/srt/layers/dp_attention.py`

The M2.5 DP-attention path is not just a model-file change. The runtime plumbing matters too.

### [#20673](https://github.com/sgl-project/sglang/pull/20673) - fused TP QK norm JIT kernel for MiniMax

Status:
Mainline in upstream `main` as commit `314d6ecf` by `2026-04-17`.

- Adds `python/sglang/jit_kernel/tests/test_tp_qknorm.py`.
- Adds `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`.
- Replaces the older in-model QK norm path with a fused JIT custom op when available.

Representative benchmark note from the PR body:

- decode performance `150 tps -> 157 tps`

MiniMax QK norm optimization is still evolving, but the mainline direction is now a fused JIT op rather than more Python-level reshaping.

### [#20967](https://github.com/sgl-project/sglang/pull/20967) - fix repeated output on MiniMax-M2.5 with `tp16`

Status:
Mainline in upstream `main` as commit `84194c25` by `2026-04-17`.

- Makes `MiniMaxM2RMSNormTP` replica-aware when KV heads are fewer than TP ranks.
- Changes the norm weight loader and reduction scope to follow logical KV-head replicas.

At high TP, MiniMax correctness depends on replica-aware norm logic, not only on total TP world size.

### [#19652](https://github.com/sgl-project/sglang/pull/19652) - NVFP4 Marlin fallback for non-Blackwell GPUs

Status:
Mainline in upstream `main` as commit `991f3aa5` by `2026-04-17`.

- Not MiniMax-specific in code ownership, but directly relevant to MiniMax-M2.5 NVFP4 deployments.
- The PR body explicitly names `mistralai/Minimax-M2.5-NVFP4` as a motivating example.
- Keeps FP4 weights compressed and routes unsupported native FP4 paths through Marlin fallback for both linear and MoE paths.

For non-Blackwell MiniMax-M2.5 NVFP4 debugging, first validate the mainline fallback path in the current tree before assuming a MiniMax model-file gap.

Some MiniMax-M2.5 deployment blockers belong to the generic FP4 runtime layer rather than the MiniMax model file itself.

## Coverage Summary

If you are trying to understand "what is already comprehensive here", the MiniMax optimization manual covers these families:

- base MiniMax model bring-up
- MiniMax-specific QK norm correctness and performance
- MiniMax-specific MoE and router-side cleanup
- Eagle3 and auxiliary-hidden-state surfaces
- piecewise CUDA graph and PP support
- packed or quantized checkpoint loader contracts
- mainline M2.5 scale-out work for DP attention, fused QK norm, high-TP correctness, and remaining still-open DP / DeepEP gaps

What is intentionally not the main focus of this manual:

- parser-only MiniMax tool-call fixes
- generic docs or CI
- generic MoE backend work unless it materially changes a MiniMax deployment path
