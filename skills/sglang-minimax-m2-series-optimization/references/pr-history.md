# MiniMax M2 Series Optimization History

This reference was built from local `git log --first-parent main`, local `git show`, and the PR pages in `sgl-project/sglang`.

As of SGLang `origin/main` commit `c122d343a` checked on `2026-04-21`, MiniMax optimization evidence falls into three buckets:

- mainline history already present in `main`
- still-open upstream PRs that are highly relevant to MiniMax-M2.5, but not fully landed in `main`
- current registered docs/tests/workflows that define MiniMax-M2.5, MiniMax-M2.7, and open M2.7-highspeed validation surfaces

This split matters. The still-open PRs are useful as an optimization manual, but they are not yet the same thing as shipped behavior. Current registered tests are also important: MiniMax-M2.7 has AMD lanes even though it currently reuses the MiniMax-M2-family runtime file.

Excluded on purpose for the historical PR ladder:

- parser-only and tool-call formatting fixes
- CI-only and docs-only changes
- cookbook redirects and other documentation reshuffles
- generic backend work that only happens to benchmark MiniMax unless it materially changes a MiniMax path

The current-main snapshot below is an exception to that exclusion rule. It captures docs, parsers, and CI lanes because they now tell you which validation shape to run before declaring a MiniMax change safe.

## Current Main Coverage Snapshot

Snapshot:
SGLang `origin/main` commit `c122d343a`, checked on `2026-04-21`.

Current serving docs and parser contract:

- `docs/basic_usage/minimax_m2.md` documents `MiniMaxAI/MiniMax-M2.5`, `MiniMaxAI/MiniMax-M2.1`, and `MiniMaxAI/MiniMax-M2`; the launch examples use both `--tool-call-parser minimax-m2` and `--reasoning-parser minimax-append-think`.
- `docs_new/docs/basic_usage/minimax_m2.mdx` currently documents M2.1/M2 and uses the same parser pair; do not use it as the only source of truth for M2.5 or M2.7.
- `python/sglang/srt/function_call/function_call_parser.py` maps `minimax-m2` to `MinimaxM2Detector`.
- `python/sglang/srt/parser/reasoning_parser.py` maps `minimax-append-think` to `MiniMaxAppendThinkDetector`; the detector prepends `<think>` to non-streaming and first streaming chunks.
- `test/registered/unit/parser/test_reasoning_parser.py` covers `MiniMaxAppendThinkDetector` and the `minimax-append-think` parser selection.

Current M2.5 registered coverage:

- `test/registered/8-gpu-models/test_minimax_m25.py` runs `MiniMaxAI/MiniMax-M2.5` with TP8+EP8 and TP8+DP8+EP8+DP-attention variants, using `--reasoning-parser=minimax-append-think`.
- `test/registered/8-gpu-models/test_minimax_m25_basic.py` is a lighter H200 lane for GSM8K and single-prompt speed, with multithread loading.
- `test/registered/amd/accuracy/mi30x/test_minimax_m25_eval_amd.py`, `test/registered/amd/perf/mi30x/test_minimax_m25_perf_amd.py`, `test/registered/amd/accuracy/mi35x/test_minimax_m25_eval_mi35x.py`, and `test/registered/amd/perf/mi35x/test_minimax_m25_perf_mi35x.py` keep M2.5 AMD accuracy/perf coverage available even though nightly workflow focus has shifted toward M2.7.
- `test/registered/ascend/llm_models/test_ascend_minimax_m2.py` validates MiniMax-M2 on Ascend with the Ascend attention backend.

Current M2.7 registered coverage:

- `test/registered/amd/accuracy/mi30x/test_minimax_m27_eval_amd.py` and `test/registered/amd/perf/mi30x/test_minimax_m27_perf_amd.py` run `MiniMaxAI/MiniMax-M2.7` on MI325/MI300X with TP8+EP8, `--attention-backend aiter`, `SGLANG_USE_AITER=1`, `--mem-fraction-static 0.85`, multithread loading, and long watchdog timeouts.
- `test/registered/amd/accuracy/mi35x/test_minimax_m27_eval_mi35x.py` and `test/registered/amd/perf/mi35x/test_minimax_m27_perf_mi35x.py` provide the matching MI35x M2.7 lanes.
- `.github/workflows/nightly-test-amd.yml` and `.github/workflows/nightly-test-amd-rocm720.yml` contain `nightly-8-gpu-minimax-m27` jobs that run M2.7 accuracy and performance suites and explicitly describe this as replacing the M2.5 combined nightly job.
- [#20873](https://github.com/sgl-project/sglang/pull/20873) is the active docs PR for M2.7 and M2.7-highspeed. Treat it as support-surface evidence, not as proof of a separate current-main runtime path.

Current TP QK norm and all-reduce implementation:

- [#16483](https://github.com/sgl-project/sglang/pull/16483) is already mainline and pads the RMSNormTP reduction buffer so `sglang::cross_device_reduce_1stage` stays on its aligned fast path.
- [#20673](https://github.com/sgl-project/sglang/pull/20673) is already mainline and adds `fused_parallel_qknorm(...)` in `python/sglang/jit_kernel/all_reduce.py`, backed by `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`.
- `python/sglang/srt/models/minimax_m2.py` now has `MiniMaxM2QKRMSNorm`, a `fused_tp_qknorm` custom op, and a fallback path. The fused path is gated by CUDA, world size, and `SGLANG_USE_FUSED_PARALLEL_QKNORM`.
- The fused path creates a `CustomAllReduceV2` object from the attention TP group, so current MiniMax TP norm debugging must inspect attention-TP metadata, not only the global TP group.
- Focused validation exists in `python/sglang/jit_kernel/tests/test_tp_qknorm.py` and `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`.

Current open PR radar:

- [#22934](https://github.com/sgl-project/sglang/pull/22934): MiniMax-M2.5 EPLB fix for missing `routed_experts_weights_of_layer` on `MiniMaxM2ForCausalLM`.
- [#22744](https://github.com/sgl-project/sglang/pull/22744): NVIDIA TF32 matmul flag for MiniMax gate GEMM; PR body reports FP32 gate GEMM dropping from `9.1%` to `3.3%` of e2e decode and batch-64 output throughput improving from `3076.99` to `3302.03 tok/s`.
- [#22300](https://github.com/sgl-project/sglang/pull/22300): FP8 GEMM performance and accuracy fix for fp16 MiniMax-M2.5 models through DeepGEMM scale-format handling.
- [#23301](https://github.com/sgl-project/sglang/pull/23301): MiniMax-M2 streaming parser change so string tool-call parameters stream token-by-token instead of buffering the full parameter.
- [#22432](https://github.com/sgl-project/sglang/pull/22432) and [#23190](https://github.com/sgl-project/sglang/pull/23190): Ascend/NPU split-QKV, TP RMSNorm, RoPE, Eagle3, and DP-attention MiniMax2 work.
- [#17826](https://github.com/sgl-project/sglang/pull/17826), [#19468](https://github.com/sgl-project/sglang/pull/19468), [#20031](https://github.com/sgl-project/sglang/pull/20031), [#20489](https://github.com/sgl-project/sglang/pull/20489), and [#20975](https://github.com/sgl-project/sglang/pull/20975) remain useful distributed, DeepEP, quant-loader, and DP-attention references.

Operational implication:

- Passing M2.5 does not prove M2.7 is healthy on AMD.
- Passing M2.7 does not remove the need for M2.5 loader or parser validation when the change is checkpoint- or docs-facing.
- Parser behavior is part of serving correctness even when a generation benchmark looks healthy.

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
- This is the "allreduce TP norm" optimization that predates the newer fused JIT path; do not look for it only under the M2.5 PRs.

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

This section was originally the open-upstream track. As of `origin/main` commit `c122d343a` checked on `2026-04-21`, several entries are now part of upstream `main`; the remaining entries are still useful as porting or gap-analysis references.

### [#17826](https://github.com/sgl-project/sglang/pull/17826) - PP and DP for MiniMax-M2

Status:
Tracked upstream PR work; not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

- Extends `minimax_m2.py` for PP and DP rather than PP alone.
- Adds attention-group-aware embedding and layer behavior.
- The PR body validates `TP2 + PP2 + DP2` on MiniMax-M2.1.

This is the upstream bridge from PP support toward a fuller MiniMax distributed contract.

### [#19468](https://github.com/sgl-project/sglang/pull/19468) - DeepEP support for MiniMax models

Status:
Tracked upstream PR work; not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

- Updates the DeepEP environment for MiniMax hidden size `3072`.
- Forces bf16 to satisfy DeepEP expectations.
- The key failures in the PR body are:
  - `Unsupported hidden`
  - DeepEP bf16 assertion failure

Some MiniMax-M2.5 DeepEP blockers are runtime-contract issues, not model-code issues.

### [#20031](https://github.com/sgl-project/sglang/pull/20031) - load fused expert weights such as `w13` for AWQ

Status:
Tracked upstream PR work; not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

- Extends `load_weights(...)` with fused expert mapping before the older mapping.
- Adds dedicated weight-loading test coverage in the PR branch.

Code focus:

- `FusedMoE.make_expert_params_mapping_fused(...)`
- try fused mapping first, then fall back to older expert mapping

M2.5 loader evolution is moving toward explicit fused-expert handling, not only generic expert-name remapping.

### [#20067](https://github.com/sgl-project/sglang/pull/20067) - DP attention, reduce-scatter, FP4 all-gather, and all-reduce fusion for MiniMax-M2.5

Status:
Mainline in upstream `main` as commit `7dbd0dd9f` by `origin/main` commit `c122d343a` on `2026-04-21`.

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
Tracked upstream PR work; not fully present in `origin/main` commit `c122d343a` as of `2026-04-21`.

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
Mainline in upstream `main` as commit `314d6ecf` by `origin/main` commit `c122d343a` on `2026-04-21`.

- Adds `python/sglang/jit_kernel/tests/test_tp_qknorm.py`.
- Adds `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`.
- Adds `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh` and `fused_parallel_qknorm(...)` in `python/sglang/jit_kernel/all_reduce.py`.
- Replaces the older in-model QK norm path with a fused JIT custom op when available.
- Reuses `CustomAllReduceV2` and is gated by `SGLANG_USE_FUSED_PARALLEL_QKNORM`, CUDA availability, and world size.
- Keeps the old `rms_sumsq_serial(...)` plus `attn_tp_all_reduce(...)` fallback for unsupported environments.

Representative benchmark note from the PR body:

- decode performance `150 tps -> 157 tps`

MiniMax QK norm optimization is still evolving, but the mainline direction is now a fused JIT op and custom-all-reduce integration rather than more Python-level reshaping.

### [#20967](https://github.com/sgl-project/sglang/pull/20967) - fix repeated output on MiniMax-M2.5 with `tp16`

Status:
Mainline in upstream `main` as commit `84194c25` by `origin/main` commit `c122d343a` on `2026-04-21`.

- Makes `MiniMaxM2RMSNormTP` replica-aware when KV heads are fewer than TP ranks.
- Changes the norm weight loader and reduction scope to follow logical KV-head replicas.

At high TP, MiniMax correctness depends on replica-aware norm logic, not only on total TP world size.

### [#19652](https://github.com/sgl-project/sglang/pull/19652) - NVFP4 Marlin fallback for non-Blackwell GPUs

Status:
Mainline in upstream `main` as commit `991f3aa5` by `origin/main` commit `c122d343a` on `2026-04-21`.

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
- current M2.7 AMD accuracy/performance validation lanes
- active M2.7-highspeed docs work
- current open PRs for EPLB, TF32 gate GEMM, FP8 GEMM, streaming parameters, and NPU split-QKV/RMSNorm/RoPE
- serving parser contracts that affect tool and reasoning output

What is intentionally not the main focus of this manual:

- parser-only MiniMax tool-call fixes unless they affect the current serving contract
- generic docs or CI unless they define the active M2.5/M2.7 validation surface
- generic MoE backend work unless it materially changes a MiniMax deployment path
