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

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG MiniMax M2 series PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2025-10-25 | [#12129](https://github.com/sgl-project/sglang/pull/12129) | merged | Support MiniMax M2 model | model wrapper, docs/config | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2025-10-27 | [#12186](https://github.com/sgl-project/sglang/pull/12186) | merged | improve mimax-m2 rmsnorm precision | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-07 | [#12798](https://github.com/sgl-project/sglang/pull/12798) | merged | Support capturing aux_hidden_states for minimax m2. | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-14 | [#13297](https://github.com/sgl-project/sglang/pull/13297) | merged | Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-25 | [#13892](https://github.com/sgl-project/sglang/pull/13892) | merged | fix: correct usage of minimax-m2 deepep moe forward | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2025-11-27 | [#14047](https://github.com/sgl-project/sglang/pull/14047) | merged | Optimize topk sigmoid in minimax_m2 | model wrapper, MoE/router | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2025-12-04 | [#14416](https://github.com/sgl-project/sglang/pull/14416) | merged | Fusing RMSNormTP in minimax_m2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-05 | [#16483](https://github.com/sgl-project/sglang/pull/16483) | merged | Optimizing all_reduce in RMSNormTP in minimax_m2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-01-27 | [#17826](https://github.com/sgl-project/sglang/pull/17826) | open | Support Pipeline and Data Parallelism for MiniMax-M2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-02-04 | [#18217](https://github.com/sgl-project/sglang/pull/18217) | merged | [piecewise graph]: support MiniMax-M2 | model wrapper, quantization, kernel | `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2026-02-27 | [#19468](https://github.com/sgl-project/sglang/pull/19468) | open | fix[minimax]: support deepep with minimax models | kernel | `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh` |
| 2026-02-28 | [#19577](https://github.com/sgl-project/sglang/pull/19577) | merged | [Feat] add PP Support for minimax-m2 series | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-02 | [#19652](https://github.com/sgl-project/sglang/pull/19652) | merged | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py` |
| 2026-03-06 | [#19995](https://github.com/sgl-project/sglang/pull/19995) | merged | Add packed_modules_mapping for MiniMax-M2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-06 | [#20031](https://github.com/sgl-project/sglang/pull/20031) | open | fix(minimax): support loading merged expert weights (w13) for awq | model wrapper, tests/benchmarks | `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-07 | [#20067](https://github.com/sgl-project/sglang/pull/20067) | merged | MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn | model wrapper, tests/benchmarks | `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py` |
| 2026-03-13 | [#20489](https://github.com/sgl-project/sglang/pull/20489) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general… | model wrapper, scheduler/runtime | `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-03-16 | [#20673](https://github.com/sgl-project/sglang/pull/20673) | merged | [Feature][JIT Kernel] Fused TP QK norm For Minimax | model wrapper, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py` |
| 2026-03-18 | [#20870](https://github.com/sgl-project/sglang/pull/20870) | merged | [MiniMax M2] Fix KV cache scale loading | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-18 | [#20873](https://github.com/sgl-project/sglang/pull/20873) | open | docs: add MiniMax-M2.7 and M2.7-highspeed model support | model wrapper, docs/config | `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md` |
| 2026-03-19 | [#20905](https://github.com/sgl-project/sglang/pull/20905) | merged | [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5 | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20967](https://github.com/sgl-project/sglang/pull/20967) | merged | 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-03-20 | [#20975](https://github.com/sgl-project/sglang/pull/20975) | open | fix(dp-attn): fix issues with dp-attention for MiniMax M2 | model wrapper, attention/backend, scheduler/runtime | `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2026-04-08 | [#22300](https://github.com/sgl-project/sglang/pull/22300) | open | [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5) | quantization | `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2026-04-09 | [#22432](https://github.com/sgl-project/sglang/pull/22432) | open | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-14 | [#22744](https://github.com/sgl-project/sglang/pull/22744) | open | [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance | scheduler/runtime, docs/config | `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md` |
| 2026-04-16 | [#22934](https://github.com/sgl-project/sglang/pull/22934) | open | Minimax eplb bugfix | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-20 | [#23190](https://github.com/sgl-project/sglang/pull/23190) | open | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode | model wrapper | `python/sglang/srt/models/minimax_m2.py` |
| 2026-04-21 | [#23301](https://github.com/sgl-project/sglang/pull/23301) | open | [sgl] Stream MiniMax M2 string parameters token-by-token | misc | `python/sglang/srt/function_call/minimax_m2.py` |

## Diff Cards

### PR #12129 - Support MiniMax M2 model

- Link: https://github.com/sgl-project/sglang/pull/12129
- Status/date: `merged`, created 2025-10-25, merged 2025-10-26; author `zhaochenyang20`.
- Diff scope read: `5` files, `+1320/-1`; areas: model wrapper, docs/config; keywords: expert, moe, spec, attention, config, deepep, doc, fp8, kv, processor.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` added +922/-0 (922 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, forward
  - `python/sglang/srt/function_call/minimax_m2.py` added +367/-0 (367 lines); hunks: +import ast; symbols: _safe_val, MinimaxM2Detector, __init__, has_tool_call
  - `python/sglang/srt/parser/reasoning_parser.py` modified +28/-1 (29 lines); hunks: def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:; class ReasoningParser:; symbols: parse_streaming_increment, MiniMaxAppendThinkDetector, __init__, parse_streaming_increment
  - `python/sglang/srt/function_call/function_call_parser.py` modified +2/-0 (2 lines); hunks: from sglang.srt.function_call.gpt_oss_detector import GptOssDetector; class FunctionCallParser:; symbols: FunctionCallParser:, __init__
  - `docs/supported_models/generative_models.md` modified +1/-0 (1 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py`; keywords observed in patches: expert, moe, spec, attention, config, deepep. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/function_call/minimax_m2.py`, `python/sglang/srt/parser/reasoning_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12186 - improve mimax-m2 rmsnorm precision

- Link: https://github.com/sgl-project/sglang/pull/12186
- Status/date: `merged`, created 2025-10-27, merged 2025-10-27; author `haichao592`.
- Diff scope read: `1` files, `+1/-1`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12798 - Support capturing aux_hidden_states for minimax m2.

- Link: https://github.com/sgl-project/sglang/pull/12798
- Status/date: `merged`, created 2025-11-07, merged 2025-11-08; author `pyc96`.
- Diff scope read: `1` files, `+34/-3`; areas: model wrapper; keywords: config, eagle, expert, processor, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-3 (37 lines); hunks: def layer_fn(idx, prefix: str) -> nn.Module:; def forward(; symbols: layer_fn, get_input_embeddings, forward, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, eagle, expert, processor, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13297 - Fix: add missing get_embed_and_head in MiniMax M2 for Eagle3

- Link: https://github.com/sgl-project/sglang/pull/13297
- Status/date: `merged`, created 2025-11-14, merged 2025-11-15; author `pyc96`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper; keywords: eagle.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-0 (3 lines); hunks: def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):; symbols: set_eagle3_layers_to_capture, get_embed_and_head, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13892 - fix: correct usage of minimax-m2 deepep moe forward

- Link: https://github.com/sgl-project/sglang/pull/13892
- Status/date: `merged`, created 2025-11-25, merged 2025-11-26; author `yuukidach`.
- Diff scope read: `1` files, `+3/-7`; areas: model wrapper; keywords: deepep, expert, router, topk.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +3/-7 (10 lines); hunks: def forward_deepep(; def forward_deepep(; symbols: forward_deepep, forward_deepep
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: deepep, expert, router, topk. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14047 - Optimize topk sigmoid in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14047
- Status/date: `merged`, created 2025-11-27, merged 2025-12-02; author `rogeryoungh`.
- Diff scope read: `2` files, `+38/-13`; areas: model wrapper, MoE/router; keywords: config, expert, topk, cuda, moe, router.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +38/-10 (48 lines); hunks: ); pass; symbols: TopKConfig:, __init__, forward_native, fused_topk_torch_native
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-3 (3 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, expert, topk, cuda, moe, router. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14416 - Fusing RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/14416
- Status/date: `merged`, created 2025-12-04, merged 2025-12-30; author `rogeryoungh`.
- Diff scope read: `1` files, `+189/-2`; areas: model wrapper; keywords: config, cuda, deepep, expert, kv, moe, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +189/-2 (191 lines); hunks: from typing import Iterable, Optional, Set, Tuple, Union; logger = logging.getLogger(__name__); symbols: rmsnorm_sumsq_kernel_serial, rmsnorm_apply_kernel_serial, rms_sumsq_serial, rms_apply_serial
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, cuda, deepep, expert, kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #16483 - Optimizing all_reduce in RMSNormTP in minimax_m2

- Link: https://github.com/sgl-project/sglang/pull/16483
- Status/date: `merged`, created 2026-01-05, merged 2026-02-01; author `rogeryoungh`.
- Diff scope read: `1` files, `+8/-2`; areas: model wrapper; keywords: triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-2 (10 lines); hunks: def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:; def forward(; symbols: rms_sumsq_serial, forward, forward_qk
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17826 - Support Pipeline and Data Parallelism for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/17826
- Status/date: `open`, created 2026-01-27; author `rogeryoungh`.
- Diff scope read: `1` files, `+167/-70`; areas: model wrapper; keywords: attention, config, cuda, deepep, eagle, expert, kv, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +167/-70 (237 lines); hunks: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; from sglang.srt.distributed import (; symbols: MiniMaxM2RMSNormTP, __init__, weight_loader, ebias_weight_loader
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, cuda, deepep, eagle, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18217 - [piecewise graph]: support MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/18217
- Status/date: `merged`, created 2026-02-04, merged 2026-02-05; author `hzh0425`.
- Diff scope read: `2` files, `+28/-7`; areas: model wrapper, quantization, kernel; keywords: config, cuda, deepep, expert, fp8, quant, router, topk.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +23/-7 (30 lines); hunks: """Inference-only MiniMax M2 model compatible with HuggingFace weights."""; def op_select_experts(self, state):; symbols: op_select_experts, op_dispatch_a, op_dispatch_b, forward
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +5/-0 (5 lines); hunks: def get_w8a8_block_fp8_configs(; symbols: get_w8a8_block_fp8_configs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; keywords observed in patches: config, cuda, deepep, expert, fp8, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19468 - fix[minimax]: support deepep with minimax models

- Link: https://github.com/sgl-project/sglang/pull/19468
- Status/date: `open`, created 2026-02-27; author `ishandhanani`.
- Diff scope read: `3` files, `+10/-2`; areas: kernel; keywords: deepep, config, cuda, doc, flash, moe, spec.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: def _handle_model_specific_adjustments(self):; symbols: _handle_model_specific_adjustments
  - `docker/Dockerfile` modified +2/-1 (3 lines); hunks: ARG HOPPER_SBO=0
  - `scripts/ci/cuda/ci_install_deepep.sh` modified +2/-1 (3 lines); hunks: if [ "$GRACE_BLACKWELL" = "1" ]; then
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`; keywords observed in patches: deepep, config, cuda, doc, flash, moe. Impact reading: CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `docker/Dockerfile`, `scripts/ci/cuda/ci_install_deepep.sh`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19577 - [Feat] add PP Support for minimax-m2 series

- Link: https://github.com/sgl-project/sglang/pull/19577
- Status/date: `merged`, created 2026-02-28, merged 2026-03-02; author `LuYanFCP`.
- Diff scope read: `1` files, `+35/-7`; areas: model wrapper; keywords: attention, config, eagle, processor, quant, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +35/-7 (42 lines); hunks: from sglang.srt.layers.quantization.base_config import QuantizationConfig; def __init__(; symbols: __init__, forward, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, eagle, processor, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19652 - [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+)

- Link: https://github.com/sgl-project/sglang/pull/19652
- Status/date: `merged`, created 2026-03-02, merged 2026-04-03; author `Godmook`.
- Diff scope read: `16` files, `+1410/-95`; areas: MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: fp4, marlin, quant, fp8, moe, expert, config, flash, topk, triton.
- Code diff details:
  - `test/registered/quant/test_nvfp4_marlin_fallback.py` added +788/-0 (788 lines); hunks: +"""Tests for NVFP4 Marlin fallback on non-Blackwell GPUs (SM75+)."""; symbols: _check_requirements, _dequant_fp4_weights, _FakeLayer, TestNvfp4MarlinLinear
  - `python/sglang/srt/layers/quantization/marlin_utils_fp4.py` added +320/-0 (320 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: is_fp4_marlin_supported, should_use_fp4_marlin_fallback, nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +82/-7 (89 lines); hunks: is_blackwell_supported,; def get_supported_act_dtypes(cls) -> List[torch.dtype]:; symbols: get_supported_act_dtypes, get_min_capability, common_group_size, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py` modified +66/-8 (74 lines); hunks: CompressedTensorsMoEScheme,; class CompressedTensorsW4A4Nvfp4MoE(CompressedTensorsMoEScheme):; symbols: CompressedTensorsW4A4Nvfp4MoE, __init__, get_min_capability, create_weights
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` modified +21/-32 (53 lines); hunks: __global__ void Marlin(; __global__ void Marlin(; symbols: void, int, int, int
- Optimization/support interpretation: The concrete diff surface is `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; keywords observed in patches: fp4, marlin, quant, fp8, moe, expert. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/registered/quant/test_nvfp4_marlin_fallback.py`, `python/sglang/srt/layers/quantization/marlin_utils_fp4.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19995 - Add packed_modules_mapping for MiniMax-M2

- Link: https://github.com/sgl-project/sglang/pull/19995
- Status/date: `merged`, created 2026-03-06, merged 2026-03-18; author `trevor-m`.
- Diff scope read: `1` files, `+12/-0`; areas: model wrapper; keywords: config, kv.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +12/-0 (12 lines); hunks: def forward(; symbols: forward, MiniMaxM2ForCausalLM, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20031 - fix(minimax): support loading merged expert weights (w13) for awq

- Link: https://github.com/sgl-project/sglang/pull/20031
- Status/date: `open`, created 2026-03-06; author `xueliangyang-oeuler`.
- Diff scope read: `2` files, `+203/-9`; areas: model wrapper, tests/benchmarks; keywords: config, expert, moe, spec, attention, processor, quant, test.
- Code diff details:
  - `tests/registered/models/test_minimax_m2_weights.py` added +145/-0 (145 lines); hunks: +import unittest; symbols: TestMiniMaxM2WeightLoading, setUp, test_load_weights_merged_w13
  - `python/sglang/srt/models/minimax_m2.py` modified +58/-9 (67 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights, load_weights, load_weights
- Optimization/support interpretation: The concrete diff surface is `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, expert, moe, spec, attention, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/registered/models/test_minimax_m2_weights.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20067 - MiniMax-M2.5 - Support dp attention, dp reduce scatter, FP4 all gather, AR fusion in prepare_attn

- Link: https://github.com/sgl-project/sglang/pull/20067
- Status/date: `merged`, created 2026-03-07, merged 2026-04-10; author `trevor-m`.
- Diff scope read: `3` files, `+39/-6`; areas: model wrapper, tests/benchmarks; keywords: attention, config, cuda, expert, flash, fp4, kv, moe, processor, quant.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-6 (31 lines); hunks: RowParallelLinear,; def forward_normal(; symbols: forward_normal, forward_prepare, forward_prepare, forward_core
  - `test/registered/8-gpu-models/test_minimax_m25.py` modified +10/-0 (10 lines); hunks: def test_minimax_m25(self):; def test_minimax_m25(self):; symbols: test_minimax_m25, test_minimax_m25
  - `python/sglang/srt/layers/layernorm.py` modified +4/-0 (4 lines); hunks: def forward_cuda(; symbols: forward_cuda
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py`; keywords observed in patches: attention, config, cuda, expert, flash, fp4. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`, `test/registered/8-gpu-models/test_minimax_m25.py`, `python/sglang/srt/layers/layernorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20489 - fix(dp-attn): fix issues with dp-attention for MiniMax M2 and general…

- Link: https://github.com/sgl-project/sglang/pull/20489
- Status/date: `open`, created 2026-03-13; author `xueliangyang-oeuler`.
- Diff scope read: `5` files, `+118/-20`; areas: model wrapper, scheduler/runtime; keywords: attention, config, cuda, kv, cache, expert, moe, quant, test.
- Code diff details:
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo; def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
  - `python/sglang/srt/layers/rotary_embedding/base.py` modified +2/-0 (2 lines); hunks: def forward_cuda(; symbols: forward_cuda
- Optimization/support interpretation: The concrete diff surface is `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, config, cuda, kv, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20673 - [Feature][JIT Kernel] Fused TP QK norm For Minimax

- Link: https://github.com/sgl-project/sglang/pull/20673
- Status/date: `merged`, created 2026-03-16, merged 2026-04-13; author `DarkSharpness`.
- Diff scope read: `11` files, `+923/-82`; areas: model wrapper, kernel, tests/benchmarks; keywords: cuda, config, test, cache, kv, processor, spec, triton, attention, benchmark.
- Code diff details:
  - `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh` added +325/-0 (325 lines); hunks: +// Adapted from https://github.com/NVIDIA/TensorRT-LLM/pull/12163; symbols: ParallelQKNormParams, auto, KernelTrait, parameters
  - `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py` added +170/-0 (170 lines); hunks: +from __future__ import annotations; symbols: parse_args, init_distributed, bench_one, rmsnorm_baseline
  - `python/sglang/jit_kernel/tests/test_tp_qknorm.py` added +168/-0 (168 lines); hunks: +from __future__ import annotations; symbols: test_tp_qknorm, init_distributed, _all_gather_cat, _rmsnorm_ref
  - `python/sglang/srt/models/minimax_m2.py` modified +113/-21 (134 lines); hunks: import logging; ); symbols: forward, fused_tp_qknorm, MiniMaxM2QKRMSNorm:, __init__
  - `python/sglang/jit_kernel/all_reduce.py` modified +50/-6 (56 lines); hunks: import torch; def config_pull(; symbols: config_pull, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module, _jit_custom_all_reduce_pull_module
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`; keywords observed in patches: cuda, config, test, cache, kv, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/csrc/distributed/tp_qknorm.cuh`, `python/sglang/jit_kernel/benchmark/bench_tp_qknorm.py`, `python/sglang/jit_kernel/tests/test_tp_qknorm.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20870 - [MiniMax M2] Fix KV cache scale loading

- Link: https://github.com/sgl-project/sglang/pull/20870
- Status/date: `merged`, created 2026-03-18, merged 2026-03-18; author `chadvoegele`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper; keywords: cache, expert, kv, spec.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +8/-0 (8 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: cache, expert, kv, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20873 - docs: add MiniMax-M2.7 and M2.7-highspeed model support

- Link: https://github.com/sgl-project/sglang/pull/20873
- Status/date: `open`, created 2026-03-18; author `octo-patch`.
- Diff scope read: `2` files, `+15/-3`; areas: model wrapper, docs/config; keywords: doc, moe, expert, test.
- Code diff details:
  - `docs/basic_usage/minimax_m2.md` modified +14/-2 (16 lines); hunks: -# MiniMax M2.5/M2.1/M2 Usage; curl http://localhost:8000/v1/chat/completions \
  - `docs/supported_models/text_generation/generative_models.md` modified +1/-1 (2 lines); hunks: in the GitHub search bar.
- Optimization/support interpretation: The concrete diff surface is `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; keywords observed in patches: doc, moe, expert, test. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `docs/basic_usage/minimax_m2.md`, `docs/supported_models/text_generation/generative_models.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20905 - [NPU][ModelSlim] adapt w2 quant layer for Minimax2.5

- Link: https://github.com/sgl-project/sglang/pull/20905
- Status/date: `merged`, created 2026-03-19, merged 2026-03-24; author `shadowxz109`.
- Diff scope read: `2` files, `+22/-30`; areas: model wrapper, quantization; keywords: config, moe, quant.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelslim/modelslim.py` modified +21/-29 (50 lines); hunks: def get_moe_scheme(; symbols: get_moe_scheme, is_layer_skipped
  - `python/sglang/srt/models/minimax_m2.py` modified +1/-1 (2 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: config, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/quantization/modelslim/modelslim.py`, `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20967 - 【BugFix】fix the bug of minimax_m2.5 model that causes repeated outputs when using tp16

- Link: https://github.com/sgl-project/sglang/pull/20967
- Status/date: `merged`, created 2026-03-20, merged 2026-04-10; author `kingkingleeljj`.
- Diff scope read: `1` files, `+34/-10`; areas: model wrapper; keywords: attention, config, kv.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +34/-10 (44 lines); hunks: def rms_apply_serial(; def __init__(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20975 - fix(dp-attn): fix issues with dp-attention for MiniMax M2

- Link: https://github.com/sgl-project/sglang/pull/20975
- Status/date: `open`, created 2026-03-20; author `xueliangyang-oeuler`.
- Diff scope read: `6` files, `+122/-20`; areas: model wrapper, attention/backend, scheduler/runtime; keywords: attention, config, cuda, kv, cache, expert, moe, quant, test.
- Code diff details:
  - `PR_DESCRIPTION.md` added +78/-0 (78 lines); hunks: +## PR Motivation
  - `python/sglang/srt/models/minimax_m2.py` modified +33/-16 (49 lines); hunks: from sglang.kernel_api_logging import debug_kernel_api; def rms_apply_serial(; symbols: rms_apply_serial, MiniMaxM2RMSNormTP, __init__, __init__
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-2 (5 lines); hunks: def _set_kv_buffer_impl(; symbols: _set_kv_buffer_impl
  - `python/sglang/srt/layers/dp_attention.py` modified +4/-0 (4 lines); hunks: def get_attention_tp_size() -> int:; symbols: get_attention_tp_size, get_attention_tp_world_size, get_attention_cp_group
  - `python/sglang/srt/model_executor/model_runner.py` modified +2/-2 (4 lines); hunks: def _dummy_run(self, batch_size: int, run_ctx=None):; symbols: _dummy_run
- Optimization/support interpretation: The concrete diff surface is `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; keywords observed in patches: attention, config, cuda, kv, cache, expert. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `PR_DESCRIPTION.md`, `python/sglang/srt/models/minimax_m2.py`, `python/sglang/srt/mem_cache/memory_pool.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22300 - [NVIDIA] Fix FP8 gemm performance with fp16 models (MInimax-M2.5)

- Link: https://github.com/sgl-project/sglang/pull/22300
- Status/date: `open`, created 2026-04-08; author `trevor-m`.
- Diff scope read: `3` files, `+30/-6`; areas: quantization; keywords: fp8, quant, triton, config, flash.
- Code diff details:
  - `python/sglang/srt/model_loader/utils.py` modified +20/-4 (24 lines); hunks: def post_load_weights(model: nn.Module, model_config: ModelConfig):; symbols: post_load_weights, should_deepgemm_weight_requant_ue8m0, should_deepgemm_weight_requant_ue8m0, should_async_load
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +5/-2 (7 lines); hunks: def flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(; symbols: flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
  - `python/sglang/srt/layers/quantization/fp8.py` modified +5/-0 (5 lines); hunks: def process_weights_after_loading_block_quant(self, layer: Module) -> None:; symbols: process_weights_after_loading_block_quant
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`; keywords observed in patches: fp8, quant, triton, config, flash. Impact reading: quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/model_loader/utils.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/layers/quantization/fp8.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22432 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2

- Link: https://github.com/sgl-project/sglang/pull/22432
- Status/date: `open`, created 2026-04-09; author `shadowxz109`.
- Diff scope read: `1` files, `+69/-11`; areas: model wrapper; keywords: attention, cache, config, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +69/-11 (80 lines); hunks: import logging; ); symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, cache, config, expert, kv, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22744 - [NVIDIA] Support TF32 matmul to improve MiniMax gate gemm performance

- Link: https://github.com/sgl-project/sglang/pull/22744
- Status/date: `open`, created 2026-04-14; author `trevor-m`.
- Diff scope read: `3` files, `+11/-0`; areas: scheduler/runtime, docs/config; keywords: moe, cache, doc, fp8, kv.
- Code diff details:
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunks: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; symbols: ServerArgs:, add_cli_args
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunks: def __init__(; symbols: __init__
  - `docs/advanced_features/server_arguments.md` modified +1/-0 (1 lines); hunks: Please consult the documentation below and [server_args.py](https://github.com/s
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md`; keywords observed in patches: moe, cache, doc, fp8, kv. Impact reading: scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`, `docs/advanced_features/server_arguments.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22934 - Minimax eplb bugfix

- Link: https://github.com/sgl-project/sglang/pull/22934
- Status/date: `open`, created 2026-04-16; author `DaZhUUU`.
- Diff scope read: `1` files, `+25/-0`; areas: model wrapper; keywords: attention, config, eagle, expert, moe, quant, topk, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +25/-0 (25 lines); hunks: from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class; # Other files (custom_all_reduce.py, hf_transformers_utils.py) also use sglang.srt.utils.; symbols: op_output, get_moe_weights, MiniMaxM2Attention, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, config, eagle, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23190 - [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode

- Link: https://github.com/sgl-project/sglang/pull/23190
- Status/date: `open`, created 2026-04-20; author `heziiop`.
- Diff scope read: `1` files, `+66/-10`; areas: model wrapper; keywords: attention, cache, config, cuda, expert, kv, triton.
- Code diff details:
  - `python/sglang/srt/models/minimax_m2.py` modified +66/-10 (76 lines); hunks: import logging; get_compiler_backend,; symbols: forward_prepare, forward_prepare_npu, forward_core, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/minimax_m2.py`; keywords observed in patches: attention, cache, config, cuda, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23301 - [sgl] Stream MiniMax M2 string parameters token-by-token

- Link: https://github.com/sgl-project/sglang/pull/23301
- Status/date: `open`, created 2026-04-21; author `lujiajing1126`.
- Diff scope read: `1` files, `+332/-280`; areas: misc; keywords: config, spec.
- Code diff details:
  - `python/sglang/srt/function_call/minimax_m2.py` modified +332/-280 (612 lines); hunks: logger = logging.getLogger(__name__); class MinimaxM2Detector(BaseFormatDetector):; symbols: MinimaxM2Detector, MinimaxM2Detector, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/function_call/minimax_m2.py`; keywords observed in patches: config, spec. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/function_call/minimax_m2.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
