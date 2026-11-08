# DeepSeek V4.1 kernel PR review — 2026-11-08

These six SGLang PRs were open and unmerged when inspected. Full diffs, PR
bodies, issue comments, reviews and inline discussions were read at the heads
below; paginated changed-file counts agree with the diff headers. No GPU
experiment was run for this skill refresh. Author measurements are identified
separately from the tests present in each proposed patch.

Mainline comparisons use `c24285bde97c3e6424cd5d8cb35788889034d8ea`.
The current `v41_indexer/{__init__,types,scoring,full_topk,dense_blocks}.py`,
`mqa_logits_utils.py`, `fp4_indexer.py`, `candidate_blocks.py`, Python/C++ KV
layout definitions, KV quantization reference and KV store tests were read in
full. The mHC dispatch/projection functions and KV pool layout selectors were
also inspected. This is a bounded source review, not a review of every model
module or every open PR.

Current source entrypoints:
[indexer dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/__init__.py),
[score producers and selection writes](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/scoring.py),
[paged FP4 indexer](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/ops/attention/dsv4/fp4_indexer.py),
[mHC dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/models/deepseek_v4_mhc.py),
[KV row encoding](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh).

## PR #48456 — BF16x3 mHC projection for Blackwell decode rows

- Link: https://github.com/sgl-project/sglang/pull/48456
- State: open, unmerged; head `43667bbfacb936149431ab00475d0856a951f7fd`,
  base `c67efab427ac9d7965123a3cea50bf1d3d899563`.
- Diff coverage: full diff read, 53 lines, 2 files, +14/-4.
- Motivation: the author reports a GSM8K regression when the FFN mHC statistics
  projection overlaps MegaGate/MegaMoE after fused decode post. Isolated TF32
  projection agrees closely with FP64; serialization or reserving SMs removes
  the observed model regression. This supports investigating concurrent
  execution, but does not establish that occupancy itself changes values.
- Implementation: `_mix_stats_impl` routes eligible small Blackwell batches to
  compensated BF16x3 instead of TF32 DeepGEMM. The wrapper's `decode_grid`
  chooses 64 K slices and 64-row blocks, replacing the ordinary 16-slice grid.
  The outer dispatch still requires CUDA BF16 contiguous rows, M>=32,
  SM100-family/SM90 eligibility, enabled DeepGEMM prenorm and no batch-invariant
  mode. The new effective Blackwell range is 32–4095 rows, including verification
  rows; `is_blackwell` in the inner predicate does not bypass those outer gates.
- Key excerpt:

```diff
+    if decode_grid:
+        slices, block_m = 64, 64
+    else:
+        slices = _HC_MIX_COMPENSATED_SLICES
+        block_m = 64 if hopper_medium else _HC_MIX_BF16X3_BLOCK_M
```

- Reviewed files: `python/sglang/kernels/ops/layernorm/mhc.py` and
  `python/sglang/srt/models/deepseek_v4_mhc.py`. The shared projection keeps
  three BF16 weight parts, FP32 partial accumulation and the final Sinkhorn
  reduction; K must remain divisible by `slices * _HC_MIX_BLOCK_K`.
- Validation: no test is added. Inline review requests SM100 dispatch and FP64
  checks at M=32,33,257,4095,4096; the existing compensated test is Hopper-only.
  The author reports GB300 NVFP4 PD/DP-attention GSM8K 96.89–97.19 versus
  95.15–95.53 on the TF32 path, and 13.1–13.4 us versus 17.4–18.5 us for the mix
  at 84–128 rows. End-to-end throughput is reported unchanged. Verify stream
  dependencies, scratch lifetime and actual model accuracy under overlap in
  addition to isolated numerical checks; profiler serialization can hide the
  condition being investigated.

## PR #48500 — Floor amax before dividing the FP8 KV scale

- Link: https://github.com/sgl-project/sglang/pull/48500
- State: open, unmerged; head `464dd8f8751c2910fc515d8f1728822b9e943e7b`,
  base `bf40ca98f4287347ec8d19662aa390eacc253a1e`.
- Diff coverage: full diff read, 101 lines, 3 files, +30/-11.
- Motivation: the current KV writer and its Torch reference both floor
  `amax / 448` at `1e-4`. Their byte-exact agreement therefore cannot detect
  disagreement with the model's intended `max(amax, 1e-4) / 448` formula.
  The minimum power-of-two scale changes from 2^-13 to 2^-22; small tiles with
  `amax <= 448 * 2^-14` are affected.
- Implementation: change the UE8M0 exponent floor from 114 to 105 in the shared
  CUDA/HIP `store_row_fp8`; change `ceil_pow2_scale` to accept amax, move the
  division inside it, and update its caller. V41 and V41_NEW share this FP8
  encoding. The FP4 writer and V4 encoding are separate contracts.
- Key excerpt:

```diff
-    scale = ceil_pow2(torch.clamp_min(x, 1e-4))
+    scale = ceil_pow2(torch.clamp_min(x, 1e-4) / E4M3_MAX)
```

- Reviewed files: `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh`,
  `python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py` and
  `test/registered/kernels/ops/attention/dsv4/test_v41_kv_store.py`.
- Validation: a new CPU test uses a direct mathematical formula, including
  zero and tiny maxima; GPU store boundary rows add 2^-20 and 0.02. Existing
  tests cover payload/scale bytes, untouched slots, norm/RoPE/compression writers
  and interleaved layouts. They explicitly omit NaN/inf writer parity. The
  author reports SM89/SM120 comparisons against `act_quant`, but no B200 or
  MI355 store run; the captured discussion reports a failed lint check. Inspect
  the selected pool layout before attributing a trace or accuracy change to
  this proposal. Keep an independent formula/reference in quantization checks,
  rather than copying the optimized implementation's arithmetic into its oracle.

## Shared provenance of the Hopper indexer proposals

The four PRs below port experiments from
`91afc4f363cc60ca0bf98d38e87cc6cfa32062fe`. Their reported A/B runs used eight
H20-3e GPUs split four prefill/four decode, DSpark block size 5, 8192-token
prefill chunks, concurrency 24, and 256 requests with 61440 shared-prefix,
4096 request-specific and 500 output tokens. The commands also pin CP/DP,
MoE, HiCache, router and graph choices; #49066 uses different EP/prefill-graph
settings from the later experiments. Do not add their percentages or treat
the tables as a controlled comparison across all four changes.

Each body says GPU validation of the upstream port is pending. Each has only
an automated review-disabled notice, no substantive approval in the retrieved
discussion. Experimental flags in the reproduction commands differ from the
new upstream flags. The following table refers to the reviewed PR code,
not frozen main or an installed wheel:

| Proposal | Reviewed PR switch | Scope |
|---|---|---|
| #49066 | `SGLANG_OPT_DSV41_SM90_DEEPGEMM_PREFILL=1` | Hopper eager/rank-local prefill scoring; default off. |
| #49067 | `SGLANG_DSV41_PREFILL_SCORE_BUDGET_MIB`, `SGLANG_DSV41_DECODE_SCORE_BUDGET_MIB` | Positive MiB overrides; unset preserves existing behavior. Decode slicing is Hopper-only. |
| #49068 | `SGLANG_OPT_DSV41_VERIFY_INDEXER_K_REUSE=1` | Fixed-width Hopper verification groups; default off. The experimental second `BF16_K_REUSE` flag is absent from this port. |
| #49069 | `SGLANG_OPT_DSV41_TRITON_PREFILL_SELECTION=1` | Eligible Hopper prefill selection/writeback; default off. |

## PR #49066 — Gather FP4 keys once, score prefill with FP8 DeepGEMM

- Link: https://github.com/sgl-project/sglang/pull/49066
- State: open, unmerged; head `cc7f11419f59df716a6911a3b6c938f8c4418dde`,
  base `043cfc4d3a2cc2a931d15862f08c4b60b692c957`.
- Diff coverage: full diff read, 637 lines, 8 files, +529/-3.
- Motivation: the current non-SM100 prefill path materializes BF16 per-head
  Torch scores. Gather each request's packed index-K once and use Hopper FP8
  tensor cores without carrying the much larger per-head score intermediate.
- Implementation: decode E2M1/UE8M0 K to finite E4M3, clamp BF16 Q to E4M3,
  and call `deep_gemm.fp8_mqa_logits` with unit FP32 K scales because the
  unpacker already applied block scales. Head weights are rounded through BF16
  then stored FP32 for the API. Per-request starts are zero; full-width output
  with `clean_logits=True` initializes causal tails to -inf for candidate
  reduction. The code requires CUDA SM90, BF16 input, head dimension 128,
  32/64 local heads and CPU request/length vectors. Full top-k, candidate
  publication and consumption share the new scorer.
- Key excerpt:

```diff
+    return deep_gemm.fp8_mqa_logits(
+        q.contiguous(),
+        (keys.contiguous(), key_scales.contiguous()),
+        # Match the head-weight input of the BF16-rounding scorer, but DG's
+        # SM90 kernel requires FP32 storage for those values.
+        weights.to(torch.bfloat16).float().contiguous(),
+        torch.zeros_like(ends),
+        ends,
+        True,
+        0,
+    )
```

- Reviewed files: kernel registration in `kernels/ops/attention/__init__.py`,
  new `kernels/ops/attention/dsv4/sm90_deepgemm_prefill.py`, `srt/environ.py`,
  `v41_indexer/{scoring,sm90_prefill,full_topk,dense_blocks}.py` under
  `srt/layers/attention/dsv4`, and
  `test/registered/kernels/ops/attention/test_dsv41_sm90_deepgemm_prefill.py`.
  Runtime paths in this and subsequent cards are below `python/sglang/`.
- Validation: tests cover FP64 arithmetic on already-FP8 operands, signed
  weights, non-unit key scales, odd row counts, causal cleaning, top-k and
  shrinking-length graph replay. A runtime test covers empty CP shards,
  padded rows, source-tail and masked-consumer handling; another tests finite
  saturation. These do not establish equivalence to the original BF16 score
  rounding chain or full-model accuracy. The reported experiment improves
  output throughput 43.11% and TTFT 48.38%, while TPOT worsens 10.13%; it is
  not a measurement of this unvalidated port. Inspect Q/K quantization error,
  selected sets and acceptance/accuracy, then measure without profiling.

## PR #49067 — Separate prefill/decode score-tile budgets

- Link: https://github.com/sgl-project/sglang/pull/49067
- State: open, unmerged; head `907854304a3f63bd42756df5801d4181f6659863`,
  base `043cfc4d3a2cc2a931d15862f08c4b60b692c957`.
- Diff coverage: full diff read, 349 lines, 5 files, +154/-77.
- Motivation: small tiles repeat scoring and selection launches; large tiles
  consume scratch outside the KV cache budget. Expose phase-specific choices
  while retaining graph-stable decode shapes.
- Implementation: prefill overrides both the flattened DeepGEMM tile budget
  and the Torch per-head score budget. Hopper decode computes a row step from
  static maximum context, rounds it down to whole request groups and slices
  inputs, output views, page tables and candidate masks together. Full top-k,
  candidate publishers and consumers all iterate chunks; publishers concatenate
  block IDs, then build their shared mask. `del d` releases decode score
  references before the next iteration.
- Key excerpt:

```diff
+    group = metadata.rows_per_request
+    step = max(group, step // group * group)
+    inputs.reset_outputs()
+    for start in range(0, num_rows, step):
+        rows = slice(start, min(start + step, num_rows))
```

- Reviewed files: `srt/environ.py`,
  `srt/layers/attention/dsv4/v41_indexer/{scoring,full_topk,dense_blocks}.py`,
  and `test/registered/kernels/ops/attention/test_fp4_indexer.py`.
- Validation: the extended replay test compares unset and 1 MiB decode budgets
  for ratios 1/2, all three runtime entrypoints, candidate masks and changing
  live lengths. The author A/B changes **prefill only**, from 64 MiB to 2 GiB,
  reporting +1.92% throughput and +0.87% TTFT; it says nothing about the speed
  of the new decode override. A tile budget is not a total-memory guarantee:
  one request group can exceed it, Torch transients exceed its base score
  tensor, and Q/K, selections, masks and allocator lifetimes also consume memory.
  Changing row groups can change GEMM dispatch/rounding. Check peak allocation,
  selection/accuracy and capture/replay separately from launch-count reduction.

## PR #49068 — Reuse a decoded K tile across verification rows

- Link: https://github.com/sgl-project/sglang/pull/49068
- State: open, unmerged; head `f9b10b6783ac782f886eeb83e20e6dafbf98b9ea`,
  base `043cfc4d3a2cc2a931d15862f08c4b60b692c957`.
- Diff coverage: full diff read, 195 lines, 4 files, +124/-0.
- Motivation: consecutive target-verification queries of one request read
  the same packed K tile. Reusing its BF16 even/odd halves removes repeated
  gathers and dequantization while keeping each query's causal limit.
- Implementation: `decode_scores` passes `metadata.rows_per_request` only for
  enabled verify. The grouped kernel requires Hopper, 32 heads, groups 2–16,
  a batch divisible by group width and no candidate mask. It loads the request
  ID from the first row, gathers/dequantizes K once up to the group's maximum
  live length, then loops over query rows using each row's own visibility.
  It reuses the existing dot/ReLU/weight/sum BF16 rounding helpers.
- Key excerpt:

```diff
+        k_low, k_high = _fp4_index_k_tile(
+            table, slots, valid, page_size, cache_stride, BLOCK_L, HALF_D
+        )
+        for offset in tl.range(GROUP, loop_unroll_factor=1):
+            row = first + offset
```

- Reviewed files: `kernels/ops/attention/dsv4/fp4_indexer.py`, `srt/environ.py`,
  `srt/layers/attention/dsv4/v41_indexer/scoring.py` and
  `test/registered/kernels/ops/attention/test_fp4_indexer.py`.
- Validation: new tests compare grouped and existing paths bitwise on visible
  logits under graph replay for ratios 1/2, group width 6, width 261 and live
  lengths including zero. Other group widths, 64 heads and candidate-mask
  fallback need their own coverage. The kernel trusts same-request grouping;
  batch divisibility alone cannot prove it. Never consume uninitialized tails
  beyond device-side lengths. Reported experiment: TPOT -8.54%, throughput
  +0.63%, TTFT +10.85%; the upstream GPU port remains unvalidated.

## PR #49069 — Fuse candidate reduction and selection writeback

- Link: https://github.com/sgl-project/sglang/pull/49069
- State: open, unmerged; head `328107fe3b8ad89a4597001cdde6816dc8c6ce99`,
  base `043cfc4d3a2cc2a931d15862f08c4b60b692c957`.
- Diff coverage: full diff read, 484 lines, 7 files, +369/-1.
- Motivation: reduce intermediate score traffic and separate launches for
  block maxima, newest-block retention, top-k, sorting, masking and slot writes.
- Implementation: a Triton selector encodes ordered block maxima and performs
  radix threshold selection, keeping the newest visible block with +inf.
  BF16 keys use 16 bits, FP32 keys 32; large 2048-block selections use a separate
  reduction into one key per block before selection. The fast path requires
  CUDA, contiguous columns, FP32/BF16 logits, block size 8, width<=131072 and
  either width<=32768 or topk_blocks=2048. Unsupported shapes fall back.
  A second kernel sorts selected positions, applies lengths and writes raw
  indices/physical slots; its runtime gate is Hopper with k<=4096.
- Key excerpt:

```diff
+    idx = tl.sort(tl.where((idx >= 0) & (idx < length), idx, length), descending=False)
+    token = tl.load(tokens + row * T).to(tl.int64)
+    visible = tl.minimum(tl.load(lengths + row * L), length)
+    valid = (cols < K) & (idx < visible)
```

- Reviewed files: `kernels/ops/attention/__init__.py`,
  `kernels/ops/attention/dsv4/{candidate_blocks,fp4_indexer}.py`, `srt/environ.py`,
  `srt/layers/attention/dsv4/v41_indexer/{dense_blocks,scoring}.py` and
  `test/registered/kernels/ops/attention/test_fp4_indexer.py`.
- Validation: tests check candidate sets with distinct FP32 maxima at widths
  21,1021,32781 and sorted writeback with invalid picks/nonconsecutive token
  rows. They do not cover BF16 ties, NaN/inf, signed zero, all fallbacks or
  graph replay. The producer must already put -inf past causal lengths; the
  selector does not reconstruct that mask. Writeback relies on the caller's
  output reset for untouched rows/columns. Reported throughput improves 1.59%
  and TTFT 4.25%, with TPOT +0.47%; no GPU result is supplied for the port.

## Applying these findings to a trace

Record the exact branch/flags, GPU family, rank-local row count, request-group
width, context capacity and live lengths. Distinguish attention KV payloads
from the separate 68-byte index-K slots. For eager and captured prefill, first
trace the caller: a proposal in `prefill_requests` does not automatically alter
`CapturedPrefillInputs` execution. Attribute the full gather/quantize/score/
select/writeback chain, measure scratch peaks and exposed stream tails, and
check selected tokens and model accuracy before carrying a kernel result into
a serving recommendation.
