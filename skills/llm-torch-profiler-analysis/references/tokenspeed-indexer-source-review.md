# TokenSpeed V4.1 Hopper index selection

Inspected on 2026-11-08. The full PR diff, body and both issue comments were
read; there are no reviews or inline comments. Current scoring, selection,
kernel registration and runtime callers were checked at TokenSpeed
`7b21034e72bb45d1edef3d3b5483b50b58c4c492`. No GPU experiment was rerun locally.

## PR #1857 — Fuse block maxima and bound candidate selection

- Link: https://github.com/lightseekorg/tokenspeed/pull/1857
- State: merged 2026-10-12; head `a78cb7e53d659cd74e7e81bb3d08c27f207d3992`,
  base `e7bfdb8ff6e820659f92e2efbaf7d3456828a8f4`,
  merge `339edcd258655b236c8dd220c0ff6355fe75122b`.
  A fresh API check matched both reviewed revisions.
- Coverage: all 656 diff lines, three files, +528/-20:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py`,
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py`,
  `tokenspeed-kernel/test/nvidia/ops/test_dsv41_block_maxima.py`.
- Motivation: the allocated logits capacity can greatly exceed visible history.
  Full-capacity cleaning/reduction and selection over padded block scores then
  cost much more than the useful work. The previous block-max path also launched
  separate PyTorch reductions, indexing and updates to keep the newest block.

### Two reductions with different read contracts

`block_maxima_with_lengths` replaces the PyTorch block reduction and newest-block
update with one Triton kernel. It returns maxima and GPU-resident
`ceil(clamp(visible, 0, width) / 8)` counts. This standalone helper still reads
the full score width; passing valid counts to the following selector does not
make that reduction itself proportional to the visible prefix. Scores outside
the prefix must already be masked before selecting by those counts.

`clean_block_maxima` is the stronger fusion for candidate-source passes. It
combines 64-token page validity, score cleaning, eight-token maxima, valid block
counts and newest-block pinning. A fixed launch uses up to 256 workers per row,
each looping only over live 1024-score tiles. Invalid/out-of-range pages within
the visible prefix become `-inf`; the last block becomes `+inf` only if its
maximum is greater than `-inf`. NaNs propagate through the maximum and do not
become an apparently valid pinned block.

The original FP32 fused cleaner allocates capacity-shaped cleaned logits and
block maxima, but writes only the live score/block prefixes. Both token and
block selectors must obey their device lengths. Scratch remains proportional
to capacity, and native logits plus the cleaned copy can coexist; the change
reduces launches and memory traffic without proving a whole-model memory saving.
No CPU length read or data-dependent output shape is required.

The caller guard is essential:

```diff
+        # Candidate gathering can read past visible into a partial block. Keep
+        # its fully initialized scores; only bounded selectors consume the
+        # partially initialized output of the fused cleaner.
+        bounded = bool(candidate_topk) and candidate_blocks is None
```

Only candidate-producing Full passes with no input candidate pool take that
fused-cleaning path. Candidate gathering checks allocated width, not the causal
length, and may load the unused positions of a partial eight-token block.
Those consumers therefore retain fully cleaned scores. A sparse Reindex path
can instead score its candidate pool directly; it does not need source block
maxima at all.

`_native_select` sends visible row lengths to token selection and the newly
produced block counts to candidate selection. DeepSelect survivor capacities
remain 512 rows and 2048 blocks. `_finish_topk` filters invalid scores, resolves
compact columns through candidate block IDs, sorts logical IDs and pads with
`-1`. Candidate-source selection is derived from every visible block maximum,
not from the already selected Top512 rows. Equal-score boundary ties do not
have a cross-provider ordering guarantee.

### Current dispatch and later changes

The current Hopper native provider requires replicated 32 query heads,
128-wide query channels, 64-row FP8 `index_v4` pages (132 bytes per row),
compatible page strides/alignment, no head process group and available
DeepGEMM. It scores FP8 queries/keys into FP32 logits and uses the in-tree
CuTe DSL DeepSelect implementation. This is separate from Blackwell's
FP4-cache/BF16-logit path and portable score accumulation.

Hopper broadcast tables with more than one query use dense scoring; paged
scoring handles other layouts. The source still chooses query tiles using
128 MiB for broadcast dense logits and 32 MiB for paged logits, in addition
to the caller's query bound. These are logits budgets, not bounds on all live
temporary storage. Runtime callers further trim tables to available residency,
construct phase/window-specific geometry and can bypass scoring entirely when
all visible rows fit the selection capacity.

The current fused cleaner additionally accepts BF16 logits and always emits
FP32 block maxima; the original PR accepted FP32 only. Row pointer arithmetic
now casts program IDs to int64. The standalone maximum helper and bounded
cleaner still have the distinct tail contracts above. Current native Hopper
dispatch retains `candidate_topk > 0 && candidate_blocks is None` for the
fused cleaner; existence of the helper does not imply every Reindex or Full
pass uses it. Blackwell has separate cleaner/selector implementations and
should not be attributed to this Hopper PR merely from similar trace names.

### Tests and measured scope

The added test file checks exact block-max agreement, including NaNs, infinities,
strided scores, partial blocks and empty rows. It compares selected candidate
sets and downstream token sets, exercises invalid page holes and
contiguous/strided/broadcast tables, and verifies compilation reuse across
capacity changes. Graph refresh changes long/empty/short visible lengths while
poisoning unwritten output tails with `+inf`; the selections must still match
the fully cleaned reference. Those tail and set checks protect real behavior.

The body reports 31 regression tests passing after rebase. Its H20-3e
microbenchmark covers cleaning + block maxima + token/block Top-K, **excluding
DeepGEMM scoring**, with medians of five alternating trials of 200 graph replays.
At capacity 1,048,640 and 1,330 visible tokens, six rows improve from
139.61 to 27.95 microseconds (4.99×), and 32 rows from 497.12 to 30.78
(16.15×). Full visible capacity still reports 1.75×; these are measured
postprocessing shapes, not general model speedups.

The separate eight-H20 request benchmark uses GSM8K prompts, concurrency 32,
greedy decoding, fixed 128 output tokens, excluded warmup and flushed prefix
cache. It reports TPOT 30.14→29.25 ms (−2.95%), output throughput
983.08→1011.86 token/s (+2.93%) and 1319/1319 successful requests.
No answer-accuracy score or measured candidate SHA is given in that body.
Successful requests do not prove model accuracy; the microbenchmark and
service benchmark are distinct evidence. The AMD bot reports 514 cases within
budget for the final head/base, but that is not execution proof for the
NVIDIA-only Hopper selection tests. This refresh did not inspect those CI
checkout logs or repeat the measurements.

Current source:
[runtime selection windows/callers](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/attention/backends/specific/deepseek_v41.py),
[native provider admission](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/__init__.py),
[Hopper scoring/selection](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/deep_gemm.py),
[cleaner, maxima and finalizer](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsv41/triton.py).
