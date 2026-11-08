# TokenSpeed index selection and distributed scoring

Inspected on 2026-11-08. The full diffs, bodies, comments and reviews for the
three PRs below were read. Current scoring, selection,
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
  Both issue comments were read; there are no reviews or inline comments.
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

## PR #1877 — Skip empty tiles in the portable DSA scorer

- Link: https://github.com/lightseekorg/tokenspeed/pull/1877
- State: merged 2026-10-11; head `87cbf2b68254a5948722fda9241d631a9a548152`,
  base `e1207a426191324733eab53a815a00ffafd3847b`,
  merge `b357a4ec88eef4560f82d86e87e6386bfd360525`. Fresh head/base check matched.
- Coverage: all 251 diff lines, three files, +177/-41; eight issue comments,
  three reviews and no inline comments. The final head has an approval; the
  earlier change request concerned DCO sign-off.
- Reviewed files:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/_triton/topk.py`,
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/README.md`,
  `tokenspeed-kernel/test/ops/test_attention_dsa.py`.
- Motivation: DCP page tables retain global column positions and mark foreign
  pages `-1`. The portable FP8 scorer still performed head/channel arithmetic
  for tiles containing no owned, causally visible keys.

When `page_size % BLOCK_N == 0`, one tile lies inside one page. Loading its
first page ID under causal/capacity bounds proves whether it contains any
valid entry. Other page geometries reduce the per-entry validity mask. Empty
tiles skip Q/K loads and scoring, but still overwrite logits with `-inf`:

```diff
+    # Foreign pages and capacity beyond the causal length can leave a whole
+    # tile empty. Skip Q/K scoring, but still overwrite its logits for Top-K.
+    scores = tl.full((BLOCK_N,), -float("inf"), tl.float32)
+    if has_valid:
```

Partially valid tiles retain their original head reduction, scales, ReLU and
per-entry mask. Forced initial/tail windows are set to `+inf` before invalid
and NaN scores are masked back to `-inf`; a forced position on a foreign page
must not become a valid local candidate. The kernel signature and compile-time
axes are unchanged. This reduces scoring arithmetic, not grid size, allocated
logits or the following Top-K scan. It differs from #1857's bounded cleaner.

Current main retains this implementation. Both its ordinary FP8 decode top-k
caller and its explicit-row DCP candidate caller use 64-key tiles. The latter
allocates FP32 logits over the full local-table width, then selects logical
offsets/scores and rejects invalid candidates. Its registered portable
NVIDIA/AMD provider does not advertise planned compact pages; on NVIDIA it
first quantizes queries and folds scales into weights to match the native
indexer convention. Runtime DCP selection resolves ownership, tiles rows and
exchanges candidates for global selection. Other selected providers can accept
compact page plans, so this portable kernel's dense-grid behavior must not be
generalized to every DCP backend. The runtime logs the selected sparse leaf;
use that or kernel scopes to establish which path a trace actually contains.

The new test covers page sizes 32 and 128, implicit and explicit request rows,
foreign pages, partial tails, empty rows and NaN queries. Graph replay changes
the table from empty to refilled, changes lengths/request mapping and poisons
all logits with `123.0` before each replay. The reference checks use
`rtol=atol=2e-4`, not a measured bitwise cross-provider guarantee.

The body reports roughly 8× less Q/K scoring time for one request at 256K
history with DCP=8 on Hopper, plus single-GPU simulated-shard correctness.
It supplies no raw timing table, measured revision or E2E result. The final
head's AMD benchmark comment reports one regression, one improvement, 475
within budget and one inconclusive case. A later author comment describes
benchmarks as passing and asks to waive two model-score failures; another
calls the run flaky. Those statements do not resolve the recorded results
without CI logs. Automated reviewers explicitly lacked GPUs. This review
does not certify those runs or convert source equivalence into fresh accuracy
validation.

Current source:
[scoring kernel and ordinary decode caller](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/_triton/topk.py),
[portable DCP caller/registration](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/triton.py),
[candidate dispatch and page plans](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/__init__.py),
[runtime ownership/selection/exchange](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/attention/dcp/indexer.py).

## PR #1781 — Closed KPool prefill row partitioning proposal

- Link: https://github.com/lightseekorg/tokenspeed/pull/1781
- State: closed **without merge** on 2026-10-16; head
  `7eee802c243a26894a5f82bc30001b4fee483cac`, base
  `c4fec5375cb64db13382c46a75499db2cc168b31`. Fresh head/base check matched.
  The closing bot cites inactivity; it is not a maintainer rejection of the
  algorithm or evidence that it shipped through this PR.
- Coverage: all 287 diff lines, five files, +185/-9; all three issue comments,
  no reviews or inline comments.
- Reviewed runtime: `python/tokenspeed/runtime/layers/attention/kpool.py`,
  `python/tokenspeed/runtime/models/glm53_flash.py`; documentation:
  `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/kpool/README.md`;
  tests: `test/runtime/distributed/test_kpool_prefill_tp_nccl.py`,
  `tokenspeed-kernel/test/ops/test_kpool_select.py`.
- Motivation: GLM-5.3-Flash attention TP ranks replicated the same long-prefill
  KPool query scoring. Assigning distinct query rows could save that duplicated
  work, at the cost of gathering selected slots and lengths.

The proposed split requires multiple TP ranks, at least one row per rank and
history exceeding twice the selected pool budget. The exact gate is:

```diff
+        if (
+            tp_size > 1
+            and num_rows >= tp_size
+            and shared_plan.max_num_pools > 2 * topk_pools
+        ):
```

It partitions contiguous rows nearly evenly, slicing query, weights, positions,
request IDs, causal lengths and per-row workspace bounds together. Shared
request boundaries and cache/workspace tables remain global. `_gather_rows`
pads each rank to the largest slice for `all_gather_single`, then trims in
rank order; it gathers selected indices and lengths separately. Padding is
uninitialized but excluded from the reconstructed result. All ranks must use
the same row counts and group. This is a query-row partition, not head
sharding or a reduction over partial scores.

Current `KPoolRuntime.select_prefill` still passes the full query and row
metadata to `kpool_prefill_topk`, with no TP group/rank arguments or result
gather. Its GLM-5.3-Flash caller likewise supplies no TP mapping. The portable
kernel accepts explicit per-row request IDs, which permit a slice crossing
request boundaries, but that API support alone does not activate partitioning.
Do not describe this closed proposal as an available mainline switch.

The added CUDA test compares split/concatenated selections against the full
batch across a request boundary; a two-GPU NCCL test checks equal/uneven
gathered rows and vector/matrix outputs. The body does not report pass counts
for those tests. It reports TP8 H20 microbenchmarks: at 32K prefix, 128 rows
improve 1.905→1.051 ms and 257 rows 3.401→1.243 ms. At 8K prefix, eight rows
regress 0.683→0.827 ms and 17 rows 0.685→0.950 ms; the gate does not establish
a universal crossover. Three of the four small cases regress; the 2K/17-row
case improves 1.4%, despite the body's blanket statement about four regressions.

The author also reports two 8×H20 TP8/EP8/MTP3 32K-input/64-output C1 pairs
with E2E p50 reductions of 7.58% and 7.34%, plus a 60K case whose TTFT improves
while TPOT worsens 3.8%. Warmups were excluded. These are historical branch
reports with no measured SHA or answer-accuracy result, not validation of a
current implementation. The author's gate-tuning question remained open in
the discussion before automatic closure.

Current source:
[KPool runtime](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/layers/attention/kpool.py),
[GLM-5.3-Flash caller](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/models/glm53_flash.py),
[portable row metadata/selection](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/kpool/triton.py).
