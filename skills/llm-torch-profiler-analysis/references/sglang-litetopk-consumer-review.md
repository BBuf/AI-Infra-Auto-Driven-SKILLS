# SGLang LiteTopK decode: selection, storage and current callers

Reviewed on 2026-11-09 (Asia/Shanghai). No GPU execution, model evaluation,
sanitizer, SASS inspection or performance rerun was performed here.

## PR #40764 — opt-in histogram-assisted DSA and V4.1 selection

- Link/title: [[Perf] [NVIDIA] attention: Add opt-in LiteTopK decode top-k for DSA and DeepSeek-V4.1 on SM100](https://github.com/sgl-project/sglang/pull/40764).
- State: open, not merged, with a merge conflict at this check. Head
  `01e293958e709be59b934e0561f4122b92d4aa77`; API base
  `00f6cdcd1b6fdd215c623a7f6e167ee197d2b138`; actual merge base
  `3626978a13694eb4c6e3a6080e4f1c4d3e5f4d52`.
- Coverage: all 5,538 lines of the complete 41-file diff read, +3,837/-224,
  including the 1,286-line selector. Full body and all three commit messages
  read; paginated API and connector queries returned no issue comments,
  formal reviews or inline comments. The last-commit-only diff is 3,231
  lines across 15 files, +2,864/-18. The other two commits carry an older
  version of the [DeepGEMM migration](sglang-deepgemm-migration-review.md).
- Motivation: that dependency's BF16 producer widened the entire padded
  score matrix for FP32 top-k. The [producer proposal](deepgemm-litetopk-producer-review.md)
  counts a coarse histogram while writing BF16 scores; this consumer avoids
  the conversion and separate histogram-building pass. **Scores remain
  materialized in HBM.** This is not the older score-free indexer proposal.

`SGLANG_OPT_LITETOPK_DECODE=1` opts in; it defaults off. Neither this patch
nor its histogram dependency is present at the frozen current source pins.
The historical gains below are not measurements of current SGLang main.

## Selection and numerical contract

The [selector](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/kernels/jit/csrc/litetopk_decode/select.cuh)
has two configurations: BF16 top-2048 with 64-slot pages for DSA, and BF16
top-512 with 128-slot pages for V4.1. It emits physical KV slots, unordered,
with `-1` padding. A page-table row may serve multiple consecutive query
rows. Live lengths are clamped to the score width; when length is at most K,
every live slot is selected.

The ordering key combines the ordered BF16 bits with the complemented
physical slot. Higher scores win, then lower slots for equal keys. This
ordering distinguishes **negative zero below positive zero**, even though
the histogram normalizes zero. “Ties choose the lower slot” therefore needs
that signed-zero qualification. Histograms exclude NaNs; the tests do not
define a NaN selection contract. The exact-ranking code assumes distinct
composite keys, so valid page-table coverage and nonduplicated live physical
slots remain caller requirements, not host-validated guarantees.

The producer's 1,024-bin histogram locates the crossing bin. The consumer
certifies its total and the observed counts strictly above and inside that
bin. Candidate overflow or inconsistent counts trigger a whole-row exact
radix selection. The decisive code is:

```diff
+    bool ok =
+        s.certified and strict_total == s.strict and inside_total == s.count and (s.whole or s.count <= p.capacity);
+    const bool drained = ok and not s.whole;  // resolve_bin reads and zeroes every candidate
+    if (drained) ok = resolve_bin(p, s, row, out_row);
```

Rows up to 16,384 scores use one 512-thread CTA and register-held scores.
Long rows split across CTAs, scan 16,384-score tiles with 512 threads (twice
that with 1,024 threads), and use finer bins plus
exact ranking/radix selection for the crossing candidates. Small page-table
slices are staged with `cp.async`; larger slices use global reads. Grid and
512/1,024-thread choices depend on row count and SM count. The thresholds
were measured on a 148-SM B200; changing the supplied SM count tests launch
logic, not performance on four different GPUs.

The host checks BF16 contiguous score columns, row stride divisible by eight,
16-byte score alignment, int32 lengths/table/output, the histogram/output
shapes, exact workspace size/alignment, and width below `2^23`. Candidate
capacity is in `[1, 2^24]`; SM count is in `[1, 2^16]`.
These checks do not establish sufficient page-table columns or
unique physical mappings. PDL waits before reading input/state and launches
dependents; it does not serialize independent streams sharing the buffers.

## Completion protocol and graph-owned memory

Each row has a 16-byte state: a packed arrival/strict/inside-count word,
`done`, and `done_base`. Candidate slots are 8-byte composites. At the
default capacity of 8,192, histogram plus workspace consumes
`4096 + 16 + 8192 * 8 = 69,648` bytes per maximum row, excluding logits and
outputs. The description “68 KiB per row” omits the additional 16 bytes.

Reservations precede candidate writes. Nonfinalizing parts publish release
completion signals per warp; the finalizer waits for nonzero candidate
payloads before ranking. Before fallback clears candidates or rewrites
outputs, `await_parts` acquires completion until
`done - done_base == (parts - 1) * 32`. At the end:

```diff
+    clear_histogram(p, row);
+    if (threadIdx.x == 0) {
+      // Every part has arrived, so the word can be reset; their `done` signals may still be in flight
+      p.state[row].word = 0;
+      atomicAdd(&p.state[row].done_base, (parts - 1) * kPartSignal);  // no return value: a fire-and-forget reduction
+    }
```

At rest the histogram, candidate entries and packed word are zero, while
`done == done_base` can be nonzero. The base may advance before late signals
arrive. Do not describe the entire workspace as zero after every call, or
remove the completion protocol because the candidate values were consumed.

The [storage/plan wrapper](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/kernels/ops/attention/litetopk_decode.py)
places smaller plans' row states immediately before the same candidate
region. Plans are overlapping views and require **serial use**. The
[runtime owner](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/srt/layers/attention/litetopk_decode.py)
caches one instance per configuration, not per CUDA device. Its intended
one-GPU-per-process, sequential-layer ownership must be preserved.

Growth outside capture allocates at least twice the previous row capacity.
Plans used during graph preparation/capture retain their old storage; eager
retired storage is protected with `record_stream`. A new plan can view
existing storage during capture, but capture needing new allocation warns
and falls back. Retained graph storage adds to the current allocation and
is outside `mem_fraction_static`. `record_stream` protects deallocation; it
does not make simultaneous calls safe. No retirement policy here releases
the retained graph plans while the owner remains alive.

Capability detection requires CUDA major 10 and explicit `indices` in the
DeepGEMM metadata signature, plus explicit `indices` and `histogram` but no
`logits_dtype` in the producer signature. `**kwargs` does not count. Current
DeepGEMM `dev` has a compatibility wrapper and no histogram implementation;
adding a private kernel argument alone would still fail this admission
check. Reconcile both wrapper and kernel when porting.

## DSA and V4.1 callers at the reviewed head

The [DSA indexer](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/srt/layers/attention/dsa/dsa_indexer.py)
requires 32 index heads, dimension 128, top-2048, and the dependency's UE8M0
index-K cache. Forced initial/local tokens are excluded because they edit
scores after histogram production. Runtime admission further requires decode
or target verify, `DSAIndexerMetadata`, the SGL_KERNEL backend, PAGED
transform, `SGLANG_DSA_FUSE_TOPK`, no forced unfused path, and matching length
and table rows. Prefill, ragged and draft-extend paths keep their defaults.

The migration adapter folds Q scaling into BF16 weights, supplies packed
unit Q scales where needed, and returns BF16 directly when given a histogram.
Chunked production slices histogram rows by `start * next_n:end * next_n`,
concatenates logits and then selects. This old stack disables native SM100
verify grouping and uses independent query rows. The causal-length fix in
[DSA metadata](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/srt/layers/attention/dsa_backend.py)
clones expanded lengths so graph refreshes cannot overwrite them through an
alias. The backend's enabling flag is broader than the model-shape gate;
that alone is not evidence of an incorrect selection.

[V4.1 integration](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/srt/layers/attention/dsv4/v41_indexer/litetopk.py)
covers ratio-1/2 FullTopK and SparseTable decode/target-verify paths after
chunked fallbacks. It requires page size 128, K=512, packed Q shaped
`[rows, 1, 32, 64]`, scales `[rows, 1, 32]`, weights `[rows, 32]`, and contiguous
output. The 64-byte Q head represents 128 FP4 elements. The actual
[fused Q packer](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/python/sglang/kernels/jit/csrc/deepseek_v4/fp4_indexer_rope.cuh)
rounds scaled head weights to BF16 before storing FP32, so converting
**those weights** back is exact; arbitrary FP32 weights need not survive.

Metadata clears the optional plan/indices every forward. It rejects chunked
or list schedules and nondivisible request grouping, creates
`arange(rows) // tokens_per_request`, reuses a matching decode schedule, and
builds grouped verify metadata. The sparse candidate publication retains
its alternate-stream wait and logits `record_stream`. These ownership rules
are needed in addition to selector correctness.

## Reconcile with current SGLang before transferring the patch

The comparison uses main `c24285bde97c3e6424cd5d8cb35788889034d8ea`, already
pinning DeepGEMM 0.2.6. All 1,193 lines of the head-to-main comparison for
V4.1 `metadata.py`, `full_topk.py`, `scoring.py` and `sparse_table.py` were
read. Current DSA producer, writer, scheduling and selection call paths were
inspected separately; this is not a claim to have read every unrelated
change in both complete DSA files.

- Current [DSA paged dispatch](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsa/dsa_indexer.py)
  still resolves legacy `fp8_paged_mqa_logits`, uses the existing FP32 index-K
  scales and FP32 output, and supports native verify when its actual length
  shape permits. The backend's separate MLA `kv_scale_ue8m0` setting does not
  prove the index-K cache uses UE8M0. Porting LiteTopK entails a producer and
  numerical/cache migration, not just changing the selector.
- DCP decode calls [dcp_local_topk](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsa/dsa_dcp_sharded.py),
  using distinct local/global lengths, distributed selection/merge, local
  physical slots and per-row local counts. The optional fused path has its
  own eight-rank SM100/multicast gate. A local LiteTopK result cannot bypass
  that protocol. Current cache writers also preserve DCP ownership and
  layer-split write targets; current verify metadata can update causal
  lengths in place. Carry these paths forward rather than replacing them
  with the old branch's metadata/writer logic.
- Current [V4.1 FullTopK](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/v41_indexer/full_topk.py)
  and sparse candidate selection already route BF16 logits through
  [topk_transform_paged_bf16](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/indexer.py)
  to DeepSelect's physical-slot API. A stale
  “add bf16 topk” comment does not override that implementation. This review
  checks the caller contract, not the complete DeepSelect kernel lineage.
- Current [metadata admission](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/layers/attention/dsv4/metadata.py)
  requires the varlen flag, non-JIT scheduling, at least 96 total rows and
  at most 29,039 rows per chunk for modern `indices` metadata. Other cases
  can use legacy schedules even with BF16 output. LiteTopK's old
  `_row_schedule` inference from “not TileLang” is insufficient on this
  mainline. The chosen producer, request grouping, indices and schedule
  format must agree.
- Current V4.1 prefill shares key slots/plans, supports query-row sharding
  and fused slot mapping, and retains per-layer candidate graph buffers
  with readiness events. Decode argument grouping is centralized in
  `DeepGEMMDecodeData.paged_logits_args`. Preserve these changes; they are
  integration context, not benefits introduced by LiteTopK.

## Validation and performance limits

The committed [GPU tests](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/test/registered/kernels/ops/attention/test_litetopk_decode.py)
contain 23 parameterized cases: eight standalone configuration/data cases,
two launch-policy cases, two shared-page-table cases, two histogram/fallback
cases, two selector graph cases, four DSA producer-chain cases, and three
V4.1 route cases. They cover ties/signed zero/infinities, million-column
rows, changed replay data, zero/wrong/off-by-one histograms and capacity four.
The standalone graph test copies a histogram before replay; it is not a
captured producer test. DSA tests run the real producer and two serial
selector calls. V4.1 tests mock projection/packing but run real metadata,
producer, selection and candidate publication inside a graph, including
next-n 1/4/6, changed lengths and nonidentity pages. Selected-score multiset
parity with the default does not require identical tie-selected slots.

The [14 CPU cases](https://github.com/sgl-project/sglang/blob/01e293958e709be59b934e0561f4122b92d4aa77/test/registered/unit/layers/attention/test_litetopk_decode.py)
check mocked capability, growth/capture ownership, causal cloning and
schedule/field handling. They are useful behavioral tests, not GPU evidence.
Exact-head queries returned 18 workflows, 122 check runs and no statuses.
Base [39106165330](https://github.com/sgl-project/sglang/actions/runs/39106165330),
extra [39106165283](https://github.com/sgl-project/sglang/actions/runs/39106165283)
and AMD [39106165266](https://github.com/sgl-project/sglang/actions/runs/39106165266)
fail the missing-`run-ci` gate; relevant runtime tests are skipped. Passing
title/lint checks do not verify the reported 23 GPU or 14 CPU results.

The body reports 18 TP1 serving smokes with eight-layer GLM-5.2, with/without
MTP, and six-layer V4.1: all generated 256 tokens. It reports 203,202 exact
selections, 22,912 causal verify-length rows and 1,008 live V4.1 score-parity
rows. These are truncated-model checks, not full-model accuracy evaluations.
V4.1 target verify, TP>1, DP attention, DCP and TBO were not served. The
reported 92/992 below-threshold selections concern the historical top-k-v2
control, not an independently reproduced defect in current main.

Historical B200 timings use real GLM layers 0/38 and V4.1 layer 20, one-call
CUDA graphs, a 1 GiB L2 flush, CUPTI first-to-last-kernel duration and
interleaved arms with A/A. They include producer histogram cost:

- DSA: 383 cells, 287 within one rank's KV pool. Against the old migration,
  geometric mean 2.934x, minimum 1.191x; within-pool mean 3.048x. The padded
  million-column conversion alone cost 92–95 us. Selector-only mean is
  1.623x against pre-widened top-k-v2, with 378/383 wins. Against BF16
  DeepSelect, mean 1.317x but 44 losses, minimum 0.870x at batch 16/16k.
- V4.1: 90 core plus 52 extra cells. Against old conversion/top-k, core mean
  4.515x and all 142 cells faster. Against BF16 DeepSelect, core mean 1.395x,
  but the 62 within-pool core cells average 1.134x; beyond-pool cells average
  2.203x. There are 28 losses across the full grid, minimum 0.821x at batch
  16/4k/five tokens. Feeding DeepSelect from the same producer reduces the
  reported gain to 1.051x; selector-only ratios are 1.433x versus DeepSelect
  and 2.012x versus top-k-v2. Grouped producer scheduling explains part of
  the chain gain. Histogram overhead reaches about 14% at five/six tokens.

The newer migration converts only live prefixes, and current V4.1 already
uses BF16 selection. Current DSA also has a different numerical producer.
Remeasure those exact callers, realistic within-pool shapes, schedule work,
allocation/graph retention and fallback rates before recommending a port.
No kernel ratio above proves serving throughput or accepted-token latency.

## Reviewed file groups

All 41 files are covered: selector and ops/runtime owners; DSA indexer and
backend; V4.1 helper, FullTopK, SparseTable, metadata, backend and registration;
environment flag; inherited cache writers, FP8 helpers, DeepGEMM adapter,
indexer/compressor/KPool callers, memory pool and serving defaults; the paged
MQA helper and FP4 benchmark; seven kernel test files and four unit test files.
The inherited migration was reviewed as part of this diff as well as against
its later head. Its later PD/storage changes are not silently attributed to
this earlier LiteTopK stack.
