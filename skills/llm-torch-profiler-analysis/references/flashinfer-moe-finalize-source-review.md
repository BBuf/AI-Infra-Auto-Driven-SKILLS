# FlashInfer indexed MoE finalize and all-reduce

Reviewed on 2026-11-09 (Asia/Shanghai). Historical source is pinned to the
final PR head below; current comparisons use FlashInfer
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU experiment was rerun.
This API consumes permuted expert rows and an inverse index map. The
[MoE reduction API](flashinfer-moe-collective-source-review.md) instead
consumes dense expert-major rows and FP32 scales; the two contracts are not
interchangeable.

## PR #4875 — feat(cake_comm): add Cake MoE finalize all-reduce backend

- Link: https://github.com/flashinfer-ai/flashinfer/pull/4875
- State: merged on **2026-09-13**, head
  `d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2`, base/merge-base
  `7629d2181a8858d7acd45820cb05f3098d76a895`, merge
  `488ffbd9fe4455c3b4c3030cf668b5603c6698e4`.
- Coverage: full 40,641-line diff, all 81 files, +40,009/-11, body,
  eight issue comments, two reviews, six inline comments and 33 commits.
  All 24 device files were reviewed as twelve complete unique bodies plus
  twelve byte-identical architecture copies. All 48 bindings were compared:
  after normalizing only generated symbol/namespace names, four distinct
  dtype/PDL bodies remain; each was read. Every world-size and quantization
  difference was inspected. The original manifest verifier passed on Python
  3.11 for all 48 routes, source lengths/digests, closures and argument plans.
- Reviewed files: all device/binding sources and the import manifest under
  `csrc/cake_moe_finalize_allreduce_fusion/`; the complete
  `flashinfer/jit/cake_moe_finalize_comm.py`; changes to
  `flashinfer/comm/{trtllm_ar,allreduce}.py`; complete new
  `tests/comm/test_cake_moe_finalize_{allreduce,dispatch}.py`; changed unified
  API tests and their existing finalize worker/reference/graph path; the
  complete `benchmarks/comm/bench_cake_moe_finalize_allreduce.py`, its CUPTI
  timing implementation, and `.pre-commit-config.yaml`.
- Motivation: offer a source-built Blackwell implementation that gathers
  routed expert results, combines their weights and optional shared expert,
  exchanges rank partials, adds residual, and emits RMSNorm and optional
  packed FP4 output. Direct `backend="cake"` and unified
  `moe_finalize_backend="cake"` are explicit opt-ins; the TRT-LLM default
  remains. This changes a shared communication API, not one model class.

### Historical admission and ownership

The [original loader](https://github.com/flashinfer-ai/flashinfer/blob/d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2/flashinfer/jit/cake_moe_finalize_comm.py)
accepts SM100/SM103, FP16/BF16, TP2/4/8, hidden width 7168, top-k 4/8 and
positive token count. Input is contiguous `[num_permuted_rows, 7168]`, which
can contain expert padding. The Int32 inverse map and activation-dtype route
scales are both `[T, K]`. Residual/shared-expert/output tensors are `[T, H]`,
gamma is `[H]`, and all tensors share a CUDA device. Residual and norm outputs
are required. `weight_bias` supports zero or one; absent routed scaling means
one. Negative inverse indices are skipped in the device code, but positive
index bounds are not inspected by the host validator. Producer validity is
still a caller responsibility.

The contiguous Int64 workspace table has exactly `3 * W + 1` pointers.
The public guard already uses finalized token bytes rather than padded expert
bytes. The new Cake loader repeats that check:

```diff
+    required_lamport_comm_size = _lamport_comm_size_bytes(
+        tokens,
+        7168,
+        residual_in.element_size(),
+        world_size,
+    )
```

This is a maximum-payload guard of 2,145,386,496 bytes, not a check of the
allocations behind the pointer table. All ranks must use compatible backing
storage, capacities and phase state. Keep the workspace alive across replay
and serialize independent uses of its mutable control words.

Profiles `110` and `111` mean residual/norm output and additional FP4/scales,
respectively. Historically the two quantization buffers had to be supplied
together using the activation dtype as a storage container. Packed FP4 needs
`T * H / 4` such 16-bit elements; the scale check required
`round_up(T, 128) * round_up(H / 16, 4)` **elements**, although the generated
kernel stores E4M3 **bytes**. Do not carry this historical over-allocation
requirement into the current API described below.

The original source manifest seals 72 CUDA files and 48 route closures for
architecture, dtype, world size, output profile and PDL. JIT builds separate
device/binding translation units with architecture flags and fast math;
unused optional buffers receive cached one-element placeholders. The grid
is `floor_to_multiple_of_4(min(SM_count, 4*T))`, with 224 threads per CTA,
four CTAs per cluster and 256 dynamic shared-memory bytes. Binding validation
requires positive X divisible by four and Y/Z exactly one, and launches on
the current device stream. PDL selects a launch attribute; it is not a
separate numerical algorithm.

### Gather, rounding, Lamport exchange and quantization

Each four-CTA cluster covers one 7168-element row with eight elements per
thread, then iterates persistently over assigned tokens. The
[BF16 TP2 device source](https://github.com/flashinfer-ai/flashinfer/blob/d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2/csrc/cake_moe_finalize_allreduce_fusion/sm_100a/cake_trtllm_moe_finalize_bfloat16_ws2_o110_device.cu)
and every dtype/world/profile difference preserve this order:

1. Gather each valid routed row, multiply by its activation-dtype route
   scale, round the contribution, and round after each addition. Apply
   routed scaling and optional shared-expert addition with further
   activation-dtype rounding.
2. Canonicalize zero to positive zero and publish the local vector to every
   peer's rank slice. Poll every low/high 16-bit payload half against the
   negative-zero sentinel, then add ranks in order with rounding after each
   addition. TP2 also uses rank order here, unlike the original reduction
   API's local-first TP2 path.
3. Add residual, round and store it. Reduce FP32 squares through descending
   warp shuffles, seven warp partials and four-CTA distributed shared memory.
   Apply reciprocal RMS and `(gamma + weight_bias)`, then round norm output.

The triple-buffer protocol reads the communication stride before
`griddepcontrol.wait`, while phase/data consumption follows the wait.
Each CTA increments the control counter after entering the protocol; block
zero waits for **entry**, then rotates the phase and resets the counter.
That counter is not proof that all output writes have completed. Cleaning
uses the previous recorded payload length in the older phase. Preserve this
ownership protocol when reasoning about PDL or changing launch residency.

The [FP4 epilogue](https://github.com/flashinfer-ai/flashinfer/blob/d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2/csrc/cake_moe_finalize_allreduce_fusion/sm_100a/cake_trtllm_moe_finalize_bfloat16_ws2_o111_device.cu)
quantizes the **rounded norm output**. It reduces all four packed pairs per
thread, then adjacent lanes, to obtain a 16-value amax. The final source fixes
the review's discarded-partial-max bug. It rounds `amax / 6` through E4M3,
uses an approximate reciprocal of that rounded scale, and packs E2M1 values.
Every even lane writes the scale byte in SWIZZLED_128x4 order. The public
loader passes global scale factor one. Padded scale rows remain unwritten;
allocation size is not a promise that every byte was initialized.

### What the tests and CI establish

`test_cake_moe_finalize_dispatch.py` has six CPU cases for default/explicit
dispatch, invalid backend and selector forwarding. The unified test adds a
CPU rejection for a non-TRT-LLM workspace and one real Cake case: TP2, FP16,
T16, K8, shared expert, both PDL settings. That case uses random inverse
indices with duplicates, repeats eager calls, then captures three calls and
replays once. Inputs and metadata do not change across replay. Its reference
sums in FP32 and tolerates `atol=rtol=0.2`; it is not an exact rounding oracle.
Its GPU gate checks device count, not Blackwell architecture, and its shared
multiprocessing helper still joins without a deadline.

The standalone distributed file has six top-level dtype/world-size cases,
gated on homogeneous SM100/SM103 hardware. Each tests both PDL modes for
T1/K4/no-shared/nonquantized and T16/K8/shared with and without FP4. Its
reference preserves intermediate rounding and rank order; residual/norm
tolerance is `atol=rtol=0.01`. Inputs are bounded and inverse indices are
identity. Each backend/case gets a fresh workspace and one eager call.
The subprocess deadline/terminate/kill cleanup was repaired during review.

FP4 is decoded and compared to the unquantized norm with `atol=1.0,
rtol=0.1`. Tests do not demand equality of Cake/TRT-LLM quant bytes or scales,
nor explicitly place the maximum in each formerly dropped pair. The new
standalone test does not cover changed-index graph replay, padded expert
rows, invalid positive indices or multi-phase reuse; the separate unified
case provides the narrower repeated-call and graph coverage above.

The body's final-head failure note is stale. A later
[PR Test run](https://github.com/flashinfer-ai/flashinfer/actions/runs/34724572444)
passed for the final head merged into its base. Its
[H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/34724572444/job/103636488890)
checks out merge `d1604c4` and records six dispatch passes, one unified CPU
pass with 37 GPU skips, and six standalone distributed skips: **seven CPU
passes, 43 skips** across these three files. This does not execute the new
Blackwell collective. The GitLab success comment reports 18/19 jobs and is
not independently inspected Blackwell execution evidence. The body also
retains older CPU counts from a JIT test file removed before the final diff.

### Timing evidence and reproducible harness limits

The body reports 30 selected B200/B300 rows: TP2/4/8 times five profiles on
each architecture, comparing standalone Cake, the exported implementation,
and TRT-LLM from FlashInfer 0.6.14. Reported geometric baseline/export ratios
are 1.02888 on B200 and 1.04431 on B300. These are historical operator
measurements, not a deployment-wide result. B200 TP8 BF16 T16 without shared
expert regresses from 44.6235 to 45.3275 microseconds, despite passing the
report's baseline/export floor of 0.98.

The selected matrix uses cold L2, CUPTI without fallback, balanced arm order,
100 ms warmup, 250 ms measurement and the median of iteration-wise rank
maxima. B200's v4 protocol adds a host rendezvous after L2 flushing; B300's
v3 uses only a completion fence. Their combined geometric mean mixes these
protocols. The separate registered benchmark reports the maximum of rank
medians, which is a different statistic. The export/source acceptance floor
of one also means a *faster* export can fail it; it is a publication-comparison
policy, not a speedup threshold.

The body retains an excluded earlier B200 TP8/T1 sample and replaces that
row after disabling a folding specialization, on another hardware cohort.
The other fourteen B200 rows retain their earlier measurements. Do not label
all fifteen rows as a fresh run of one revision. The recorded publication
head `6e7998f6ed75f3be52be2088df0efaae063c23b2` and final head have identical
generated source/manifest, loader, new standalone tests and committed
benchmark bytes; that does not retroactively align the earlier cohorts.
The reported sanitizer, end-to-end and canary promotions remain author
reports rather than reruns in this audit.

The committed [benchmark](https://github.com/flashinfer-ai/flashinfer/blob/d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2/benchmarks/comm/bench_cake_moe_finalize_allreduce.py)
is a two-provider harness, defaulting to TRT-LLM/Cake/TRT-LLM ordered legs,
five warmups and twenty measured iterations. It saves every rank's samples,
iteration-wise maxima, medians, baseline-leg spread, manifest hash and device
metadata. Every leg has separate workspace and pre/post numerical checks,
but only residual/norm are checked even when FP4 output is enabled. The
default inputs use identity indices, weight bias zero and routed scaling
2.5 when shared output is present; they do not reproduce all unified
profiles in the body's external three-arm matrix.

The [CUPTI helper](https://github.com/flashinfer-ai/flashinfer/blob/d6c57c20ca5d667e73d54ea5018dcb0d2fdbd3c2/flashinfer/testing/utils.py)
flushes L2 and synchronizes locally before each measured launch, then
synchronizes after it. The harness has distributed barriers around each
leg, not a per-iteration post-flush host rendezvous. CUPTI reports the span
from the earliest to latest associated GPU activity; collective waiting is
included, and arrival skew is therefore relevant. No CUDA graph is used for
timing. It cannot reproduce the body's v4 rank-arrival control merely by
selecting `cold_l2_cache=True`.

### Current source comparison and profiling decision

The current [Python loader](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/jit/cake_moe_finalize_comm.py)
and complete [C++ dispatch binding](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
were read alongside the current
[direct API](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/comm/trtllm_ar.py)
and [unified route](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/comm/allreduce.py).
The 48-route manifest is gone: 63 source programs and one binding compile
into one module per architecture. Validation moved to the binding, and
FP4/scales now accept any contiguous CUDA element type with sufficient
**byte** capacity. The historical verifier's variadic `PurePath.relative_to`
call fails under local Python 3.14; current code no longer uses that path.

The current binding retains 224 threads/four-CTA clusters, but allocates
7424 shared-memory bytes and bounds the grid by both per-kernel occupancy
(at most five CTAs per SM) and active cluster capacity. The two-shot branch
is tested before small-token selection: TP4 starts at T72 and TP8 at T56;
an early form applies through T128. Deferred forms start at T1536/768, with
an exclusion for TP8/K4/no-shared/no-quant. Shared-prefetch variants have
their own availability gates. Other small-token cases use T<=64, with a
conditional extension through T71. TP2 additionally selects quad-gather
through T142 after the small path, and K4 large-token programs above T512
(T1280 with shared expert). Record the actual symbol and dimensions rather
than deriving the route from the loader's abbreviated small-token docstring.

These current selector/launch facts do **not** constitute a complete review
of all 63 successor device programs or their originating PRs. In particular,
the historical one-shot kernel review cannot validate current two-shot,
deferred or shared-prefetch numerical/protocol equivalence. Those source
lineages, including catalog references #5852/#5870, remain separately
reviewable. Current unified Cake dispatch still requires a TRT-LLM workspace;
the MNNVL CuTe-DSL finalize and FP8-finalize routes are distinct providers.
The unified Cake route also supplies routed scaling as `None`/one, so a
direct benchmark with 2.5 is not automatically representative of it.

For a serving trace, retain inverse-map construction, expert padding,
dtype conversions, route weights, shared-expert ownership and downstream
quantized consumption in the measured boundary. Check numerical behavior
and rank-arrival/cache regime before interpreting a shorter finalize symbol
as a service improvement.
