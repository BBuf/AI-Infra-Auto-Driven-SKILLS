# FlashInfer indexed MoE finalize and all-reduce

Reviewed on 2026-11-09 (Asia/Shanghai). Historical source is pinned to the
final PR heads below; current comparisons use FlashInfer
`31c146b16789f112e9b9861bd79247d4701e437a`. No GPU experiment was rerun.
This API consumes permuted expert rows and an inverse index map. The
[MoE reduction API](flashinfer-moe-collective-source-review.md) instead
consumes dense expert-major rows and FP32 scales; the two contracts are not
interchangeable.

The later [five-family source consolidation](flashinfer-comm-source-consolidation-review.md)
reviews #7054's removal of unused finalize loader exports alongside union
source folding; it does not add a finalize performance optimization.

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
lineages remain separately reviewable. The #5852 host consolidation and
#5870 template are reviewed below. Current unified Cake dispatch still requires a TRT-LLM workspace;
the MNNVL CuTe-DSL finalize and FP8-finalize routes are distinct providers.
The unified Cake route also supplies routed scaling as `None`/one, so a
direct benchmark with 2.5 is not automatically representative of it.

For a serving trace, retain inverse-map construction, expert padding,
dtype conversions, route weights, shared-expert ownership and downstream
quantized consumption in the measured boundary. Check numerical behavior
and rank-arrival/cache regime before interpreting a shorter finalize symbol
as a service improvement.

## PR #5852 — consolidate MoE finalize bindings and source imports

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5852
- Title: `refactor(cake_moe_finalize_allreduce_fusion): drop the SM103 copy, fold the PDL/binding clones and replace the import manifest`
- State: merged on **2026-10-01**, head
  `e7e4a312d6517285ef6f085b51d35a9aae9d272b`, base/merge-base
  `909aa7b4050ec598c65f9935eb9311ad92acc4a6`, merge
  `6eb9eb41e4389a1204a283373a9619732f184929`.
- Coverage: full 31,823-line diff, all 67 files, +544/-30,555, body,
  ten issue comments, one review, no inline comments and all five commits.
  The 61 deleted files are byte-identical to files reviewed for #4875:
  48 bindings, twelve SM103 device copies and the manifest. All twelve
  retained SM100 device files are also byte-identical across #4875, this
  PR's base and its head. The old loader, benchmark and two standalone
  test files match their reviewed #4875 versions; every change was read.
- Reviewed files: all changed files under
  `csrc/cake_moe_finalize_allreduce_fusion/`, the complete new 167-line
  `flashinfer/jit/cake_moe_finalize_comm.py` and 270-line dispatch binding,
  `flashinfer/comm/trtllm_ar.py`, both
  `tests/comm/test_cake_moe_finalize_{allreduce,dispatch}.py` and the complete
  401-line `benchmarks/comm/bench_cake_moe_finalize_allreduce.py`.
  Supporting review includes `csrc/tvm_ffi_utils.h` validation macros,
  `gen_jit_spec` flags, the CUPTI timing helper and current loader/API/binding.
- Motivation: eliminate duplicated architecture/PDL bindings and manifest
  interpretation around an unchanged indexed finalize algorithm. The shared
  communication API retains its TRT-LLM default and explicit Cake opt-in.
  This is a host implementation and packaging change across model callers.

### One module, one validator and byte-sized outputs

The [new loader](https://github.com/flashinfer-ai/flashinfer/blob/e7e4a312d6517285ef6f085b51d35a9aae9d272b/flashinfer/jit/cake_moe_finalize_comm.py)
builds the twelve retained device translation units plus one binding into a
cached module per architecture. SM103 compiles the SM100-directory sources
with SM103 flags. Architecture and SM-count queries are cached per device;
selection is a C++ `[dtype][world-size][quantized-output]` table. PDL becomes
a runtime launch attribute. The 224-thread/four-CTA-cluster launch, 256-byte
dynamic shared memory and `floor4(min(SM_count, 4*T))` grid remain unchanged
at this historical head. Removing the loader's explicit fast-math flag does
not disable fast math: `gen_jit_spec` still enables it by default.

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/e7e4a312d6517285ef6f085b51d35a9aae9d272b/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
checks contiguous CUDA tensors on one device, activation dtype/shape,
positive T and expert-row count, H7168, K4/8, TP2/4/8, rank, positive epsilon,
weight bias zero/one and the exact Int64 workspace-table length. Optional
shared/quant pointers can be null instead of cached placeholder tensors.
FP4 and scale outputs remain paired, but their element dtype is unrestricted;
capacity is checked in bytes. The new scale requirement is explicit:

```diff
+    const int64_t padded_rows = (tokens + 127) / 128 * 128;
+    const int64_t padded_columns = (kHiddenDim / 16 + 3) / 4 * 4;
+    const int64_t scale_bytes = padded_rows * padded_columns;
+    TVM_FFI_CHECK(byte_size(scale_out.value()) >= scale_bytes, ValueError)
+        << "scale_out needs at least " << scale_bytes
+        << " bytes for the padded SWIZZLED_128x4 layout";
```

Packed FP4 still needs `T * 7168 / 2` bytes. The old 16-bit scale-container
check demanded twice the physical scale bytes. Tests now use UInt8 packed
FP4 and E4M3 scale storage; this corrects allocation admission without
changing the device quantizer or initializing its unwritten padding.

Validation is not entirely equivalent. The exported internal
`run_cake_moe_finalize` helper loses the old Lamport maximum-payload check,
and the new C++ binding does not restore it. It also narrows token count to
Int32 without an explicit upper-bound check. The
[public API](https://github.com/flashinfer-ai/flashinfer/blob/e7e4a312d6517285ef6f085b51d35a9aae9d272b/flashinfer/comm/trtllm_ar.py)
still rejects `residual_in.numel() * element_size * world_size > MAX_COMM_SIZE`
before calling Cake. This separation persists at the current pin: use the
public boundary or enforce its payload limit when testing the lower-level
helper. Neither version inspects actual capacities behind workspace pointers
or positive inverse-index bounds; those remain caller contracts.

The body reports 24 normalized SASS hashes equal before/after on B200 and
B300. This audit verified source-byte identity, not a CUDA recompilation.
The final diff leaves `.pre-commit-config.yaml` unchanged, despite an earlier
bot walkthrough describing a narrower exclusion. The handwritten binding
remains inside the generated directory's formatting exclusion.

### Benchmark cleanup changes its evidence contract

The [trimmed harness](https://github.com/flashinfer-ai/flashinfer/blob/e7e4a312d6517285ef6f085b51d35a9aae9d272b/benchmarks/comm/bench_cake_moe_finalize_allreduce.py)
retains separate workspaces, TRT-LLM/Cake/TRT-LLM ordered legs, five warmups,
twenty measured iterations, cold L2 and the median of iteration-wise rank
maxima. It builds the entire Cake module before the measured legs. Sample
counts are now checked on every rank after gathering, avoiding the previous
single-rank early raise before that collective. CLI lists become
space-separated `nargs="+"` values; old comma-separated commands need updating.

Two removed checks/outputs matter to profiling:

- The harness drops its explicit CUPTI >=13 check and only imports `cupti`.
  The unchanged
  [timing helper](https://github.com/flashinfer-ai/flashinfer/blob/e7e4a312d6517285ef6f085b51d35a9aae9d272b/flashinfer/testing/utils.py)
  still falls back to CUDA events if the installed version is below 13 or
  distribution metadata cannot be read. Thus the comment claiming no
  fallback, and JSON `enable_cupti: true`, do not establish the timer used.
  A CPU-only execution of the unmodified extracted helper with mocked
  imports/metadata confirms both fallback branches; no GPU timings were run.
- Raw per-rank samples and iteration-wise maxima, package/source identity
  and schema/fallback metadata are removed from the report. Remaining
  medians/minima cannot reconstruct dispersion, ordering or confidence
  intervals. Preserve these inputs separately for a new comparison.

Residual/norm pre/post checks remain; quantized output is allocated but not
numerically checked by this harness. There is still no per-iteration
post-flush rank rendezvous. CUPTI's GPU interval includes waiting for peer
rank launches, so removing CPU work can shorten a collective's measured
residence without changing its device instructions. The old first-use source
hash is cached; do not describe it as a per-call 3.9 MB hashing cost.

### Performance and validation limits

The body reports TP2's 128-shape matrix on each architecture. B200's two-run
per-shape median ratios have geometric mean 1.025, but its drift-normalized
per-run geometric mean is 1.011, and an individual raw ratio falls to 0.962. B300's
four-run per-shape median ratios have geometric mean 1.479 and minimum 0.816;
four rows fall below 0.98. Restricting to the last two, separately warmed
runs gives geometric mean 1.674 and minimum 1.008 for the **per-shape
medians of those runs**; the drift-normalized per-run geometric mean is 1.355.

The body's claim that all warmed runs are at least one conflicts with its
own table: FP16/T128/K8/PDL0/profile110/no-shared has a warmed run at **0.826**,
and FP16/T2048/K8/PDL1/profile110/shared has one at **0.961**. Keep these
individual regressions even though their two-run medians exceed one.
Alternating tree order and a TRT-LLM control expose drift; they do not prove
that it cancels. The proposed slow-rank-launch explanation is compatible
with the source, but the body supplies no CPU/launch timeline that isolates
that causal contribution. Do not recast the 1.674 ratio as a device-algorithm
or serving speedup.

TP4 performance runs were terminated after more than fifty minutes on both
nodes. TP2 does not validate TP4/8 timing or their distinct communication
participants, even though host dispatch is shared. The separate body-reported
B200/B300 correctness runs cover six standalone dtype/world-size cases and
two unified cases per node; they were not rerun here.

The final-head [PR Test run](https://github.com/flashinfer-ai/flashinfer/actions/runs/36826473502)
passed. Its [H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/36826473502/job/110253514626)
checks out `f300495`, explicitly merging final head `e7e4a312` into base
`909aa7b4`. The three relevant files yield **six CPU passes and 43 GPU skips**:
five dispatch passes, one unified CPU pass/37 skips, and six standalone skips.
This is not Blackwell execution evidence. Six AOT import/build jobs pass;
the discussion's GitLab start announcements contain no final result.

The dispatch cleanup removes the direct default-TRT-LLM forwarding test.
The retained unified selector test mocks that direct API, so it does not
replace the deleted boundary's coverage. The distributed test otherwise
retains #4875's narrow numerical and graph coverage, with only output-buffer
types/sizes changed. Treat deletion of duplicate bindings as verified
deduplication; do not classify every removed check or test as redundant.

Current source retains this single-binding and byte-buffer design, but adds
the 63-program selector and different launch resources described above.
Those later device schedules still require their own source review before
borrowing performance or equivalence claims. The intermediate template
consolidation is reviewed next.

## PR #5870 — regenerate the twelve kernels as one templated source

- Link: https://github.com/flashinfer-ai/flashinfer/pull/5870
- Title: `refactor(cake_moe_finalize_allreduce_fusion): regenerate the 12 kernels as one templated source`
- State: merged **2026-10-04**, head
  `cefb12a93f997b4e6e218a81e66cea53ee39d679`, API base
  `1cad38165e5fe9d3007a07f79b3d5a2d73d6fce4`, actual merge-base
  `883825a20f9b9ff92f64101168ed4c324642121c`, merge
  `613d12e08d3230f79e5d9e57a676164508525486`.
- Coverage: full 9,251-line diff, all fourteen files, +679/-8,480;
  body, all twenty issue comments, one approval, no inline comments and
  both commits. The twelve deleted device sources are byte-identical to
  the already reviewed #4875/#5852 files. The complete new **675-line**
  template and loader diff were read, including the second commit's poll
  rewrite. The body's 672-line count describes the earlier revision.
- Reviewed files: `csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu`,
  every deleted `sm_100a/*_device.cu`, and
  `flashinfer/jit/cake_moe_finalize_comm.py`. Binding, benchmark, three
  relevant test files and timing helper are byte-identical to #5852.
  Current source registration was checked separately.
- Motivation: remove the remaining device-source duplication while retaining
  the shared indexed-finalize API, kernel symbols, launch geometry and
  numerical protocol. No new dtype, world-size or quantization capability
  is introduced, despite the bot release notes presenting them as features.

### Template parameters and compiler-sensitive polling

The [new source](https://github.com/flashinfer-ai/flashinfer/blob/cefb12a93f997b4e6e218a81e66cea53ee39d679/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu)
has `finalize_body<T, WS, QUANT>`, dtype traits for BF16/FP16 conversion and
packed arithmetic, unrolled peer operations, and a compile-time FP4 epilogue.
Twelve wrappers preserve the old `extern "C"` symbols and 224-thread,
four-CTA-cluster declarations. The loader now compiles this translation unit
and the unchanged binding rather than twelve device units plus the binding.

The reviewed body retains per-contribution/per-addition rounding, canonical
positive-zero publication, negative-zero sentinel polling, ascending rank
reduction, rounded residual/norm and the FP4 scale layout described for #4875.
It also retains the entry-counter protocol, prior-phase cleaning and PDL
ordering; consolidation does not add recovery or concurrent-workspace safety.

The first revision used an unrolled loop and `pending |=` sentinel chain.
The second commit expands the peer loads and uses a logical fold expression
to reproduce the original compiler instruction selection:

```diff
+template <typename F, size_t... _i> __device__ __forceinline__ void poll_expand(F&& f, std::index_sequence<_i...>) {
+    (f(std::integral_constant<int, static_cast<int>(_i)>{}), ...);
+}
```

`poll_any_0` examines both 16-bit halves of every loaded word through a
C++17 `||` fold over `4 * WS` words. This is a concrete example where
equivalent source-level loop logic did not initially preserve SASS across
nvcc versions. The final-head
[gate report](https://github.com/flashinfer-ai/flashinfer/pull/5870#issuecomment-5935798950)
claims all 24 dtype/world/profile/architecture combinations match their
respective baselines under both nvcc 13.4 and 13.0. Instruction counts differ
between those compilers; equality is within each compiler/build comparison.
The report also corrects whole-library disassembly: extract each embedded
cubin before comparing, because ELF header lines can be attributed to the
preceding symbol when the object count changes from thirteen to two.
These SASS comparisons were not rerun locally.

### Keep rank arrival and unsuccessful benchmark gates visible

The unchanged harness inherits #5852's timer-fallback and missing-raw-sample
limitations. External reports name main `883825a2` and final head `cefb12a9`,
TP2, the default 128 shapes, alternating tree order and a warmup run. Their
rank-min and control-normalized estimators are additional analysis, not the
committed harness's iteration-wise rank-max median.

Initial four-round B300 results have rank-max geometric mean **0.893**
(minimum 0.617; 92/128 rows below 0.98), while rank-min is 1.001 and
TRT-LLM-normalized is 0.981. The author observes 13 versus 19–23 microsecond
rank medians affecting both providers. Rank-min helps investigate arrival
skew but discards the waiting rank's residence; it cannot replace rank-max
as the same latency metric or establish end-to-end parity.

The B200 A/A control itself has 22/128 rank-max rows below 0.98, minimum
0.880 and geometric mean 0.999. This shows substantial measurement variation;
it does not prove every A/B regression is noise. Follow-up rounds pool
multiple node cohorts. The
[final 22-round report](https://github.com/flashinfer-ai/flashinfer/pull/5870#issuecomment-5947768323)
still has two rank-max rows below 0.98, table minimum **0.970** and geometric
mean 0.999; control-normalized has three failures, minimum 0.965. Rank-min
has none, minimum 0.988. Its prose describes the two rank-max rows as
0.977–0.980, which does not reconcile with the table minimum. Preserve that
inconsistency rather than choosing the more favorable value. One run lost
its log; the author reports its JSON survives and supplies a 21-run summary.
The raw artifacts named in comments are not committed in this diff.

The author ultimately calls the remaining roughly 0.3-microsecond waiting-rank
difference persistent. Reported identical binaries do not erase process
placement, launch skew, clock state or changed build/loading effects. The
TP4 harness reportedly hangs at full GPU utilization on both baseline and
candidate; no TP4 performance result is supplied. Keep that as an unresolved
reported baseline-and-candidate failure, not a TP2-validated TP4 result.

### Test evidence and current applicability

No tests change in this PR. Author-reported B200/B300 runs exercise the
existing standalone dtype/world-size cases and two selected unified cases;
the underlying numerical, graph and invalid-input coverage remains #4875's.
The [final-head GitHub run](https://github.com/flashinfer-ai/flashinfer/actions/runs/36898390021)
passes six AOT jobs and the
[H100 targeted job](https://github.com/flashinfer-ai/flashinfer/actions/runs/36898390021/job/110491265369).
The log checks out `e3a70ba`, merging `cefb12a9` into API base `1cad3816`,
and records **six CPU passes/43 GPU skips** across the three relevant files.
An earlier failed run stops at authorization before test setup.

Discussion first reports internal pipeline 70939847 at 13/29 passed, with
baseline-attributed failures and unresolved infrastructure/results, then
pipeline 71161328 at 29/29. The later bot success is reported evidence; it
does not supply independently inspected per-test Blackwell logs here or
justify treating every earlier failure as proved unrelated.

At the current pin `cake_moe_finalize_kernels.cu` has been removed. The
current [module registry](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/flashinfer/jit/cake_moe_finalize_comm.py)
compiles separate program sources, with the shape/occupancy selection above.
Some symbols remain, so a familiar profiler name does not establish that it
executes this historical template. Record the source revision and selected
program before reusing its equivalence or timing conclusions.

## PR #6057 — pair prefetch, DSM slot ring and occupancy-sized grid

- Link: https://github.com/flashinfer-ai/flashinfer/pull/6057
- State: merged 2026-10-06; head
  `047f5352eab172571088437dba7893ec4c56ad62`, API base
  `cab5c6172454484701dfda0e6fa4fd639909b0b4`, merge-base
  `613d12e08d3230f79e5d9e57a676164508525486`, merge
  `55edb4288eb5c82fbfb5f110d1329bd75d0b9849`.
- Coverage: full 642-line diff, two files, +259/-128, body, thirteen issue
  comments, one approval, no inline comments and all three commit messages.
  Both base files match the reviewed #5870 head byte-for-byte.
- Reviewed files: complete changes to
  `csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu` and
  `cake_moe_finalize_allreduce_fusion_binding.cu`, with unchanged loader,
  benchmark, timer and three test files compared to #5870. The indexed public
  finalize function and unified dispatch are unchanged; intervening changes
  elsewhere in `trtllm_ar.py` are not part of this patch.
- Motivation: expose more independent routed loads, shorten the cluster
  normalization dependency, and keep more token work resident at large T.
  Admission, twelve entrypoints and the workspace ABI remain unchanged.

### Data dependencies and launch geometry

The [device change](https://github.com/flashinfer-ai/flashinfer/blob/047f5352eab172571088437dba7893ec4c56ad62/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu)
consumes routes in pairs. It fetches the next two inverse indices before
loading the current pair's valid expert rows and scales. Negative indices
skip both the load and contribution; positive-index validity remains the
producer's responsibility. Contributions are still scaled, rounded, added
and rounded in route order. This is load scheduling, not reassociation.

For RMSNorm, thread zero in each CTA writes its FP32 partial into its own
slot in **all four CTAs'** distributed shared memory. One release/acquire
cluster barrier makes the four slots available locally; every thread then
performs the same ordered sum and reciprocal square root:

```diff
+        float cluster_sum = 0.0f;
+        cluster_sum = cluster_sum + rms_slots[rms_parity * 4];
+        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 1];
+        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 2];
+        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 3];
+        float _rsqrt_0 = rsqrtf(cluster_sum / 7168.0f + epsilon);
```

The two banks occupy 32 bytes from shared-memory offset 136. Alternating
`rms_parity` prevents the next token from immediately overwriting the bank
being consumed. Correctness still depends on matching token iterations and
converged cluster barriers. The old two-barrier reduction/broadcast is gone;
rank-addition rounding, residual rounding, FP4 conversion and the Lamport
entry-counter/phase protocol remain as described above.

The [host change](https://github.com/flashinfer-ai/flashinfer/blob/047f5352eab172571088437dba7893ec4c56ad62/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
caches `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per device/kernel,
caps it at four CTAs/SM and launches
`4 * floor(min(ctas_per_sm * SM_count, 4*T) / 4)` CTAs. The launch still uses
224 threads/CTA, four CTAs/cluster and 256 dynamic shared-memory bytes.
At small T the grid is unchanged. This historical calculation does **not**
query active-cluster occupancy; the current launcher adds that bound.

The second revision polls locally published rank slices at TP2/TP4. TP8
retains the remote peer diagonal because the author's earlier local-poll
experiment lost about 4% at T1. That choice is superseded by #6128 below;
it is not evidence that TP8 fundamentally needs remote polling.

### Timing and correctness evidence

The body describes external paired in-process old/new arms with interleaved,
counterbalanced passes and geometric means of **rank-min** ratios. The
committed benchmark is byte-identical to #5870's TRT-LLM/Cake/TRT-LLM harness;
it does not implement that old/new protocol. Its CUPTI fallback and missing
raw-sample caveats therefore still apply. Rank-min is also a different
estimator from the committed iteration-wise rank-max median.

The first revision's TP2 aggregate is 1.673 on B200 and 1.370 on B300.
TP4/TP8 aggregates are smaller and retain individual regressions: B200 TP4
minimum 0.994; B300 TP4/TP8 minima 0.980/0.963. The second table compares
first versus second revision, not main versus final head; its B300 TP4
minimum is 0.991. Combined B200 TP2/T2048 ratios around 4.19 (K8) and 3.83
(K4) are operator rank-min results, not service throughput gains.

TP4/TP8 T2048 reportedly hangs intermittently in both Cake and TRT-LLM,
with ranks inside `cudaDeviceSynchronize`; launch blocking avoids it in the
reported runs. Only isolated processes that finish before a hard timeout
contribute results. Keep this completion selection and unresolved failure
visible. The absence of the newer cluster-capacity bound alone does not
establish the cause of those historical hangs.

The author reports full dtype/T/K/PDL/profile/shared-expert comparisons,
arange/adversarial routing, eight-seed FP64 checks at TP2/TP4, 1024 rank-equality
rows and sanitizer runs. No tests or external precision harness are added
by this PR. These are historical Blackwell reports, not local reproductions.
The [final-head H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/37415464574/job/112113090954)
checks out `f2f7cf8`, merging `047f5352` into `49a1cd2f`, and records
**six CPU passes/43 GPU skips**. All six AOT build/import jobs pass.

Discussion's internal pipelines report 11/29 and then 8/23 passed, including
environment/compiler failures and B200 timeouts; a later pipeline is only
announced. The author's attribution of red jobs to infrastructure is not
independent proof that every failure is unrelated. A successful public H100
run does not resolve the skipped Blackwell coverage or internal timeouts.

## PR #6128 — local owner-segment polling at TP8

- Link: https://github.com/flashinfer-ai/flashinfer/pull/6128
- State: merged 2026-10-06; head
  `f389e7d5a072d80d5e6a113710d9630a458c91ab`, base/merge-base
  `ea8135c0bdf641f54214e9ae777d5b75b5133079`, merge
  `a9d3d64fee145fa2be06e08063de7d6ca1c45095`.
- Coverage: complete 61-line diff, one file, +12/-23, body, four issue
  comments, one approval, no inline comments and the commit message.
- Reviewed file: the full change to `cake_moe_finalize_kernels.cu`; its base
  matches #6057's head. Binding, loader, benchmark, timer and tests are
  byte-identical to #6057. Public indexed finalize behavior is unchanged.
- Motivation: consume each peer's contribution from the local rank's
  published copy, avoiding repeated remote loads while waiting.

The [poll rewrite](https://github.com/flashinfer-ai/flashinfer/blob/f389e7d5a072d80d5e6a113710d9630a458c91ab/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu)
loads rank zero explicitly and expands the remaining ranks at compile time:

```diff
+            poll_expand([&](auto _pc) {
+                constexpr int p = 1 + decltype(_pc)::value;
+                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4 * p]), "=r"(_sysv_poll_group_0[(4 * p + 1)]), "=r"(_sysv_poll_group_0[(4 * p + 2)]), "=r"(_sysv_poll_group_0[(4 * p + 3)]) : "l"(workspace_local + (data_base + (long long)(p * total_access * 8) + (long long)(access_1 * 8))) : "memory");
+            }, std::make_index_sequence<WS - 1>{});
```

All supported world sizes now use local rank slices. Publication to all
peers, sentinel checks and rank-addition order remain unchanged. Local
polling still waits for peer publication; it does not remove collective
arrival skew or establish concurrent workspace safety.

The author attributes #6057's TP8/T1 regression to an earlier rolled-loop
code generation, reporting sixteen fewer SASS instructions for the new
straight-line local form. TP2/TP4 SASS is reported unchanged. The diff also
changes three quant-scale conversions from volatile to ordinary inline
assembly; the claimed unchanged SASS for all twelve kernels concerns that
conversion-only change, not the complete TP8 poll rewrite. Neither compiler
comparison was rerun in this audit.

The external rank-min experiment covers 192 TP8 rows per node class:
32 dtype/K/PDL/profile/shared-expert combinations at T1/2/4/8/16/128.
B200 ratios range from 1.097 at T1 to 2.048 at T128; B300 class A reports
1.070 to 1.990 at those endpoints, while class B reports 1.183 and 1.18328.
Keep node cohorts separate. The commit message contains earlier values;
the body is the later report. TP2/TP4 controls are approximately 0.998–1.007.
The broader correctness claim includes T2048, but the TP8 performance table
does not. This is not a resolution of #6057's large-token timeout caveat.

No test/harness changes accompany the claimed bit-identical and sanitizer
results. The [final-head H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/37470313974/job/112291973579)
checks out `6da3119`, merging `f389e7d5` into `ea8135c0`, and passes six CPU
cases while skipping 43 GPU cases. Six AOT jobs pass. The discussion contains
an internal pipeline start announcement, with no final result to inspect.

## PR #6166 — four-route gather and TP4 residual prefetch

- Link: https://github.com/flashinfer-ai/flashinfer/pull/6166
- State: merged 2026-10-08; head
  `693ac8b2b84dd52baf4274f74f2577fa3956db01`, API base
  `b8e81a7dae6c2aa5c9aa19c06e81e9a729271493`, merge-base
  `a9d3d64fee145fa2be06e08063de7d6ca1c45095`, merge
  `fb2b35145ef84ba25f56b58b3579ece14c242c0c`.
- Coverage: full 1598-line diff, one file, +1061/-381, body, ten issue
  comments, one approval, no inline comments and all three commit messages.
- Reviewed file: complete changes to `cake_moe_finalize_kernels.cu`, including
  both gather bodies, poll loops, residual/RMSNorm and quantized tails. The
  base kernel matches #6128; binding, loader, benchmark, timer, tests and
  indexed public finalize function remain unchanged.
- Motivation: overlap independent expert-row loads, and place independent
  residual loads before the collective wait where their live registers pay off.

The [final template](https://github.com/flashinfer-ai/flashinfer/blob/693ac8b2b84dd52baf4274f74f2577fa3956db01/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_kernels.cu)
selects the gather body by the world-size template literal:

```diff
+    int use_quad = 1;
+    int quad_ws_lit = WS;
+    if (quad_ws_lit < 4) {
+        use_quad = 0;
+    }
+    if (use_quad == 1) {
```

Within that TP4/TP8 body, residual prefetch has a separate gate:

```diff
+            int hoist_on = 1;
+            int hoist_ws_max_lit = WS;
+            if (hoist_ws_max_lit > 4) {
+                hoist_on = 0;
+            }
+            if (hoist_on == 1) {
```

| Historical program | Gather | Residual load |
|---|---|---|
| TP2 | Previous two-route body | After polling |
| TP4 | Four routes at a time | Before polling; addition remains after it |
| TP8 | Four routes at a time | After polling |

The quad body keeps four valid rows as raw 128-bit words, unpacking each
just before its scale step, and prefetches the next four indices. It retains
the exact route order and contribution/addition rounding. Residual loading
does not depend on rank partials; residual addition does. Gamma loading
remains after polling. Both bodies retain the local poll, two-bank DSM
reduction, FP4 scale layout, previous-phase cleaning and entry counter.
The body selection is compile-time world size, not a runtime T/K threshold.

### Keep rejected variants and measurement cohorts

TP8 residual hoisting was removed after B300 non-quantized T>=128 regressions:
individual ratio down to 0.93, T128 class means 0.980–0.985 and K4/T2048
means 0.975–0.980. Quad gathering remains. TP2 quad gathering was removed
after K4/shared-expert/T>=1024 losses: T2048 class means 0.978–0.991 and
individual rows down to 0.945. A runtime two-body fallback still lost about
1.5%, and token-gated hoisting recovered less than 1%. The final TP2 programs
are reported SASS-identical to the old ones, eight dtype/profile/architecture
comparisons; the final TP4/TP8 programs match the measured earlier revisions
in sixteen comparisons. Reported register counts are 56 at TP2/TP4 and 72
at TP8, retaining four CTAs/SM. These are author compilation reports.

The table mixes TP4 first-revision measurements and TP8 second-revision
measurements, supported by those reported SASS comparisons. Final-head
remeasurements cover two B300 nodes and one B200 node: TP4 T1/T16 means
1.081–1.096/1.089–1.102 on B300 and 1.090/1.096 on B200; unchanged TP2
controls span 0.995–1.002. The broader table reports TP8 small-T gains around
4–7%, but node coverage differs by cell. T128 mixes 16-row K4-only sets and
32-row K4/K8 sets; B300 TP4/K8/T2048 is unmeasured in the table. Do not
combine these ranges as one uniform full-matrix experiment or serving gain.
The external paired rank-min protocol is still absent from the unchanged
committed harness.

The final-head body reports tests on two B300 nodes and one B200 node:
per node, TP2 dispatch five passes, standalone two passes/four skips and
unified 22 passes/16 skips; standalone TP4 four passes/two skips and TP8
six passes. These counts do not establish full unified coverage at TP4/TP8.
It also reports bit-identical A/B rows, rank equality and zero device errors
from final-head eight-rank synccheck/memcheck; allocator capability-probe
host entries are separately qualified. No new tests or raw run artifacts
are committed in this diff.

The [final-head H100 job](https://github.com/flashinfer-ai/flashinfer/actions/runs/37707230253/job/113084511504)
checks out `6a9dd3b`, merging `693ac8b2` into `b5e50c20`, and records six CPU
passes/43 GPU skips. Six AOT jobs pass. All three internal pipeline comments
only announce starts; they do not provide completed Blackwell results.

### Applying this lineage to a current trace

These three PRs explain useful scheduling choices, not a universal dispatch
table for the November pin. The current template is retired. Current
[host selection](https://github.com/flashinfer-ai/flashinfer/blob/31c146b16789f112e9b9861bd79247d4701e437a/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
caps block occupancy at **five** and also bounds resident clusters with
`cudaOccupancyMaxActiveClusters`. It selects separate small, persistent,
TP2 quad/large and TP4/TP8 two-shot programs using token count, K, shared
expert and quantization. In particular, a historical TP2 decision against
unconditional quad gathering does not mean the current selector never uses
it. Read the selected program before applying that historical conclusion.
All successor device bodies and their intermediate PRs are not yet covered
by this review. Preserve the per-shape fallback, source revision, rank timing
estimator and resource bounds when turning these precedents into a new edit.
