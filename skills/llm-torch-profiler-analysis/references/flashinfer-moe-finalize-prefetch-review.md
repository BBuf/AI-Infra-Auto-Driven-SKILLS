# FlashInfer indexed finalize: small-token prefetch and dispatch

Reviewed on 2026-11-09 (Asia/Shanghai). This is a historical source review
of #7905/#8073/#8139/#8321, following the [two-shot/deferred lineage](flashinfer-moe-finalize-two-shot-review.md).
Current comparisons use `31c146b16789f112e9b9861bd79247d4701e437a`.
No GPU execution, SASS comparison, sanitizer or benchmark was run here.

## PR #7905 — retain packed operands and lower the TP4 crossover

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): issue the residual and norm weight ahead of the one-shot poll](https://github.com/flashinfer-ai/flashinfer/pull/7905).
- State: merged 2026-10-29 at 23:16:08 UTC; head
  `a81c2e14d2997002623b09eec0659229c57fece2`, API base
  `73beeb0e0a6c4e8de5c48a90a6c4d7b3f5a71a5b`, actual merge-base
  `9f5c44ea103e50840b57643465e6d0c40d6b349f`, merge
  `97beaeba99b4f6abd25ca37a3c9444f7cbe0eea1`.
- Coverage: full 1,703-line diff fetched; all unique changes reviewed across
  19 files, +611/-627. Complete BF16 and FP16 canonical deltas were inspected;
  every one of the 16 device deltas matches its dtype's canonical operations
  after bijective temporary renaming, preserving assembly strings, constants
  and order. All 40 base bodies match #7813 and the other 24 head bodies are
  byte-identical. All body prose, structured checks of 720 timing rows, both
  issue comments, both reviews, no inline comments and both commit messages
  are covered.
- Reviewed files: twelve `*_ws{2,4,8}_o{110,111}_small_kernel.cu` programs
  and four `*_ws4_o{110,111}_kernel.cu` persistent programs under
  `csrc/cake_moe_finalize_allreduce_fusion`; the host binding,
  `flashinfer/jit/cake_moe_finalize_comm.py`, and
  `tests/comm/test_cake_moe_finalize_allreduce.py`.
- Motivation: keep residual/norm-weight operands packed while waiting for
  peer data, defer conversion until after polling, and select the already
  available two-shot protocol earlier for TP4.

### What is moved, and what is preserved

Previously the changed one-shot programs loaded and unpacked residual before
the poll, then loaded/unpacked norm weight afterward. The
[BF16 small-token body](https://github.com/flashinfer-ai/flashinfer/blob/a81c2e14d2997002623b09eec0659229c57fece2/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_small_kernel.cu)
now loads residual and norm weight into four packed 32-bit words each before
the poll. The norm-weight load is introduced as:

```diff
+        unsigned int norm_weight_words[4];
+        {
+            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(norm_weight + (cluster_thread * 8) + 0);
```

An empty `asm volatile` consumes all eight words before the volatile workspace
poll. This is a compiler operand dependency, not a GPU synchronization
instruction or evidence that a later build achieved a particular overlap.
Residual is unpacked after the poll; norm weight is unpacked at its later
consumption point. Packed words remain live across polling instead of eight
unpacked residual floats. Do not infer register counts or occupancy without
the compiled program.

The loads are ordinary C++ `uint4` reads, unlike #7813's explicit
`ld.global.nc.v4.b32` residual path. Norm weight's old
`access_in_token * 8` equals the new `cluster_thread * 8`; residual retains
`access_1 * 8`. BF16 unpacking still uses bit shifts/masks; the
[FP16 body](https://github.com/flashinfer-ai/flashinfer/blob/a81c2e14d2997002623b09eec0659229c57fece2/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws2_o110_small_kernel.cu)
retains the two half-to-float conversions. Routing, rank order,
activation rounding, residual addition, RMS exchange and both output
epilogues are otherwise unchanged. These early reads still depend on the
existing producer/input readiness contract; moving conversion does not
authorize moving loads across that dependency.

The second commit changes only the TP4 two-shot constant, loader comment
and test mirror from 88 to 72. In the
[binding](https://github.com/flashinfer-ai/flashinfer/blob/a81c2e14d2997002623b09eec0659229c57fece2/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu),
two-shot remains ahead of small-token selection. With the default small
limit of 64, TP4 T72–87 therefore changes from persistent to two-shot.
The 40-program registry, TP8 T56, deferred T1536/T768 and its TP8 exception,
small extension to T71, TP2 large gates, residency queries and workspace
protocols retain #7813's definitions. Custom small limits must still respect
the earlier two-shot test.

### Validation and performance limits

The [test diff](https://github.com/flashinfer-ai/flashinfer/blob/a81c2e14d2997002623b09eec0659229c57fece2/tests/comm/test_cake_moe_finalize_allreduce.py)
changes only `TWO_SHOT_MIN_TOKENS[4]`. The generated boundary set substitutes
T71/72 for T87/88; because T71/72 were already present, the TP4 token set
shrinks from ten to eight values. The revised mirror still tests the boundary
and retains T1535/1536. Existing dtype/top-k/shared/profile/weight-bias/PDL
coverage, conditional kernel-name assertions, tolerant reference checks and
per-backend workspace reuse remain. There is no new byte-identity,
negative-route, graph-replay or model-accuracy test.

The public APIs, unified MoE test, CPU table test, committed benchmark and
timing helpers are byte-identical to #7813. Final-head
[run 40962198619](https://github.com/flashinfer-ai/flashinfer/actions/runs/40962198619)
has ten jobs and six successful AOT build/import jobs, but no requested
distributed test job. Setup checked merge `7d0bb1916` of the exact head
into the API base; H100 PR testing was disabled for runner capacity and
all A10G/T4 targeted scopes were empty. The summary found no JUnit artifacts.
The two earlier final-head runs failed unauthorized-status reporting before
setup. The bot approval attests a CODEOWNERS gap, not numerical correctness.

All 720 unique table shapes match the preceding review's inventory. They
use external CUDA 13.4 interleaved previous/new Cake measurements, rank-min
CUPTI times and median paired speedup ratios. Only the T1/2/4/8/16 rows run
changed kernels: 160 per world size, 480 total. No table row runs the changed
TP4 persistent program or falls in the moved T72–87 range.

| Cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| All 720 | 1.010 / 0.984–1.059 | 1.011 / 0.983–1.062 |
| Changed 480 | 1.015 / 0.984–1.059 | 1.016 / 0.983–1.062 |
| Unchanged 240 | 1.001 / 0.990–1.021 | 1.000 / 0.988–1.012 |
| TP2 small, 160 | 1.019 / 0.993–1.059 | 1.021 / 0.998–1.062 |
| TP4 small, 160 | 1.019 / 0.994–1.055 | 1.020 / 0.997–1.055 |
| TP8 small, 160 | 1.006 / 0.984–1.040 | 1.007 / 0.983–1.043 |

Among changed rows, 62 B300 and 59 B200 ratios are below one; TP8 accounts
for 53 and 57. The worst B300 row is TP8/T16/BF16/K8/PDL-off/NVFP4/shared
expert (0.984x); the worst B200 row is TP8/T16/FP16/K4/PDL-off/ordinary
output/shared expert (0.983x). These regressions must remain visible beside
the averages.

The body separately reports 32 TP4/T72 shapes: geometric means 1.025x on
B300 (1.001–1.051) and 1.027x on B200 (1.003–1.048). Those individual rows
are not in the 720-row table, so the aggregates cannot be recomputed from
it. Keep this crossover experiment separate from packed-load gains and
from unmeasured T73–87. The committed rank-max TRT/Cake/TRT harness is not
the external experiment, and neither measurement proves a serving gain.

Current source retains the inspected TP4 T72 selector constant, but every
one of the 40 whole device files differs from this PR head. Later load and
scheduling edits need their own review. For profiling, identify the selected
program and shape, preserve small-batch regressions, and verify instruction
placement/register pressure before attributing a latency change to prefetch.

## PR #8073 — prefetch first-token inputs after the dependency wait

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): issue the first token's route indices ahead of the control words](https://github.com/flashinfer-ai/flashinfer/pull/8073).
- State: merged 2026-10-31 at 04:14:28 UTC; head
  `50f1872d213a2b3dd9cdfd01e30f47a97afc80ba`, API base
  `e7ee9232eb2f6a1ec23312e29b3a9383c7feef36`, actual merge-base
  `36ec6101e76d34f364d7dee51a90e7660ac780a7`, merge
  `795601c070aa78d8eb280398cff3ef154ecb84b7`.
- Coverage: full 9,766-line diff fetched; all unique changes reviewed across
  twelve files, +4,002/-2,394. Complete-file comparisons explicitly reverse
  the inspected prefetch, shared-input selection and address-base edits in
  every variant, then prove equality to its base under bijective temporary
  renaming. Assembly strings, member fields, constants and operation order
  remain exact. Both dtype-specific unpacking blocks were read separately.
  All 40 base bodies match #7905; 28 head bodies are byte-identical. All body
  prose, structured checks of 720 timing rows, two issue comments, one empty
  approval, no inline comments and the single commit message are covered.
- Reviewed files: all twelve
  `cake_trtllm_moe_finalize_{bfloat16,float16}_ws{2,4,8}_o{110,111}_small_kernel.cu`
  files under `csrc/cake_moe_finalize_allreduce_fusion`. Binding, loader,
  public APIs, three test files, benchmark and timing helpers are unchanged.
- Motivation: shorten the small-token startup dependency chain by fetching
  route indices and optional shared-expert input before deriving communication
  addresses. The registry, dispatch and arithmetic contract are preserved.

### Input readiness and control-word lifetime

The [BF16 TP2 body](https://github.com/flashinfer-ai/flashinfer/blob/50f1872d213a2b3dd9cdfd01e30f47a97afc80ba/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws2_o110_small_kernel.cu)
still reads `control_0[3]`, the buffer-size word, **before** the existing
`griddepcontrol.wait`. Immediately after that wait, the new block begins:

```diff
+    int first_token = cluster_id;
+    int first_thread = cta_rank * 224 + tid;
+    int first_index[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
+    unsigned int first_shared[4] = {0};
+    if (first_token < tokens) {
```

With a valid first token, it reads up to eight route indices, each guarded
by `top_k > slot`; a shared expert additionally supplies one packed 16-byte
load per thread at `first_token * 7168 + first_thread * 8`. An empty volatile
assembly statement consumes all twelve words before the source reads
`control_0[2]` (current epoch) and `control_0[4]` (clear extent). Thus “before
the control words” refers to these later reads, not every control field.
The empty assembly is an operand dependency, not a hardware barrier or a
SASS-level scheduling measurement. Input reads remain after the PDL wait.

The publish loop uses the cached values only when `token == token_begin`;
subsequent tokens retain their ordinary loads. `token_begin == first_token`
and `cluster_thread == first_thread`. Shared data stays packed until the
selection and is then unpacked: BF16 retains the shift/mask conversion, while
the [FP16 body](https://github.com/flashinfer-ai/flashinfer/blob/50f1872d213a2b3dd9cdfd01e30f47a97afc80ba/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_float16_ws2_o110_small_kernel.cu)
retains two half-to-float conversions. Negative route indices still suppress
the corresponding expert read and contribution. Positive row bounds remain
the caller's contract. Cached indices/shared words are reset after publication;
later tokens do not consume those reset values.

Peer pointers now hold raw workspace bases. At publication, local copies
`publish_current_word` and `publish_bytes_word` pass through empty `+r`
assembly before calculating `(current_word % 3) * (bytes_word / 2)`. The
volatile peer stores add that base to their existing slot. After publication,
equivalent `late_*` copies form the reduction base and dirty-buffer base
`((current_word + 2) % 3) * stride` used for clearing. These are copies of
the **same already-read words**, not fresh samples of mutable control memory.
The bot walkthrough's wording about sampling at each operation point is
therefore too strong. Entry accounting, epoch advancement, clear extent,
route/rank rounding, RMS exchange, residual/norm operands and output epilogues
retain their prior operations. No changed control ownership or additional
synchronization is established by this scheduling rewrite.

### Actual test and timing evidence

There are no changed tests. The requested final-head
[run 41161568588](https://github.com/flashinfer-ai/flashinfer/actions/runs/41161568588)
contains ten jobs, including six successful AOT build/import jobs, and no
targeted numerical job. Setup checked merge `9d47ead59` of the exact head
into the API base. H100 PR tests were disabled for runner capacity; all five
A10G shards and the T4 scope were empty. The summary found no JUnit artifacts.
The earlier final-head run failed unauthorized-status reporting before setup.
The discussion contains the author's three-file test request and an empty
maintainer approval, without an additional numerical or sanitizer report.

The 720 unique table shapes match #7905. The same external CUDA 13.4 protocol
reports rank-min CUPTI times and median paired previous/new ratios from
interleaved runs. T1/2/4/8/16 supply the 480 changed-program shapes; the other
240 use unchanged bodies. Larger small-token boundary shapes are absent.

| Cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| All 720 | 1.012 / 0.989–1.053 | 1.012 / 0.988–1.071 |
| Changed 480 | 1.018 / 0.989–1.053 | 1.019 / 0.995–1.071 |
| Unchanged 240 | 1.000 / 0.989–1.015 | 1.000 / 0.988–1.009 |
| TP2 small, 160 | 1.020 / 1.000–1.047 | 1.023 / 1.003–1.071 |
| TP4 small, 160 | 1.022 / 1.003–1.053 | 1.021 / 1.007–1.039 |
| TP8 small, 160 | 1.011 / 0.989–1.031 | 1.012 / 0.995–1.033 |

All twelve B300 and two B200 changed-program ratios below one are TP8.
The B300 minimum is T8/FP16/K8/PDL-off/ordinary output/shared expert;
the B200 minimum is T2/BF16/K8/PDL-off/NVFP4/shared expert. Unchanged rows
also fluctuate: 112 B300 and 92 B200 ratios fall below one. Preserve these
controls and individual regressions alongside the geometric means.

The committed rank-max TRT/Cake/TRT harness remains different from this
external experiment. No end-to-end or model-accuracy improvement is proved.
Every one of the 40 whole historical device files differs at the current pin;
the source transformation proof covers this PR, not later generated bodies.
When profiling a small-token finalize, verify the selected revision, input
readiness and actual compiled load placement before applying this technique.

## PR #8139 — admit TP4/K4 without a shared expert to the T71 extension

- Link/title: [perf(cake_moe_finalize_allreduce_fusion): extend the small-token program to top_k 4 without a shared expert at world size 4](https://github.com/flashinfer-ai/flashinfer/pull/8139).
- State: merged 2026-10-31 at 21:43:40 UTC; head
  `c6f10f726c6b63b476324d8d001f98192dd9ef20`, API base
  `5086eb9db8be53c5b7bd331bfe810843034ce4fe`, actual merge-base
  `78420faf3092e2b9f617551e214e3eddd7dfec91`, merge
  `58ece34ea74ad39f9d641443fa14be28e5c7dabb`.
- Coverage: full 6,809-line diff fetched; all unique changes reviewed across
  44 files, +1,386/-1,344. All 40 base bodies match #8073. Full-file comparisons
  show twenty bodies only delete the unused `NUM_MAIN_STAGES` definition;
  twenty also bijectively rename local temporaries, including loop indices
  and unpacked buffers. All other bytes, constants, assembly strings and
  operation order are preserved. All body prose, 16 changed-shape rows and
  structured checks of 720 control rows, two issue comments, two reviews,
  no inline comments and the single commit message are covered.
- Reviewed files: the host binding, all forty device files,
  `flashinfer/jit/cake_moe_finalize_comm.py`, the GPU test's program mirror and
  `tests/cpu/comm/test_cake_moe_finalize_dispatch.py`.
- Motivation: use the existing small-token implementation for one more TP4
  routing class when a whole token's four-CTA cluster can reside on the GPU.
  The 40-program registry and kernel arithmetic remain unchanged; the
  regeneration is not forty new kernel optimizations. The commit's claim of
  unchanged compiled code was not independently checked with nvcc or SASS.

### Dispatch and the capacity condition

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/c6f10f726c6b63b476324d8d001f98192dd9ef20/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
changes the class term to:

```diff
+  const bool small_token_extended =
+      world_size == 4 || (world_size == 2 && (top_k != 4 || v_has_shared_expert != 0));
```

With the default `small_token_max=64`, the new class is TP4/K4/no-shared at
T65–71. A smaller configured limit expands the affected range to
`small_token_max + 1` through 71. In both cases the extension additionally
requires `tokens <= cluster_capacity(small_token)`: the minimum of the
per-SM residency-derived capacity and `cudaOccupancyMaxActiveClusters`.
Insufficient capacity retains the persistent fallback. The ordinary
`tokens <= small_token_max` branch is separate from this extension check.

Two-shot selection remains first: TP4 switches at T72 and TP8 at T56,
including when a custom small-token limit would otherwise admit them.
TP2's K4/no-shared exclusion and TP8's lack of an extension do not change.
The release-note suggestion of newly extended TP2 support is therefore
misleading. Small and persistent programs share the one-shot protocol,
so their device-local capacity decision may differ between ranks without
changing the exchange format. This does not relax cross-rank agreement
between one-shot and two-shot protocols.

The current pin retains the exact inspected class term, small-branch
predicate, T71 cap and earlier T72/T56 tests. The early two-shot successor
is reviewed below; other later variants and the changed current device
bodies still require separate review.

### Tests and measured coverage

The new [CPU test](https://github.com/flashinfer-ai/flashinfer/blob/c6f10f726c6b63b476324d8d001f98192dd9ef20/tests/cpu/comm/test_cake_moe_finalize_dispatch.py)
extracts the binding's class and small-branch expressions, translates their
logical operators and checks limits 16/64, all three world sizes, K4/K8,
shared/no-shared, five capacities and T1–79. Executing those two unchanged
test functions from their AST, with only the source-directory provider
substituted, passes 9,480 branch combinations locally. This checks predicate
logic, not actual occupancy or the earlier two-shot branch. Comparing old
and new expressions changes only TP4/K4/no-shared within the admitted range.

The GPU test updates `_expected_program` but adds no token values or new
numerical cases. Its mirror omits the real occupancy query; the existing
kernel-name assertion is conditional on PDL/weight-bias and capacity-related
device/shape checks. The public APIs, unified MoE test, benchmark and timer
are byte-identical to #8073. No new graph-replay, model-accuracy or compiled
identity result is supplied by this diff.

Final-head [run 41230991164](https://github.com/flashinfer-ai/flashinfer/actions/runs/41230991164)
enumerates ten jobs and six successful AOT builds, with no targeted CPU or
GPU test job. Setup checked merge `1c44dcf0f` of the exact head into the API
base; H100 was disabled, all A10G/T4 targeted scopes were empty, and the
summary found no JUnit artifacts. Two earlier final-head runs failed
unauthorized-status reporting before setup. The empty maintainer approval
and CODEOWNERS-gap bot attestation add no numerical validation result.

The external CUDA 13.4 rank-min CUPTI experiment measures sixteen changed
shapes: TP4/K4/no-shared, T65 or T71, both dtypes, PDL settings and output
profiles. It reports sufficient grid capacity at both measured token counts.
Median paired ratios are distinct from ratios of the displayed median times.

| Cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| Changed 16 | 1.018 / 1.004–1.031 | 1.021 / 1.002–1.033 |
| Unchanged standard 720 | 1.000 / 0.977–1.023 | 1.001 / 0.978–1.027 |

No changed row is below one. The 720 control rows have the same inventory as
#8073 and run unchanged programs, with 338 B300 and 271 B200 ratios below one.
T66–70, the range exposed by a smaller configured limit, and capacity-limited
devices have no separate timing rows. Keep these limits and the unchanged
rank-max committed harness distinct from the sixteen-shape dispatch result.
For a profile at this boundary, verify class, token count, resident capacity
and selected program before attributing the gain to a new kernel body.

## PR #8321 — issue first-token routing indices before the current/clear words

- Link: [FlashInfer #8321](https://github.com/flashinfer-ai/flashinfer/pull/8321).
- State: merged 2026-11-02 at 12:47:48 UTC; head
  `54b33fede8cfd63baba6d513d884e7266abcd3c1`, API base
  `3d4c266d9499be1005969e089aad851466717e11`, actual merge-base
  `3b6e31e5565a2e539ade639507a5d363588fd0b7`, merge
  `473993f4eb44bc0558c511d7a9444afae2a2a536`.
- Coverage: full 15,107-line diff fetched; unique changes reviewed across
  24 files, +11,686/-823. All 40 retained base bodies match #8139; 28 head
  bodies are byte-identical and twelve TP8 bodies only bijectively rename
  local temporaries. The eight new early bodies were compared with their
  same-head regular two-shot counterparts: explicitly undoing each inspected
  read/scheduling edit reproduces every complete body modulo temporary
  renaming, preserving dtype-specific arithmetic, constants and assembly.
  All body prose, structured checks of 720 timing rows and intervals, two
  issue comments, two reviews, no inline comments and the single commit
  message are covered.
- Reviewed files: eight `*_ws{4,8}_o{110,111}_tse_kernel.cu` additions,
  twelve renamed TP8 bodies, the binding, JIT loader and CPU/GPU dispatch
  tests. The registry grows from 40 to 48 programs per architecture.
- Motivation: shorten the first-token input dependency chain for small
  two-shot batches without changing their rank exchange or epilogues.

### Read order and buffer lifetime

In the [BF16 TP4 early body](https://github.com/flashinfer-ai/flashinfer/blob/54b33fede8cfd63baba6d513d884e7266abcd3c1/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws4_o110_tse_kernel.cu),
the buffer-byte word `control_0[3]` is still read before
`griddepcontrol.wait`. After that wait, the kernel reads the first token's
first four inverse-routing indices, guards the token/slot accesses, and
feeds the four words into an empty volatile assembly statement. Only then
does it load the current-epoch and clear-range words `control_0[2]`/`[4]`.
Thus the new early input reads do not cross the PDL readiness boundary.
The cached token is `cluster_id`; later published tokens use ordinary
index loads. K8's second group of four indices and shared-expert data are
not newly hoisted. All eight early bodies retain this source order.

The first publication substitutes cached indices into the original route
body. Before scattering, copies of the already loaded current/byte words
pass through empty `asm` read/write operands, and `scatter_base` is computed
as `(current % 3) * (bytes / 2)`. The kernel then initializes the ordinary
read base and dirty-buffer clearing schedule in an unconditional
`if (step == 0)` after the publication block, before clearing or reduction.
The dirty slot remains `(current + 2) % 3`; the original clear-budget formula
and `local_tokens + 2` loop/drain remain intact. A cluster with zero local
tokens still executes the initialization and drain. Each of the three
control words is read exactly once; later address calculations are not
fresh epoch sampling. Cached first indices are reset at each iteration end.

The [TP8 body](https://github.com/flashinfer-ai/flashinfer/blob/54b33fede8cfd63baba6d513d884e7266abcd3c1/csrc/cake_moe_finalize_allreduce_fusion/cake_trtllm_moe_finalize_bfloat16_ws8_o110_tse_kernel.cu)
uses the same transformation with its existing chunk/owner geometry. The
FP16 and quantized-output bodies preserve their respective unchanged
conversion and epilogue operations. Full-file proofs establish the source
transformation, not emitted load placement, register pressure or hardware
synchronization: empty assembly supplies compiler operand dependencies.
The two-shot protocol and BF16 escape encoding are inherited, including
the [finite-value versus NaN marker limitation](flashinfer-moe-finalize-two-shot-review.md).
Do not turn the source comment about identical bits into arbitrary-input
byte-parity or a new numerical-validation claim.

### Selection and boundary tests

The [binding](https://github.com/flashinfer-ai/flashinfer/blob/54b33fede8cfd63baba6d513d884e7266abcd3c1/csrc/cake_moe_finalize_allreduce_fusion/cake_moe_finalize_allreduce_fusion_binding.cu)
adds a TP4/TP8 early table with null TP2 entries and selects:

```diff
+  const void* two_shot_now =
+      (two_shot_early != nullptr &&
+       tokens <= (world_size == 8 ? kTwoShotEarlyMaxTokensWorldSize8
+                                  : kTwoShotEarlyMaxTokensWorldSize4))
+          ? two_shot_early
+          : two_shot;
```

Both early maxima are 128. This choice is used in the first two-shot branch
and the TP8 fallback after small-token selection, with deferred selection
still taking precedence. Default ranges are TP4 T72–128 and TP8 T56–128.
At `small_token_max=16`, TP8 T17–55 also reaches the early fallback. TP4
below T72 retains the small/capacity/persistent decision. Early and deferred
ranges cannot overlap; static assertions enforce this. The current pin
retains the exact early table, constants and `two_shot_now` expression,
although later shared-expert deferred selection has changed. All 48
historical whole device files differ at the current pin; this proof does
not certify those later bodies.

The [CPU test](https://github.com/flashinfer-ai/flashinfer/blob/54b33fede8cfd63baba6d513d884e7266abcd3c1/tests/cpu/comm/test_cake_moe_finalize_dispatch.py)
checks table contents/coverage and translates the binding's expressions.
Executing the original function ASTs with decorators removed and a local
source/module-record provider passes nine cases: six tables, their union,
and both world-size boundary cases. The latter assert T128 early, T129
regular, T16 small and T17 TP8-early/TP4-persistent at capacity zero. This
is eight boundary assertions, not a full shape/numerical or occupancy test;
the evaluator does not execute deferred selection.

The [GPU test](https://github.com/flashinfer-ai/flashinfer/blob/54b33fede8cfd63baba6d513d884e7266abcd3c1/tests/comm/test_cake_moe_finalize_allreduce.py)
adds T128/T129 and the early program mirror. T128 already existed, so the
actual new token value is only T129: TP4 grows from eight to nine token
counts and TP8 from ten to eleven. The existing dtype/shape matrix and
conditional kernel-name assertions remain. Public APIs, the unified MoE
test, committed benchmark and timer are byte-identical to #8139.

Final-head [run 41436257291](https://github.com/flashinfer-ai/flashinfer/actions/runs/41436257291)
contains ten jobs, including six successful AOT builds and no targeted
CPU/GPU test job. Setup checked merge `25140ae35` of this head into the API
base. H100 was disabled; every A10G/T4 targeted scope was empty; the summary
downloaded zero JUnit artifacts and skipped its skip-all check. An earlier
summary failed while waiting for a cancelled run, with its own setup skipped.
Maintainer approval and the CODEOWNERS-gap attestation supply no numerical
test result. No GPU test was rerun in this review.

### Measured cohort and controls

The external CUDA 13.4 experiment interleaves previous/new Cake runs and
uses rank-min CUPTI times, median paired ratios and 95% bootstrap intervals.
Its standard table has the same 720 shape identities as the earlier
controls; only 64 TP4/TP8 T128 rows select new programs.

| Printed-row cohort | B300 geometric mean / range | B200 geometric mean / range |
|---|---|---|
| Changed 64 | 1.039 / 1.010–1.079 | 1.038 / 1.006–1.080 |
| Unchanged 656 | about 0.9995 / 0.971–1.031 | about 0.9995 / 0.985–1.015 |
| All 720 | 1.003 / 0.971–1.079 | 1.003 / 0.985–1.080 |

All 64 changed rows have printed interval lower bounds above one on both
GPUs. The controls include 302 B300 and 306 B200 ratios below one; 52/48
intervals respectively lie entirely below one, while 29/28 lie entirely
above. Retain these unchanged-program fluctuations when interpreting small
gains; the intervals are per-row, not a simultaneous all-shape guarantee.
The body reports control means of 0.999; the rounded printed rows recompute
to 0.999505/0.999453. Underlying unrounded ratios are not published here.

The body separately reports 320 changed measurements: TP4 T72/88/95/96/104/128
and TP8 T64/65/72/128, each crossed with 32 dtype/K/PDL/output/shared choices.
It reports geometric means 1.036/1.035 and no ratio below one. Only 64 of
those rows are published individually; the other 256 cannot be recomputed
from the table. This is a sampled cohort, not every changed token count:
TP8 T56–63 and custom-small-limit fallback ranges have no separate rows.
The committed rank-max TRT/Cake/TRT harness is unchanged and uses a different
protocol. Neither the table nor source inspection establishes a service
throughput or model-accuracy improvement.
