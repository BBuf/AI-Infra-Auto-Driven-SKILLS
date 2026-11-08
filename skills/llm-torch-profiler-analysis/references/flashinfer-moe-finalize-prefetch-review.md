# FlashInfer indexed finalize: packed operands and first-token prefetch

Reviewed on 2026-11-09 (Asia/Shanghai). This is a historical source review
of #7905/#8073, following the [two-shot/deferred lineage](flashinfer-moe-finalize-two-shot-review.md).
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
