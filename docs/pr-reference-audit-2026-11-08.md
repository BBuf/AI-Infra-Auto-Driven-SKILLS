# PR reference audit — November 8, 2026

This is the reference-validation portion of the broader source refresh. It
does not certify that every cited or open PR has been manually reviewed.
The [status snapshot](../model-pr-optimization-history/pr-reference-status.tsv)
records each lookup's UTC time, object type, state, exact head/base/merge
revision and GitHub change counts. Missing fields are unknown, not zero.
The source census began at skill-repository commit `a53ddfc42ee9`.

## Explicit and qualified references

The initial scan found 5,602 unique `owner/repo/pull/N` or `owner/repo#N`
references. Seven fixture-only/example references were excluded. All 5,595
remaining numbers were queried through GitHub, with failed PR lookups checked
again using `issueOrPullRequest`:

| Result | Count | Meaning |
|---|---:|---|
| Merged PR | 5,179 | GitHub reports a merge; this is not a manual diff-review count. |
| Closed, unmerged PR | 247 | Do not describe its patch as shipped. |
| Open PR | 154 | Its recorded head can change; recheck before use. |
| Issue | 14 | Some were incorrectly promoted to PR links by the history generator; others are valid shorthand issue references inside source excerpts. |
| Unavailable | 1 | vLLM #50704: PR/issue API lookup failed, but the mainline commit exists. |

## Bare references and open-PR path coverage

The supplemental sweep resolved 17,536 bare-reference occurrences into 6,585
repository/number pairs using their framework directory, section, same-line
source paths and explicit dependency names. Conflicting cross-framework
contexts were reviewed separately; inclusive ranges were expanded. HTML
numeric entities and a pronunciation example were excluded. Repeated numbers
in different repositories remain distinct.

This adds 1,339 objects absent from the initial census: 1,316 PRs (1,286 merged,
18 closed without merge, 12 open) and 23 issues. The historical union contains
6,934 objects. Thirty-four additional PRs cited by the new review cards bring the
TSV to **6,968 objects**; all 75 PRs manually reviewed in this refresh have
head checks matching their reviewed diffs, at the recorded per-PR timestamps. The initial table
above describes the initial census, not this expanded union. Issue references
inside quoted titles, code comments and RFC
discussions are valid context; they must not become implementation PRs.
FlashInfer #5678 in the training-kernel catalog is now explicitly identified
as a feature issue rather than mainline implementation evidence.

All open-list pages and changed-path pages were collected for the same
November 8 snapshot: SGLang 5,582, vLLM 7,224, FlashInfer 913, TensorRT-LLM
1,277 and TokenSpeed 105, totaling **15,101 open PRs**. Each listed PR has a
path inventory. For five diffs beyond GitHub's file-list limit, paths came
from the recorded merge-base/head Git diff: SGLang #41663/#41672, vLLM #54955,
FlashInfer #3034 and TensorRT-LLM #13322. Rename-disabled inventories retain
both old and new paths; their path counts need not equal GitHub's changed-file
counts. GitHub and local rename detection also differ for the two SGLang
diffs. These inventories establish discovery coverage, **not code-review
coverage**. Relevance screening and full open-PR source review remain ongoing.

## Corrections supported by this audit

- Removed 154 issue-as-PR links from implementation coverage tables in 20
  bilingual history files. Source excerpts containing valid issue references
  are unchanged. The generator now takes the enclosing merge/squash PR number,
  rather than every `#N` in a commit subject, and drops rejected numbers from
  rendered PR-coverage tables. First-parent merge diffs supply actual touched
  paths; an empty path list no longer attributes every file in a query chunk.
- SGLang [#21194](https://github.com/sgl-project/sglang/pull/21194) merged on
  October 29. Its DeepSeek-V3.2 cards now describe the final complete diff and
  PP placeholder guards, with the AMD/GLM cross-model implications and the
  limits of the reported validation.
- vLLM [#50704](https://github.com/vllm-project/vllm/pull/50704) returns 404 from
  the PR API and null from `issueOrPullRequest`. The local full history contains
  commit [`0601850791155003afbe5a0d5d086350cada8deb`](https://github.com/vllm-project/vllm/commit/0601850791155003afbe5a0d5d086350cada8deb),
  dated August 2 and titled with that number. Keep this commit evidence separate
  from unavailable live PR metadata; a failed lookup must not erase it.
- Open/closed/error cache entries are refreshed on the next generator run.
  Closed PRs can reopen. `--refetch-existing` now also bypasses the persisted
  cache instead of merely rendering it again. Removed unused title-derived
  motivation helpers and an unused coverage-table renderer.
- The open watcher fetches every page before term matching and rejects a
  report if any repository failed. A keyword match still does not replace a
  changed-file relevance check or a full diff review.

The incorrectly typed coverage entries were TensorRT-LLM issues
[15179](https://github.com/NVIDIA/TensorRT-LLM/issues/15179) and
[15613](https://github.com/NVIDIA/TensorRT-LLM/issues/15613), SGLang issue
[29630](https://github.com/sgl-project/sglang/issues/29630), and vLLM issues
[32278](https://github.com/vllm-project/vllm/issues/32278),
[32736](https://github.com/vllm-project/vllm/issues/32736),
[34028](https://github.com/vllm-project/vllm/issues/34028),
[38633](https://github.com/vllm-project/vllm/issues/38633),
[41962](https://github.com/vllm-project/vllm/issues/41962),
[45834](https://github.com/vllm-project/vllm/issues/45834),
[50337](https://github.com/vllm-project/vllm/issues/50337) and
[59271](https://github.com/vllm-project/vllm/issues/59271).

## Profiler review

[Seven complete profiler diff reviews](../skills/llm-torch-profiler-analysis/references/profiler-open-pr-review.md)
distinguish current main from incompatible pending fixes for stage limits,
filename handling, PREBUILT mapping, argparse and MEM teardown. They support
the client's label/step validation and failed-probe cleanup. The inspected
native `sglang.profiler --help` was also run under Python 3.14 and reproduced
the `BooleanOptionalAction(type=...)` constructor failure. This is a CLI
reproduction, not a GPU profiling test.

## V4.1 kernel review

[Six complete kernel diff reviews](../skills/llm-torch-profiler-analysis/references/dsv41-open-pr-review.md)
cover Blackwell compensated mHC, independent FP8 scale validation and Hopper
indexer scoring, scratch budgets, grouped K reuse and selection/writeback.
They record all changed files, exact revisions, implementation excerpts,
current callers, discussion and test limitations. The four Hopper ports have
experimental-branch measurements but no reported GPU validation of the ports;
their flags must not be advertised as mainline capabilities.

[Three complete merged TRT-LLM attention reviews](../skills/llm-torch-profiler-analysis/references/dsv41-trtllm-source-review.md)
cover uniform-FP8 KV storage, decoder bounded replay with breakable prefill,
and the public persistent-counter API. Current source still retains Q's
separate BF16-to-FP8 conversion. The catalog no longer labels that integration
as open or attributes the pending #36652 rewrite to main. A duplicated
619-line dispatch/control section in the overlap catalog now links to its
canonical counterpart, preserving the section anchor.

The same review also covers the complete 3,500-line pending #36652 rewrite
and all of its discussion. It separates dense per-token calls, shared sparse
tables, direct top-k writeback, draft metadata capture, FP8 producer epilogues
and PDL dependencies from the current mainline. The card preserves revision,
hardware and acceptance-mode limits on performance reports, including a
profile-only improvement that has not been measured end to end.

[Three earlier Hopper indexer reviews](../skills/llm-torch-profiler-analysis/references/dsv41-hopper-indexer-review.md)
cover the closed-unmerged #40160 grouped-WGMMA prototype and pending #42365
dense prefill fusion / #40860 compact candidate scoring. The catalog now records
their current states and separates the current 1 GiB prefill budget and fused
32-head epilogue from older benchmark baselines. Current paged decode initializes
only the live score prefix; compact scoring instead scans all compact columns
and restores logical positions before slot mapping. All three full diffs and
discussions were read and their heads/bases rechecked; no GPU results were rerun.

The [complete merged mHC state review](../skills/llm-torch-profiler-analysis/references/dsv41-mhc-source-review.md)
corrects #42245's old open status and retired source paths. Its final 19-file
diff and all discussion establish explicit reduction ownership, state lifetime,
the 128-row collective quant epilogue and removal of combine-only variants.
Current boundary callers are reviewed separately because TP2 support, optional
push planes, cached statistics and new post/norm fusions evolved after the merge.
Historical intermediate-head test results are not promoted to final-head GPU
validation, and earlier failed workflow partitions remain distinct from files
that actually passed inside them.

The [TokenSpeed mHC review](../skills/llm-torch-profiler-analysis/references/tokenspeed-mhc-source-review.md)
covers #1767's full 27-file diff and all discussion, with a fresh head/base
check. It replaces the stale open-proposal row with the merged implementation:
communication admission before deferral, stage/Engram handoff completion,
MegaMHC replay exclusion and current FP8 output layout. CI plans and earlier
performance reports remain distinct from executed final-head validation.

The [TokenSpeed Hopper selection review](../skills/llm-torch-profiler-analysis/references/tokenspeed-indexer-source-review.md)
reads #1857's complete three-file diff, both comments and current admission,
scoring, cleanup and selection callers. Its merged implementation bounds traffic
with device lengths while retaining capacity-shaped buffers; candidate consumers
that can read a partial block keep fully initialized scores. Performance and
successful request counts are not promoted to answer-accuracy evidence.
It also covers #1877's full three-file DCP tile-skipping diff and #1781's full
five-file KPool row-split diff, including all discussion and current callers.
The former is merged but retains full-width logits; the latter was closed
without merge. Benchmark regression comments and unverified accuracy-waiver
requests remain qualified rather than being reported as a clean test result.

The [TokenSpeed selected-slot attention review](../skills/llm-torch-profiler-analysis/references/tokenspeed-dsa-source-review.md)
covers #1780's complete 967-line diff, body, all comments/reviews and current
provider/facade/runtime paths. It replaces the stale open-proposal entry with
merged tensor-core tiles, residual scoring and split-LSE semantics. Current
KVP record output and numerical-policy admission are distinguished from the
original change; early H20 measurements are not final-rebase validation.

The [TokenSpeed routing review](../skills/llm-torch-profiler-analysis/references/tokenspeed-routing-source-review.md)
covers #1770's complete 2,374-line, 31-file diff and all 11 issue comments,
36 reviews and 39 inline comments. It checks current V4.1 callers, plan traits,
router eligibility, top-k storage, quantization and finalization. The stale
open-proposal row and moved paths are replaced with merged-source evidence;
main's centralized missing-weight recovery is distinguished from the original
local fallback. Historical B300 cycles and AMD CI comments retain their actual
measurement and validation scope.

The [TokenSpeed operator review](../skills/llm-torch-profiler-analysis/references/tokenspeed-operator-source-review.md)
covers the complete #1606/#1607/#1889 diffs and all discussion, with fresh
head/base checks. Current provider registration, runtime callers and FlashInfer
contracts distinguish preserved BF16 routing precision, the chain sampler's
selective reduction and owner-local MHA counter reuse. The old MoE source path
has moved; the counter owner can now grow during tree binding, and generic
attention wrappers do not inherit the runtime leaf's reuse. Historical
FlashInfer 0.6.18 attention measurements are not validation of the current
0.7.3 dependency.
The same review covers #1931's full 436-line diff and all review replies,
including removed redundant tests, current `gemm.mv` registration and Beam's
new FP32-weight caller. Historical “no model caller” and old file/API names
are no longer used as current guidance; final-rebase GPU coverage remains
qualified because the reports do not consistently identify a tested SHA.
It also covers #1741's full seven-file persistent gfx1250 MoE proposal and
all discussion. Closed without merge, it is absent from the current regular
apply callers. The card preserves the unresolved public-launcher activation
parameter finding and separates architecture-skipped GPU tests from CI plans.

The [vLLM routing review](../skills/llm-torch-profiler-analysis/references/vllm-routing-source-review.md)
covers #59737's complete one-file diff, body, 23 issue comments, nine reviews
and ten inline comments. Current gating, router dispatch and regression code
show the later host SM90 gate, row-stride fallback and shape-specific producer
cutovers. The catalog separates this merged change from #45120, now closed
without merge, and still-open #43190. The latter two received fresh metadata
checks only and are not added to the manual-review count.

The [vLLM compiler review](../skills/llm-torch-profiler-analysis/references/vllm-compiler-source-review.md)
covers the complete four-file #57349 and five-file #57900 diffs and all
discussion. #57349 is closed without merge and folded into still-open #57900.
Current config ordering, splitting tests, all three cache-fusion files and
backend layout methods establish why flags can fail to produce a match and
why matched Q quantization remains a separate operator. The cards preserve
128/256-token regressions, pre-rebase measurements and nonidentical outputs.
The same review covers the complete #49142 MLA default and #54851 DSA capture
binding proposals and all discussion. The first lacks the successor's hoisted
layer-name guard; its small H100 serving report is qualified. The second's
metadata-free capture mechanism is distinguished from current MRV2 metadata
construction, separate indexer slots and HiSparse/PCP ownership.

The [CUDA RoPE/cache review](../skills/llm-torch-profiler-analysis/references/vllm-rope-cache-kernel-review.md)
covers all six files of #57271 and five files of #58798, including tests,
benchmark code and all discussion. It distinguishes full rotary from MiMo
partial rotary, short-slot handling, intermediate rounding and fresh-Q/cache
ownership. Current MiMo's padded checkpoint chunks require adapting the old
proposal's plain QKV split. B200/SM80 reports retain their actual reference,
A/A controls, overlay limitations and corrected discussion claims. The same
review covers all 21 files and all discussion of #52363, including its native
kernel, descriptor-based manual dispatch, dynamic threshold regression tests,
engine checks and paired benchmark. It retains the above-threshold copy
regressions, near-equal E2E result and the pre-refactor dates of earlier
hardware validation and approval.

A separate metadata-only reconciliation refreshed 78 stale PR status/head
mentions across 45 fusion-catalog rows after another API check. Closed PRs
are no longer described as active proposals, and merged PRs record the actual
merge date. Original heads/dates remain attached to historical implementation
notes and measurements; these rows explicitly await final-source review.
The snapshot also refreshes base/merge revisions and change counts. This adds
**zero** to the manual-review count. An unrelated personal-memory
instruction was removed from the public catalog.

The [SGLang FP8 store review](../skills/llm-torch-profiler-analysis/references/sglang-fp8-kv-store-source-review.md)
covers the complete six-file #40785, five-file #42982 and seven-file #31652
diffs and all discussion, including numerical tests and the benchmark harness.
It documents the prefix writer's asymmetric-row rejection, preserves scale-kind
and source-dtype rounding, and records the later AMD saturation fix and its
independent overflow test. Current unified MHA views share prefix commits but
have a separate ordinary writer not covered by #31652. Benchmark descriptions
now include layout-copy cost, graph-level versus per-call cache flushing,
unchanged caller copies and the actual head/configuration of serving reports.
The final saturation-fix MI300 JIT jobs passed within a failed overall
workflow; #31652's final-head PR gate failed and its GPU jobs were skipped.
These three source reviews replace one previously historical catalog row.

The [SGLang Q-quantization review](../skills/llm-torch-profiler-analysis/references/sglang-query-quantization-source-review.md)
covers closed #35142's complete 12-file diff and discussion, separating
native compiler visibility from eager RoPE/FP8 fusion. It records the missing
intermediate rounding boundary and current Qwen3 producers that bypass the
old rotary entry point. The [vLLM QK-norm/cache review](../skills/llm-torch-profiler-analysis/references/vllm-qknorm-rope-cache-source-review.md)
covers open #57682's complete nine-file diff and all discussion, including
its stacked shuffle fix, actual O2/O3 defaults, homogeneous-layer admission
and compile-range gate. Source test counts, old-wheel dummy-weight benchmarks
and missing final-head GPU validation remain separate. Current Step5 and Muse
Spark model-specific paths are not conflated with the generic compiler pass.
These two reviews bring the completed manual-review count to 46.

The [FlashInfer QK-norm/append review](../skills/llm-torch-profiler-analysis/references/flashinfer-qknorm-rope-append-source-review.md)
covers closed #5405's complete 15-file diff and all discussion, including
its imported CUDA implementation, numerical tests and BF16-only harness.
It distinguishes the FP8 validator/flag-fill launches from the fused kernel,
documents intermediate rounding and unwritten scale padding, and qualifies
the historical CUPTI timing span. Frozen-main CAKE API/backend, stage policies
and two selected generated kernels/bindings establish current availability
and PDL readiness constraints. The completed manual-review count was 47.

The same card now includes merged #5952's complete 19-file diff and all
discussion. All six generated source pairs were compared, with their common
kernel/binding read in full and every remaining difference inspected. It
separates the original head/architecture-only dispatch from current policies,
records cache-tail ownership and empty-input limitations, and checks all 21
test cases plus the benchmark and timing helper. The 17-shape committed
harness, median-only output and fixed graph-before-eager ordering limit what
the PR body's broader performance claims establish. #5956 and #6008 were
pending at that checkpoint. The completed manual count was 48.

The #5956 review completes its 12-file FP8 diff and all 24 comments, 15
reviews and 22 inline comments. It separates obsolete stacked-version
replies from the standalone final source, checks every generated variant,
and traces eight-lane reductions, fused validation and tail-write ownership.
Final-head H100 CI logs confirm 51 tests passed on CUDA 12.9. The card
corrects the tail-byte validation claim and distinguishes external paired
performance reports from the committed fixed-order harness and its
numerically different comparator. The completed manual-review count was 49.

The #6008 review covers its complete 52-file diff, seven comments and one
review, with all 20 generated pairs compared and all 40 source hashes checked.
It traces the exact shape/architecture policy, row/V ownership and clearing
grid. Final-head H100 CI confirms 42 passes, but half are CPU selector tests;
the numerical fixtures do not cover every generated program. A historical
page-1024 missing-build error was reproduced from the pure policy, with the
current rule change recorded separately. The external 28-shape paired report
is distinguished from the unchanged 17-shape committed harness. The completed
manual-review count was 50.

The [SGLang MoE collective review](../skills/llm-torch-profiler-analysis/references/sglang-moe-collective-source-review.md)
covers closed #39214's full seven-file diff and all discussion. It traces
layout materialization, actual BF16 producer admission, workspace and signature
gates, consume-once behavior and the shared deterministic-mode override.
Current main instead uses a native-layout CuTe-DSL boundary provider and keeps
different speculative shared-load ordering. The card retains the author's
passing tolerant payload checks and failed per-question score-equality result;
final-head CI stopped at draft gating, with tests skipped. No performance
gain is inferred from correctness-campaign durations or dependency API exports.
The completed manual-review count was 51.

The HY3 #4890 card completes its eleven-file diff, two issue comments and
one review. It reconciles the bot's overbroad base-to-head file inventory with
the actual merge-base diff, records uniform-request admission and unwritten
empty-request status, and distinguishes meaningful stream/parity tests from
missing replay coverage. It qualifies the static comparator's rounding and
side effects and the maintainer's later CAKE performance observation. Final-head
CI awaited authorization; the PR remains open despite the suggestion to close
it in favor of the merged API. The completed manual-review count was 52.

The [FlashInfer MoE collective review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-collective-source-review.md)
completes #4730's ten-file diff and all discussion, comparing all 28 generated
entrypoints and the current union selector/launcher. It documents expert-major
layout, per-addition rounding, Lamport entry-counter ownership and optional
output scratch retention. Candidate/final runtime bytes match; the source
inventory test was repaired separately. Fresh final-head CI is green but each
H100/T4/A10G shard passes twelve CPU cases and skips six distributed cases.
That narrower run does not resolve earlier all-communication failures or
independently validate the external Blackwell timing reports. The completed
manual-review count was 53.

The same card completes #5894's two-file scratch-retention fix and all
discussion. Its helper remains unchanged at the current pin. The review
derives the geometric retained-memory bound and identifies the graph test's
incidental strong references, while verifying 70 passing H100 CI cases from
the final-head merge. No collective performance gain is inferred. The completed
manual-review count was 54.

The same card completes #5856: all nine changed files, the twelve deleted
kernel bodies compared byte-for-byte with the reviewed #4730 bodies, all
discussion, and the unchanged union source tree. It qualifies the external
paired timing protocol and verifies 69 CPU passes with six distributed skips
from H100 final-head CI. The original graph helper ends before cache growth;
#5894 repairs the lifetime gap. The completed manual-review count was 55.

The [indexed MoE finalize review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-source-review.md)
completes #4875: all 81 files, twelve unique device bodies plus identical
architecture copies, all binding differences, manifest validation, tests,
benchmark and discussion. It corrects the merge date and stale CI failure
note, verifies seven CPU passes/43 skips from the final-head H100 log, and
separates the external timing protocols/cohorts from the committed harness.
Current loader/binding/API comparisons retain the new byte-buffer contract
and shape/occupancy gates without claiming all successor kernels reviewed.
The completed manual-review count was 56.

The same review completes #5852: all 67 changed files and discussion, with
61 deleted files verified identical to already reviewed #4875 content and
all twelve retained device sources unchanged. It documents the new binding,
byte-sized quant outputs, missing internal payload guard and the public API
guard that remains. A CPU-only reproduction confirms the trimmed harness's
CUPTI-to-event fallback, and final-head H100 logs establish six CPU passes
with 43 GPU skips. Warmed-run regressions and unfinished TP4 timing remain
visible. The completed manual-review count was 57.

The same card completes #5870: fourteen changed files, the full 675-line
final template, byte-identical deleted predecessors and all discussion.
It distinguishes compiler-specific poll/SASS equivalence from source
deduplication, keeps the rank-max regressions and contradictory final table
minimum, and records the reported TP4 hang. Current code no longer registers
that template file. H100 final-head CI confirms six CPU passes/43 GPU skips.
The completed manual-review count was 58.

The same card completes #6057/#6128/#6166: all changed device/host code,
discussion and commit messages, with predecessor and unchanged supporting
files compared byte-for-byte. It traces pair/quad gathering, the DSM slot
ring, occupancy sizing, local polling and TP4-only residual prefetch. The
review preserves rejected TP2/TP8 variants, different timing cohorts and
large-token timeouts. Each final-head H100 log confirms six CPU passes and
43 GPU skips; internal pipeline start announcements are not completed test
evidence. Current cluster-capacity and shape dispatch are distinguished from
the retired template. The manual-review count at that increment was 61: 27 SGLang, 11 TokenSpeed,
nine vLLM and fourteen FlashInfer PRs. The three newly reviewed heads/bases
match their latest API checks; the previous 58 checks retain their recorded
timestamps. The metadata ledger contains 6,954 objects.

The [five-family communication consolidation review](../skills/llm-torch-profiler-analysis/references/flashinfer-comm-source-consolidation-review.md)
completes #7054: all unique edits, full-file clone/preprocessor comparisons,
body, discussion and five commits. It records the actual scope of shared
compile definitions, preserves launch-bound variants and separates unchanged
source from a fresh algorithm audit. Local CPU checks reproduce the Ulysses
header digest gap with a listed-source negative control; Ninja dependency
tracking is a separate mechanism. Final-head H100 CI has 215 passes/196 skips,
not Blackwell distributed validation. No benchmark was reported for this
cleanup. Current norm-combine dispatch and the 96-library Ulysses manifest
are distinguished from the historical three-variant/48-library portfolios.
The manual-review total is now 62: 27 SGLang, 11 TokenSpeed, nine vLLM and
15 FlashInfer PRs. The new head/base check matches; earlier entries retain
their individual check timestamps. The ledger now contains 6,955 objects.

The [indexed finalize scheduling review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-schedule-review.md)
completes #7392's 38 changed files using canonical device bodies, every
schedule/world-size difference and full-file dtype/quantization comparisons.
It includes all discussion, five commits and structured checks of the 720
timing rows. Preserve the literal occupancy-dependent large-path gate, the
complete RMS synchronization protocol and per-route rounding. The committed
tests miss large-token programs on ordinary B200/B300 devices; final-head
H100 CI records nine CPU passes and six GPU skips. Earlier authorization-gated
failures are separate. Twelve historical device paths are retired at the
current pin and all twenty remaining files differ.
The total is now 63 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine
vLLM and 16 FlashInfer. The new head/base check matches its reviewed diff;
earlier entries retain their check timestamps. The ledger has 6,956 objects.

The [indexed finalize two-shot review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-two-shot-review.md)
completes #7511's 41 changed files, all discussion and two commits, plus
structured checks of all 720 timing rows. Full-file comparisons retain
the four BF16/FP16 two-shot protocol differences instead of normalizing
away the sign-mask sideband. The review covers TP2 own-partial storage,
all historical dispatch boundaries and conditional kernel-name assertions.
The complete final-head job list and setup/summary logs show six AOT jobs
and no targeted CPU/GPU numerical jobs. All 32 head device paths still
exist at the current pin, but every file has changed.
The total is now 64 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine
vLLM and 17 FlashInfer. The new head/base check matches; earlier check
timestamps are preserved. The ledger contains 6,957 objects. The catalog
entry now links these detailed reviews without repeating their full history.

The same two-shot review completes #7650: four full-file protocol
transformations, all discussion and the sole commit, plus all 720 timing rows.
An exhaustive halfword mapping check preserves finite values but identifies
the NaN escape collision; current corresponding helpers are identical while
the surrounding bodies differ. Final-head CI again contains six AOT jobs
and no requested unit-test results. The total is now 65 manually reviewed
PRs: 27 SGLang, 11 TokenSpeed, nine vLLM and 18 FlashInfer. Only the new
head/base check is fresh in this increment; prior timestamps are retained.
The ledger has 6,958 objects.

The two-shot review also completes #7813's 44 files, all discussion and two
commits, plus all 720 timing rows. Whole-file comparisons prove 26 programs
change only temporary identifiers; the remaining six existing programs and
eight new deferred programs retain explicit load, dtype and quantization
differences. The review traces shared carry ownership and final-token drain,
the moved dispatch gates, negative-route row-zero loads and file rendezvous.
It preserves two changed-program B300 regressions and unmeasured threshold
ranges. Final-head CI has six AOT jobs and no targeted numerical results.
The total is now 66 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine
vLLM and 19 FlashInfer. Only #7813's head/base check is fresh in this increment;
earlier timestamps are retained. The ledger contains 6,959 objects.

The [indexed finalize prefetch review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-prefetch-review.md)
completes #7905's 19 files, all discussion and two commits. Complete source
comparisons cover all sixteen load rewrites and twenty-four unchanged device
programs; all 720 timing rows are reconciled with actual dispatch. The review
preserves TP8 small-batch regressions, distinguishes the separate T72 report,
and verifies that final-head green CI contains AOT jobs but no targeted GPU
results. The total is now 67 manually reviewed PRs: 27 SGLang, 11 TokenSpeed,
nine vLLM and 20 FlashInfer. Only #7905's check is new in this increment;
earlier timestamps remain. The ledger contains 6,960 objects.

The same prefetch review completes #8073's twelve changed files, all discussion
and its single commit. Explicit full-file transformations cover every changed
program; 28 other bodies and all dispatch/API/test/harness files are unchanged.
It distinguishes the pre-PDL buffer-size read, post-PDL first-token input reads
and later calculations using retained epoch values. All 720 timings are
reconciled; twelve B300 and two B200 changed-program regressions are preserved.
Final-head CI has six AOT jobs and no targeted numerical jobs. The total is now
68 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine vLLM and 21 FlashInfer.
Only #8073's head/base check is fresh in this increment; prior timestamps are
retained. The ledger contains 6,961 objects.

The prefetch/dispatch review completes #8139's 44-file diff, all discussion
and its single commit. Full-file comparisons isolate unused-macro removal and
temporary renaming in all forty device bodies from the actual TP4 class-gate
change. The new CPU functions pass 9,480 branch combinations locally; this
does not establish occupancy or full-selector behavior. Sixteen changed timing
rows are separated from 720 unchanged controls, and final-head CI again has
only AOT results. The total is now 69 manually reviewed PRs: 27 SGLang,
11 TokenSpeed, nine vLLM and 22 FlashInfer. Only #8139's head/base check is
fresh in this increment; earlier timestamps remain. The ledger has 6,962 objects.

The prefetch/dispatch review completes #8321's 24-file diff and all discussion.
Explicit inverse transformations check all eight early bodies against their
regular two-shot counterparts; the forty retained bodies are unchanged or
temporary-only rewrites. The review preserves the post-PDL boundary, cached
first-four routing indices, single control-word sampling and zero-local-token
clear initialization. Nine original CPU table/boundary cases pass locally;
T129 is the only new GPU-test token value. The 720-row table has 64 changed
rows, while 256 other reported changed measurements lack individual data.
Final-head CI contains six AOT builds and no targeted numerical test job.
The total is now 70 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine vLLM
and 23 FlashInfer. Only #8321's head/base check is fresh in this increment;
all previous per-entry timestamps remain. The ledger has 6,963 objects.

The [TP2 quad-route review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-quad-review.md)
completes #8503's 56-file diff, all discussion and commit message. Explicit
transformations account for 48 regenerated bodies; complete region comparisons
cover all four new bodies. All 52 match current device-file bytes. Ten original
CPU cases pass, including 80 quad-policy assertions. The review preserves
actual capacity limits, NVFP4 negative-route reads, identity-only test inputs,
the distinction between 32 published and 176 reported changed timing shapes,
one B300/two B200 regressions and AOT-only final-head CI. The total is now 71
manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine vLLM and 24 FlashInfer.
Only #8503's head/base check is fresh in this increment; prior per-entry check
timestamps remain. The ledger has 6,964 objects.

The [shared-expert prefetch review](../skills/llm-torch-profiler-analysis/references/flashinfer-moe-finalize-shared-prefetch-review.md)
completes #8697's eight-file diff and discussion. Full-file inverse
transformations cover four new TP4 programs; all 52 retained bodies are
unchanged, and all 56 match current bytes. Thirteen original CPU cases pass
locally. Final-head CI and JUnit establish 26 CPU passes and six skipped GPU
nodes, with no numerical execution. The card preserves mixed-rank selector
coverage, the original shared-addition order, sixteen published changed
shapes and a second set of measurements of those same shapes. The total is
now 72 manually reviewed PRs: 27 SGLang, 11 TokenSpeed, nine vLLM and
25 FlashInfer. Only #8697's head/base check is fresh in this increment;
prior per-entry timestamps remain. The ledger has 6,965 objects.

The same shared-expert review completes #8864's eleven-file diff and all
discussion. Seven explicit whole-file inverse transformations account for
the new TP2/TP8 programs; 56 retained bodies are unchanged. All 63 current
device files, binding, loader and both tests match the reviewed head.
Sixteen original CPU cases pass locally; final-head CI/JUnit show 29 CPU
passes and six GPU skips. Preserve the omitted TP8 FP16 slot, TP2's earlier
small-token branch, missing mixed-rank GPU coverage and distinction between
28 measurements and 22 shapes. The total is now 73 manually reviewed PRs:
27 SGLang, 11 TokenSpeed, nine vLLM and 26 FlashInfer. Only #8864's metadata
check is fresh in this increment; previous timestamps remain. The ledger
has 6,966 objects.

The [DeepGEMM LiteTopK producer review](../skills/llm-torch-profiler-analysis/references/deepgemm-litetopk-producer-review.md)
completes open #94's twelve-file, 983-line diff and all discussion. It corrects
the catalog's conflation of a producer-side histogram with score-free top-k
fusion. The card records additive counts, 16-bit window budgeting, FP4 stage
changes, live-BF16 parity and the sanitizer's graph skip. Current DeepGEMM
`dev` still lacks this argument and has newer wrapper/CC 10.7/TMEM contracts.
Exact-head workflow/check/status queries contain no records, so the GPU,
SASS and performance results remain author-reported. Total: 74 manually
reviewed PRs, comprising 27 SGLang, 11 TokenSpeed, nine vLLM, 26 FlashInfer
and one DeepGEMM. Only #94's check is fresh in this increment; previous
timestamps remain. The ledger has 6,967 objects. Full SGLang consumer and
migration reviews were still in progress at that increment and added nothing
to that count.

The [SGLang DeepGEMM migration review](../skills/llm-torch-profiler-analysis/references/sglang-deepgemm-migration-review.md)
completes open #44350's 49-file, 3,985-line diff, all 23 commit messages and
seven issue comments. It reconciles current main's 0.2.6 legacy/modern
dispatch, prefix-only conversion versus allocation, request grouping and
graph-owned schedule addresses. Source checks substantiate old-decoder and
cache-format limitations while qualifying the reviewer's backend list.
Exact-head base/extra/AMD logs show missing-label gates and skipped tests;
historical B300/H200 results remain author-reported, including accuracy
threshold misses and environment failures. The count is now 75: 28 SGLang,
11 TokenSpeed, nine vLLM, 26 FlashInfer and one DeepGEMM. Only #44350's
check is fresh in this increment; previous check timestamps remain. The
ledger has 6,968 objects. The SGLang LiteTopK consumer diff has been read,
but its current-source reconciliation is not yet counted as complete.

## Local validation of this increment

`pytest -q`: 273 passed, one optional test skipped, 241 subtests passed.
Pre-commit, offline Lychee and diff whitespace checks pass. New behavior tests
cover all-page discovery, partial inventory failure, label/step validation,
failed-probe stop, refreshed unmerged caches, read-only dry runs and outer-PR
git attribution. These CPU checks do not validate any proposed GPU kernel.

Further framework, kernel and open-PR audits are still required before the
broader refresh can be called complete. Downloaded metadata and file lists
are not counted as code read.
