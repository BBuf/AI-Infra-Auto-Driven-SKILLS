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
6,934 objects. Fifteen additional PRs cited by the new review cards bring the
TSV to **6,949 objects**; all 38 PRs manually reviewed in this refresh were
queried again and their heads still match the reviewed diffs. The initial table
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
construction, separate indexer slots and HiSparse/PCP ownership. Three remaining
neighboring catalog PRs received fresh metadata checks only; their full source
review remains pending and is not included in the count.

## Local validation of this increment

`pytest -q`: 273 passed, one optional test skipped, 241 subtests passed.
Pre-commit, offline Lychee and diff whitespace checks pass. New behavior tests
cover all-page discovery, partial inventory failure, label/step validation,
failed-probe stop, refreshed unmerged caches, read-only dry runs and outer-PR
git attribution. These CPU checks do not validate any proposed GPU kernel.

Further framework, kernel and open-PR audits are still required before the
broader refresh can be called complete. Downloaded metadata and file lists
are not counted as code read.
