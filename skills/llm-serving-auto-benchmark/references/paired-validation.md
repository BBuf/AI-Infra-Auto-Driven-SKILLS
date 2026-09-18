# Performance and accuracy after a kernel change

Use when comparing an optimization branch, combining PRs, replacing a kernel,
or producing a Pareto curve. A deployment search and a code-only A/B answer
different questions; keep them separate. See the
[source contracts](../../../docs/upstream-source-contracts.md) for current
framework/branch differences and the
[DSV4.1 cases](../../llm-torch-profiler-analysis/references/dsv41-kernel-optimization.md)
for dispatch and numerical pitfalls.

## Freeze what is being compared

Record base/head SHAs and a clean source diff, image/dependency versions,
checkpoint/tokenizer revisions, GPU model/count, TP/EP/DP, cache policy, graph
settings, quantization and effective environment variables. Combining PRs means
build a separate integration branch/worktree, resolve overlaps, and record its
SHA; individual gains do not add. Check open/merged/closed status again at use.

For a code-only comparison hold deployment parallelism fixed. If comparing each
engine's best configuration, disclose different configurations for every point.
A c=1 EP1 point and c=32/64 EP4 points can form a best-configuration frontier,
but are not one fixed deployment. Preserve the original data for historical
curves; if only a screenshot remains, mark digitized points as approximate.
A new run on a different machine/date is a new experiment, not recovered JSON.

Freeze the same client version, endpoint, prompt IDs/content, lengths, sampling,
seed, request count, concurrency/arrival process and warmup policy. Save
per-request records as well as summaries, failures and truncations. Alternate
base/candidate order or repeat fresh starts where practical; report repeated
medians and spread. Very short `5 * concurrency` sweeps are exploratory:
startup/drain effects can flatter median TPOT and change throughput rankings.

## Speculation and metric definitions

A fixed **simulated acceptance length** isolates compute/scheduling cost, but
it is not actual speculative correctness. Save the simulation variable names,
values and effective worker environment. For GSM8K/AIME or any accuracy check,
remove simulation overrides and use real draft/verify acceptance. Do not reuse
a simulated-throughput service for accuracy without restarting and auditing it.

Read the actual acceptance counters. Some implementations report accepted draft
tokens plus the target's guaranteed correction/bonus token (“golden token”);
others report draft-only acceptance. A length of 5.5 need not mean 5.5 accepted
draft tokens. Record this convention, DSPARK block size and graph-padded target
M separately from request concurrency.

| Metric | Definition / caveat |
|---|---|
| Output throughput | Total generated output tokens / measured elapsed seconds, with the exact interval and success policy stated. |
| Per-GPU throughput | Above total / participating physical GPUs; do not divide again by TP or EP. |
| Interactivity | `1000 / median TPOT_ms` when using that convention; not the reciprocal of mean TPOT and not automatically server log throughput. |
| Iteration latency | With the same accounting interval, approximately emitted tokens per iteration / decode token rate. This is not per-token latency. |
| Acceptance length | State draft-only versus total committed tokens, real versus simulated, and aggregation method. |

For example, 1200 tok/s and 5.5 total tokens/iteration imply about 4.58 ms per
iteration, not per token. Server log windows, streaming TPOT and whole-request
benchmark throughput can legitimately differ. Label chart axes and legends
with metric definitions, workload, exact source revision and added PRs.

## Validate the code that actually ran

After timing, take a short representative trace or dispatch count for every
intended phase/backend. Match shape/output format, including padding, packed
expert IDs, VL routing bias and epsilon. A faster C++ router can remain unused
by the real model because of a fallback. In that case similar serving speed
or task scores validate the fallback, not the replacement kernel.

At the operator level, compare IDs/indices and integer metadata exactly where
required. For top-k, compare selected expert sets and dense-by-expert weights;
slot permutation is different from a changed set under ties. Include zero
batch, BF16 rounding/cancellation, masked rows and graph buffer reuse when the
diff affects them. Preserve clamp, scale rounding/layout, reduction order and
intermediate rounding contracts. Set numeric tolerances from the intended
operator contract before inspecting the candidate's errors.

At the model level, freeze dataset version, split, few-shot policy, prompt/chat
template, parser, max output tokens, concurrency and seed. Record the actual
held-out denominator (some GSM8K five-shot harnesses remove prompt examples).
Do not drop truncated or failed cases to improve a score. Compare matched
question IDs and prompt hashes; report base→candidate correct/incorrect
transitions and truncations. AIME 2024/2025/2026 are different datasets.

Task-score similarity is a regression observation, not proof of bitwise equality
or statistical non-inferiority. A small, nonsignificant difference cannot
establish equivalence; use a predeclared margin and enough data if that claim
is needed. Conversely, do not blame an unexecuted kernel for score fluctuation.
Keep operator correctness, dispatch coverage, task accuracy and unprofiled
performance as separate evidence in a PR description.
