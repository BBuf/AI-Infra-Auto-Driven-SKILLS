# Progress Contract

`PROGRESS.json` is a projection of campaign evidence. It is not an independent
source of scientific state.

## Performance

```text
performance_progress =
  clamp((best_verified_speedup - 1) / (target_speedup - 1), 0, 1)
```

`best_verified_speedup` is the best independently verified isolated or
integrated full-workload result. Projected kernel gains and unverified
executor claims do not move the bar.

Every baseline, isolated candidate, and integrated candidate is optimized and
compared using the arithmetic mean E2E seconds per successful request:

```text
mean_e2e_s = workload_total_s / request_count
speedup = baseline_mean_e2e_s / candidate_mean_e2e_s
```

`request_count` must be exactly five and failed requests must be zero. Keep
`workload_total_s` as a separately labelled audit metric. Never treat the
five-request total as a single-request latency or expose ambiguous `total_s`
fields in new campaign artifacts.

## Search

Search progress is consumed routed-technique rounds divided by their total
reviewed round budgets. It measures budget consumption, not time remaining.
A round is counted after the controller authenticates a distinct complete
frozen-workload candidate measurement. Agent launches/resumes, crashes,
preflight, profile capture, NCU/microbenchmarks, pre-measurement malformed
submissions, and unmeasured hypotheses consume no scientific round. If a full
measurement passes the frozen command/native backend/5-request checks but a
later evidence or audit field is malformed, that measurement consumes one
round. Resubmitting the same run is idempotent.

## Technique Rows

- `best_isolated_e2e_speedup` is a full frozen-workload result for that
  technique alone.
- `integrated` means the current integrated recipe contains the technique.
- `integrated_stack_speedup` is measured for the combined recipe.
- `marginal_attribution` remains `not_measured` unless a real ablation was run.

Never sum isolated speedups.

The lossless-only search begins with `residency`, then `kernel`. A quality-gated
target of at least 3x uses `residency`, `cache`, `pisa`, `quantization`,
`token_pruning`, then `kernel` so high-leverage methods are not blocked by a
marginal kernel tail. Multi-GPU
does not automatically add a topology-changing lane: GPU UUIDs, rank map, and
parallel degrees are frozen by the user baseline. Collective and layout
implementations may still be optimized under that frozen topology.

A residency point is verified only when `RESIDENCY-EVIDENCE.json` binds the
profile, GPU set, memory safety envelope, transfer measurements, placement map,
conflict checks, engagement, performance, and equivalence evidence. Historical
VRAM thresholds do not advance performance or search progress.

## Tokens

`TOKEN-USAGE.jsonl` stores one normalized record per Agent invocation and stream
digest. The progress projection uses only the latest record for each invocation.

- `available: true, exact: true` means the runtime emitted usage.
- `available: false` means the runtime did not expose supported usage.
- Do not estimate missing tokens from text, bytes, elapsed time, or price.
- Cached input is displayed separately and is not added to `total_tokens` a
  second time.
