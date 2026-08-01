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

## Search

Search progress is consumed routed-technique rounds divided by their total
reviewed round budgets. It measures budget consumption, not time remaining.
A round is counted only after the controller authenticates a distinct complete
frozen-workload candidate measurement. Agent launches/resumes, crashes,
preflight, profile capture, NCU/microbenchmarks, malformed submissions, and
unmeasured hypotheses consume no scientific round.

## Technique Rows

- `best_isolated_e2e_speedup` is a full frozen-workload result for that
  technique alone.
- `integrated` means the current integrated recipe contains the technique.
- `integrated_stack_speedup` is measured for the combined recipe.
- `marginal_attribution` remains `not_measured` unless a real ablation was run.

Never sum isolated speedups.

The default lossless search begins with `residency`, then `kernel`. Multi-GPU
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
