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

## Technique Rows

- `best_isolated_e2e_speedup` is a full frozen-workload result for that
  technique alone.
- `integrated` means the current integrated recipe contains the technique.
- `integrated_stack_speedup` is measured for the combined recipe.
- `marginal_attribution` remains `not_measured` unless a real ablation was run.

Never sum isolated speedups.

## Tokens

`TOKEN-USAGE.jsonl` stores one normalized record per Agent invocation and stream
digest. The progress projection uses only the latest record for each invocation.

- `available: true, exact: true` means the runtime emitted usage.
- `available: false` means the runtime did not expose supported usage.
- Do not estimate missing tokens from text, bytes, elapsed time, or price.
- Cached input is displayed separately and is not added to `total_tokens` a
  second time.
