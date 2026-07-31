# Progress Contract

`PROGRESS.json` projects durable evidence. It is not scientific source data.

## Performance

```text
performance_progress =
  clamp((best_verified_speedup - 1) / (target_speedup - 1), 0, 1)
```

`best_verified_speedup` is the best deterministically verified isolated or
integrated full-workload result. Projected kernel gains and unsubmitted claims
do not move the bar.

## Search

Search progress is submitted scientific rounds divided by the total reviewed
round budgets. It is not elapsed time or an ETA. Infrastructure failures and
preflight failures consume no round.

## Technique Rows

- `suggested` means profile evidence exposed a possible lane.
- `active` means the current root agent owns its sole work order.
- `verified` means a complete isolated run passed deterministic gates.
- `integrated` means the current measured recipe contains it.
- `unsupported`, `no_gain`, and `blocked` come from explicit dispositions.
- `marginal_attribution` stays `not_measured` without a real ablation.

Never sum isolated speedups.

## Interactive Boundary

`yielded: true` with `status: AWAITING_AGENT` is not terminal. Read
`legal_actions` and continue with `claim` or `skip`.

The CLI does not report per-role or per-agent tokens. The AI owner is the
current conversation, whose token usage is not observable by the controller.
