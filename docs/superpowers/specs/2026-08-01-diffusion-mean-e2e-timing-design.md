# Per-Request Mean E2E Timing for Diffusion Campaigns

## Goal

Make the Controller use the arithmetic mean end-to-end latency of the five
successful frozen requests as the only authoritative latency for baseline,
candidate comparison, integration, progress, target evaluation, and final
delivery. Preserve the complete five-request wall time as audit evidence.

## Motivation

The native offline benchmark emits both a complete-workload duration and a
per-request latency. The current Controller stores the complete duration in a
field named `total_s`, while historical reports commonly publish the mean per
request. Both yield the same speedup when request count is unchanged, but the
different displayed scale makes an approximately 78-second request look
inconsistent with an approximately 390-second five-request campaign.

The flow already freezes exactly five prompts and rejects missing requests.
Using an explicitly named mean therefore improves reporting without weakening
the full-workload gate.

## Chosen design

### Authoritative and audit metrics

Every normalized benchmark record carries:

- `mean_e2e_s`: arithmetic mean across successful frozen requests;
- `workload_total_s`: measured duration covering the complete five-request
  workload;
- `request_count`: successful request count, required to equal the frozen goal;
- `peak_memory_mib`, `timing_scope`, and the raw benchmark result.

The Controller computes `mean_e2e_s` itself as
`workload_total_s / request_count`. If the benchmark also reports
`latency_per_request_seconds`, the Controller requires it to agree within a
small floating-point tolerance. Executor-provided averages are never trusted
without the raw total and count.

### Versioned public artifacts

New campaigns use explicit schema-version-2 timing fields:

- `BaselineRecord`: `mean_e2e_s`, `workload_total_s`, and `request_count`;
- normalized `PERFORMANCE.json`: the same timing triple;
- delivery `PerformanceRecord`: baseline/candidate mean, workload total, and
  request count, plus the speedup;
- progress JSON: `baseline_mean_e2e_s` and `integrated_mean_e2e_s`, with the
  totals retained under separately named audit fields.

The ambiguous `total_s`, `baseline_total_s`, and `candidate_total_s` fields are
not accepted in new artifacts. Campaigns are installed from immutable
Controller revisions, so an existing campaign continues on its pinned version.
The new Controller fails closed if manually pointed at an old timing artifact
instead of silently reinterpreting it.

### Measurement and verification flow

1. Run all five prompts under the frozen command.
2. Require exactly five successful and zero failed requests.
3. Extract the complete workload duration and independently compute the mean.
4. Freeze both values in `BASELINE.json`.
5. Require every candidate and integrated run to emit the same count and both
   timing values.
6. Recompute authoritative speedup only as
   `baseline_mean_e2e_s / candidate_mean_e2e_s`.
7. Use the mean for target latency, frontier admission, progress, terminal
   status, and user-facing reports.
8. Keep workload totals for audit and consistency checks; never use them as the
   displayed optimization baseline.

Profile capture, media production, quality evidence, GPU identity, frozen
topology, and five-prompt authenticity remain unchanged.

## Error handling

The run fails closed when:

- successful request count is missing or differs from five;
- the complete workload duration is missing, non-finite, or non-positive;
- the recomputed mean is non-finite or non-positive;
- a reported per-request latency disagrees with total divided by count;
- baseline and candidate request counts differ;
- a delivery reports mean or total values that differ from its hashed raw
  benchmark evidence; or
- an artifact uses the obsolete ambiguous timing schema.

## User-facing behavior

Progress renders latency as, for example:

```text
latency     78.1011s/request baseline -> 39.3386s/request integrated
workload    390.5056s/5 requests -> 196.6928s/5 requests
```

The first line is authoritative. The second is audit context. Skill and flow
documentation consistently describe the target as a ratio of per-request mean
E2E latency. `prompt.md` uses the same wording while keeping the exact five
prompts and command unchanged.

## Testing

- Driver tests prove a five-request total is divided by five and cross-checked
  against the raw mean.
- Driver tests reject count drift and inconsistent raw per-request latency.
- Baseline, delivery, integration, verifier, progress, controller, and E2E
  fixtures use the versioned fields.
- Verifier tests reject tampered mean, total, and request count independently.
- Checked-in JSON Schemas must match deterministic regeneration.
- The full pytest, compileall, Ruff, pre-commit, diff-check, and wheel build
  suite must pass before publication.

## Delivery

Publish the implementation as a pull request, wait for required checks, squash
merge it, reinstall the merged `sglang-diffusion-auto-optimize` skill locally,
and update `/Users/bbuf/工作目录/Common/prompt.md` to the merged
Controller revision and per-request-mean wording.
