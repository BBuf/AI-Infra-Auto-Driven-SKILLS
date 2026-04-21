# Worked Example: TTFT Spike With Low Queue Time

Use this case study when you want a production-style latency incident that:

- looks like a generic "the server feels slow" complaint
- does not obviously reproduce as a crash or hang
- keeps `/health` and `/health_generate` green
- is better explained by first-pass bundle evidence than by immediate profiling
- should teach when *not* to jump to `torch.profiler`

This example is the lightweight counterpart to the MoE crash case. The point is
to show that some incidents can be narrowed down with cheap evidence first:
bundle collection, metrics parsing, and first-pass ownership hints.

## Target Symptom

The operator-visible complaint is:

- TTFT is bad
- requests feel slow
- queueing is not obviously growing

That combination is easy to misread. Many investigations immediately assume:

- the scheduler is starving requests
- the queue is backing up
- profiling must start right away

This worked example shows a different pattern:

- queue time stays very low
- queue depth stays near zero
- prefill-side compute dominates first-pass request latency

## Minimal Artifact Set

You only need a read-only incident bundle:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_ttft_case

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle_ttft_case
```

No replay, request dump, trace, or profile is required for the first-pass
diagnosis in this example.

## Validated Bundle Shape

One validated summary looked like:

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=1 waiting=0 total=1 token_usage=0.410 throughput=29.800 cache_hit_rate=0.970
Metrics: requests=2 prompt_tokens=1540 generation_tokens=128 avg_ttft_s=3.210 avg_e2e_s=4.150 avg_queue_s=0.030
Stage Averages (max across TP ranks): prefill_forward=2.900s, request_process=0.090s
```

The important part is not the exact numbers. The transferable pattern is:

- `waiting=0`
- queue time is very small
- TTFT is still large
- `prefill_forward` dominates the early request path

## What The Summary Logic Should Infer

For this bundle shape, `incident_artifact_tool.py summarize-bundle` emits:

```text
Average TTFT is high (3.210s) while average queue time is low (0.030s). Suspect compute-side prefill cost or a request-path slowdown rather than queue pressure.
Prefill forward dominates first-pass stage timing: prefill_forward≈2.900s vs request_process≈0.090s.
```

That is the key lesson of the example.

The tool is not claiming it has already found the final root cause. It is doing
something narrower and more useful for incident triage:

- ruling out queue buildup as the first-order explanation
- pointing at prefill-side ownership
- justifying why profiling may be the *next* step instead of the first step

## Why This Example Is Useful

This case study exists to teach three habits.

### 1. Separate queueing symptoms from compute-side symptoms

If TTFT is high but:

- `num_waiting_reqs` is near zero
- `avg_queue_time_seconds` is tiny
- `cache_hit_rate` is still healthy

then a queue-pressure story is weak.

### 2. Use the cheapest evidence before tracing or profiling

This is exactly the kind of incident where a first-pass bundle can already
change the next action from:

- "open a profiler now"

to:

- "collect one bundle, confirm queue is not the culprit, then decide whether prefill compute needs deeper analysis"

### 3. Escalate only when the cheaper evidence runs out

After this bundle shape, the likely next move is:

- OTel trace if stage ownership is still unclear across router or worker boundaries
- `sglang-torch-profiler-analysis` if the problem now clearly looks compute-side

The likely *wrong* next move is:

- assuming scheduler or queue bugs without evidence

## Production-Oriented Flow

### 1. Capture a bundle during the actual slowdown

Do not capture the bundle long after the incident is gone. If the server is
effectively idle, the summary will say so and the result is much less useful.

### 2. Read the four most important lines together

Do not read TTFT in isolation. Read these together:

- health state
- point-in-time load
- metrics summary
- stage averages

The example only works as intended when those lines agree with each other.

### 3. Form a bounded hypothesis

The right first-pass hypothesis is:

- "This does not look like queue pressure. Prefill-side compute or request-path work is a better explanation."

The wrong first-pass hypothesis is:

- "The exact slow kernel is already known."

### 4. Hand off only if needed

If the incident stays stable and bundle evidence keeps pointing at compute-side
prefill ownership, then hand off to:

- `sglang-torch-profiler-analysis`

If router, worker, or PD boundaries are still ambiguous, hand off to:

- tracing via the workflow in `replay-trace-profile.md`

## Expected Triage Result

If the example is behaving as intended, the triage result should say something
close to:

1. the service is alive and generation health is alive
2. queue depth is not the main bottleneck
3. queue time is too small to explain the TTFT spike
4. prefill-side work dominates early request latency
5. the next highest-signal step is trace or compute profiling, not queue debugging

## Anti-Patterns

Avoid these mistakes:

- treating TTFT alone as evidence of queue pressure
- profiling before checking `/v1/loads` and `/metrics`
- collecting the bundle only after the server goes idle
- claiming root cause when you only have first-pass ownership

## Relationship To The Other Worked Example

This case study complements `moe-shared-oob-case-study.md`.

- the MoE case teaches the heavy incident path:
  crash dump -> replay -> coredump -> walk upstream
- this TTFT case teaches the cheap incident path:
  bundle -> summarize -> narrow the likely owner -> escalate only if needed
