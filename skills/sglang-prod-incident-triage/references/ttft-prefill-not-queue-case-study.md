# Example: TTFT Spike With Low Queue Time

Use this case when:

- the service feels slow
- `/health` and `/health_generate` stay green
- queue growth is not obvious
- you want the full debug flow, not an immediate profiler jump

Use the same loop here:

```text
baseline bundle
  -> request dump
  -> replay the same request
  -> trace or profile only if replay still points to compute-side ownership
```

## Baseline Bundle

Collect a bundle during the actual slowdown:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_ttft_case

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle_ttft_case
```

In one run the summary looked like:

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=1 waiting=0 total=1 token_usage=0.410 throughput=29.800 cache_hit_rate=0.970
Metrics: requests=2 prompt_tokens=1540 generation_tokens=128 avg_ttft_s=3.210 avg_e2e_s=4.150 avg_queue_s=0.030
Stage Averages (max across TP ranks): prefill_forward=2.900s, request_process=0.090s
```

The first useful signal is:

- `waiting=0`
- queue time is tiny
- TTFT is still high
- `prefill_forward` dominates

That is enough to rule out queue pressure as the first-order explanation, but it
is not yet the full flow.

## Capture And Replay

Preserve the exact slow request or request batch:

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump \
  --dump-requests-threshold 1
```

Then replay the captured request on a clean target:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /tmp/sglang_request_dump \
  --parallel 1
```

The replay run should preserve the same qualitative symptom:

- TTFT stays high
- queue time stays low
- the issue looks compute-side, not queue-side

## Next Step

After replay, the likely next move is:

- OTel trace if router/worker or stage ownership is still unclear
- `sglang-torch-profiler-analysis` if replay still points at prefill-side compute

## Expected Result

1. the service is healthy
2. queue pressure is not the main explanation
3. the same slow request shape is reproducible through replay
4. the next step is trace or compute profiling, not queue debugging
