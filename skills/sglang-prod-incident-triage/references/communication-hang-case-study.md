# Worked Example: Replay-First TP Communication Hang

Use this case when:

- one request hangs instead of returning
- the symptom looks like a generic serving stall
- you want the hang to follow the same skill flow as the crash case

This example should not stop at "the live server is hung". The standard path is:

```text
baseline bundle
  -> capture the trigger request
  -> replay the same request on a clean target
  -> collect replay-time hang artifacts
  -> hand off to debug-distributed-hang
```

## Injection Shape

The validated incident used a temporary serving-side fault injector with this
shape:

1. rank 0 arms a one-shot flag only when a real extend batch satisfies
   `extend_num_tokens == 769`
2. the next TP logits `all_gather` on rank 0 is skipped
3. the peer TP rank still enters the real collective

That creates a real collective mismatch:

- rank 0 returns local data
- rank 1 waits inside the collective
- the request stops making progress

One validated trigger prompt was:

```text
"hello " * 768
```

which tokenized to:

```text
prompt_tokens = 769
```

## Baseline Bundle

Before triggering the incident, collect a healthy bundle:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_ok

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle_ok
```

One validated baseline summary looked like:

```text
Health: /health=ok /health_generate=ok
Point-in-time load: running=0 waiting=0 total=0 token_usage=0.000 throughput=0.000
```

## Capture And Replay

Do not diagnose only from the first live hang. Preserve the trigger request:

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump_hang \
  --dump-requests-threshold 1
```

After the live incident is captured, restart a clean debug target with the same
model path and the same injection, then replay the captured request:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /tmp/sglang_request_dump_hang \
  --parallel 1
```

On the validated run, the replayed request hit the same serving path:

```text
Prefill batch, #new-seq: 1, #new-token: 769, #cached-token: 0
```

and then hung again instead of returning.

## Replay-Time Incident Artifacts

While the replayed request is hung, collect another bundle:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_hang
```

One validated replay-time bundle looked like:

```text
health.txt.error.json:
  TimeoutError: timed out

health_generate.txt.error.json:
  TimeoutError: timed out

loads_all.json:
  ConnectionResetError: [Errno 104] Connection reset by peer

loads_core_queues_disagg.json:
  URLError: <urlopen error [Errno 111] Connection refused>
```

Then let the watchdog capture the first stack evidence. One TP rank showed:

```text
cuEventSynchronize
cudaEventSynchronize
synchronize (torch/cuda/streams.py:231)
process_batch_result_prefill
process_batch_result
event_loop_overlap
```

## Next Step

At this point, the skill should stop and hand off to:

- `debug-distributed-hang`

The point of this example is that the hang is now:

- request-shaped
- replayable
- already narrowed to a collective-style stall

## Expected Triage Result

If the example is behaving as intended, the result should say:

1. the server was healthy before the trigger request
2. the same trigger request was preserved and replayed
3. replay reproduced the same hang
4. replay-time bundle and watchdog evidence point at a distributed stall
5. the next step is `debug-distributed-hang`, not profiling
