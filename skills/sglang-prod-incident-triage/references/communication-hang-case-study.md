# Worked Example: Shape-Armed TP All-Gather Hang

Use this case study when you want an intentional distributed incident that:

- hangs only after a specific serving request shape shows up
- looks like a generic "the server stopped answering" production failure
- does not start as a clean Python exception
- should first be captured as a live incident bundle before deeper rank-by-rank debugging
- naturally hands off to `debug-distributed-hang` after the incident boundary is clear

This example is the distributed counterpart to the CUDA crash case study.
The point is not to build a toy NCCL repro in isolation. The point is to make a
real SGLang serving stack transition from healthy to hung, preserve the
control-plane evidence, and only then narrow it into a collective mismatch
investigation.

## Target Path

One validated incident used:

- model: `Qwen/Qwen3-4B`
- `--tp 2`
- `--attention-backend triton`
- `--disable-radix-cache`
- `--disable-cuda-graph`
- `--disable-piecewise-cuda-graph`

The important runtime path was:

1. request enters the normal HTTP generate path
2. rank 0 sees an extend batch with `extend_num_tokens == 769`
3. that request shape arms a one-shot fault injection
4. the next TP logits `all_gather` on rank 0 is skipped
5. rank 1 still enters the collective and the request stops making progress

What matters is not the exact model family. What matters is the structure:

- arm the fault from a real serving batch property
- skip exactly one collective participant
- let the incident surface first as a live serving stall

## Why This Incident Is Useful

This pattern teaches the right top-level behavior for a production incident
skill:

- capture a normal bundle first
- trigger the request-shaped stall
- capture an incident bundle while the server is degraded
- compare health and control-plane behavior
- then switch to watchdog, py-spy, NCCL logs, and the specialized distributed-hang workflow

It also teaches an important negative lesson: if you jump directly into per-rank
instrumentation before preserving the live HTTP and load-plane symptoms, you
lose valuable evidence about how the incident looked to operators.

## Injection Design

Do not inject an unconditional NCCL failure.

Use a two-step design:

1. arm the incident only when rank 0 sees a real serving batch with:
   - `extend_num_tokens == 769`
2. when armed, skip one matching TP logits `all_gather` on rank 0

In one validated run, the matching collective input on rank 0 was:

```text
shape=(1, 75968) dim=-1 dtype=torch.bfloat16
```

The exact tensor shape is less important than the design goal:

- the arm condition comes from request shape
- the skipped collective comes from the real distributed model path
- the resulting symptom is a serving hang, not an immediate injected exception

## Trigger Shape

One validated trigger prompt was:

```text
"hello " * 768
```

On the validated serving build, that prompt tokenized to:

```text
prompt_tokens = 769
```

That prompt length armed the one-shot distributed fault only for the long
request, while short health or smoke requests stayed healthy beforehand.

## Production-Oriented Reproduction Flow

### 1. Launch a normal multi-rank server and verify baseline health

Before triggering the incident, collect a baseline bundle:

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

That is important. The later hang should be contrasted against a known-good
control-plane snapshot from the exact same launch.

### 2. Trigger the long request and keep the server running

Issue the long request through the normal HTTP API. Do not attach a debugger
first.

The validated request shape was:

```text
prompt_tokens=769
max_new_tokens=1
```

In the validated run, the scheduler log recorded:

```text
Prefill batch, #new-seq: 1, #new-token: 769, #cached-token: 0
```

Then the request stopped making progress instead of returning.

### 3. Capture a bundle during the hang window

While the request is still hung, collect another bundle:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle_hang
```

In one validated incident bundle, the control-plane artifacts looked like:

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

That transition is exactly why the first bundle matters. The same server moved
from:

- healthy

to:

- timed out health checks
- reset or refused load snapshots
- a hung in-flight request

### 4. Let the watchdog capture the first stack evidence

Do not kill the server immediately. Let SGLang's watchdog emit py-spy dumps.

In one validated run, the soft watchdog dumped rank-local stacks after the hung
request stayed live long enough. One TP rank showed:

```text
cuEventSynchronize
cudaEventSynchronize
synchronize (torch/cuda/streams.py:231)
process_batch_result_prefill
process_batch_result
event_loop_overlap
```

That is already enough to classify the problem as a distributed or device-side
progress stall rather than a simple HTTP routing issue.

### 5. Hand off to the specialized hang workflow

At this point, switch to `debug-distributed-hang`.

That specialized workflow should take over for:

- NCCL collective identification
- per-rank state logging
- first-diverge diffing
- binary search toward the true branch mismatch

This incident-triage skill should stop after:

- preserving the healthy and unhealthy snapshots
- proving that a real request-shaped serving incident exists
- identifying the hang as collective-related

## Expected Triage Result

If this example is behaving as intended, the triage result should say something
close to:

1. the server was healthy before the trigger request
2. a specific long prompt shape armed the incident
3. the request then hung rather than returning a normal response
4. during the incident, both `/health` and `/health_generate` timed out
5. load-plane snapshots degraded from timeout to reset or refused connections
6. watchdog py-spy output showed at least one TP rank stuck in a GPU synchronize path
7. the next highest-signal step is `debug-distributed-hang`, not torch profiling

## Anti-Patterns

Avoid these mistakes:

- building a toy standalone NCCL repro and calling it equivalent to a serving incident
- skipping the baseline bundle
- killing the server before watchdog or py-spy captures stacks
- jumping to kernel profiling before proving the failure is compute-side
- assuming `/health` staying green means the request path is healthy

## Relationship To The Other Worked Examples

This case study complements the other two references:

- `ttft-prefill-not-queue-case-study.md`
  - cheap path: bundle first, rule out queue pressure, escalate only if needed
- `moe-shared-oob-case-study.md`
  - heavy crash path: dump -> replay -> coredump -> walk one kernel upstream
- this file
  - live hang path: baseline bundle -> trigger stall -> incident bundle -> watchdog -> distributed-hang handoff
