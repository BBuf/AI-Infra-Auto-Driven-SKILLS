---
name: sglang-prod-incident-triage
description: Debug SGLang serving issues from live endpoints, request or crash dumps, and replay files. Use when investigating health-check failures, latency or throughput regressions, queue growth, request timeouts, distributed stalls, crash dumps, wrong outputs after deploys, or PD/EP/HiCache-related serving issues.
---

# SGLang Serving Debug Flow

## Overview

Use this skill for the first round of SGLang serving debug.

There is one public workflow:

- collect a baseline bundle
- capture a request or crash dump
- replay on a clean target
- continue with the deeper tool only after replay

Do not start with profiling by default.

This skill should work with narrower skills instead of re-implementing them:

- `debug-cuda-crash` when replay plus coredump points to a CUDA crash path
- `debug-distributed-hang` when the problem is clearly a TP/PP/DP/EP hang
- `sglang-torch-profiler-analysis` when the issue is already narrowed to a
  compute-side path

The references also include three worked examples:

- TTFT spike with low queue time
- replay-first CUDA crash flow
- request-shaped distributed hang flow

## When To Use It

- `/health` or `/health_generate` is unhealthy
- latency or throughput regressed under serving load
- queue size grows while health still looks green
- one request class times out or hangs
- the server crashes only after some requests
- outputs changed after a deploy, topology change, or weight switch
- one older commit is known-good and a newer commit is known-bad

## Main Flows

### 1. Collect a baseline bundle

If a live server is reachable, collect a read-only bundle before anything more
intrusive:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --outdir /tmp/incident_bundle

python3 scripts/incident_artifact_tool.py summarize-bundle \
  /tmp/incident_bundle
```

If the server is protected:

```bash
python3 scripts/incident_artifact_tool.py collect-bundle \
  --base-url http://127.0.0.1:30000 \
  --token "$SGLANG_BEARER_TOKEN" \
  --outdir /tmp/incident_bundle
```

The bundle script collects:

- `/health`
- `/health_generate`
- `/model_info`
- `/server_info`
- `/v1/loads?include=all`
- `/v1/loads?include=core,queues,disagg,spec`
- `/metrics`
- `/hicache/storage-backend` on a best-effort basis

Use the summary as the quick read for:

- health vs. active health state
- topology and runtime flags
- point-in-time queue and token usage
- TTFT / E2E / queue-time heuristics from Prometheus metrics

If the summary says the bundle was captured while the server was idle, recollect
it during traffic or move quickly to dump plus replay.

If no live server is reachable, start from the best dump or log already available:

- crash dump
- request dump
- logs
- CUDA coredump
- OTel trace
- torch profile

### 2. Save the trigger request

Load [references/decision-tree.md](references/decision-tree.md) and identify the
dominant class before choosing tools:

- server down or unhealthy
- latency or throughput regression
- wrong output or behavior regression
- intermittent timeout or hang

Then preserve the request payload that actually triggers the problem:

- crash path: use `--crash-dump-folder`
- non-crash path: enable request dump or save the exact trigger request

Do not jump straight from a live symptom to low-level debugging without first
saving something you can replay.

### 3. Replay on a clean target

Load [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
when reading the baseline bundle or the replay target.

Load [references/replay-trace-profile.md](references/replay-trace-profile.md)
and replay the captured request or dump against a clean target with the same
model path and important runtime flags.

Canonical order:

1. collect baseline bundle
2. capture request dump or crash dump
3. restart a clean debug target if needed
4. replay the same issue
5. collect replay-time logs and dumps

### 4. Switch tools only after replay

#### Replay

Prefer replay when:

- a crash dump exists
- a request dump exists
- the problem depends on request shape or workload mix

If a crash dump exists, summarize it first:

```bash
python3 scripts/incident_artifact_tool.py summarize-dump \
  --input-file /path/to/crash_dump.pkl
```

Then replay:

```bash
python3 /path/to/sglang/scripts/playground/replay_request_dump.py \
  --input-file /path/to/crash_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 128
```

If stock replay is blocked by `safe_pickle_load` and the dump is locally
captured and trusted, use:

```bash
python3 scripts/replay_trusted_request_dump.py \
  --input-file /path/to/request_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 1
```

If replay indicates a CUDA crash path, restart the same build with coredumps
enabled before reproducing again:

```bash
SGLANG_CUDA_COREDUMP=1 \
SGLANG_CUDA_COREDUMP_DIR=/tmp/sglang_cuda_coredumps \
python -m sglang.launch_server \
  --model-path ... \
  --crash-dump-folder /tmp/sglang_crash_dump \
  ...
```

Then inspect the generated coredump:

```bash
cuda-gdb "$(which python3)" \
  -ex "target cudacore /tmp/sglang_cuda_coredumps/cuda_coredump_<host>.<pid>.<ts>"
```

For a concrete replay-first crash flow, load
[references/moe-shared-oob-case-study.md](references/moe-shared-oob-case-study.md).

#### OTel trace

Prefer tracing when:

- request-stage timing is unclear
- router vs. worker attribution is unclear
- PD prefill/decode transfer may be implicated

Only change trace level if tracing was enabled at startup:

```bash
curl "http://127.0.0.1:30000/set_trace_level?level=1"
curl "http://127.0.0.1:30000/set_trace_level?level=2"
```

#### Torch profile

Prefer profiling when:

- the issue is already narrowed to compute-side ownership
- replay already reproduces the problem
- metrics and loads do not explain the regression

At that point, switch to `sglang-torch-profiler-analysis`. Do not duplicate
its profiling workflow here.

For a low-noise bundle-first latency example, load
[references/ttft-prefill-not-queue-case-study.md](references/ttft-prefill-not-queue-case-study.md).

#### Distributed hang

If the issue looks like a collective stall, preserve the trigger request,
replay it on a clean target, collect the replay-time hang logs, and then switch
to `debug-distributed-hang`.

For a concrete example of that flow, load
[references/communication-hang-case-study.md](references/communication-hang-case-study.md).

#### Known-good vs. known-bad regression

If one commit is known-good and another is known-bad, build a deterministic
harness before doing deeper manual debugging:

1. choose a stable reproducer: request replay, benchmark command, or correctness check
2. make the harness return `0` on good behavior and non-zero on bad behavior
3. run `git bisect start <bad> <good>`
4. run `git bisect run <harness>`
5. return here only after a candidate commit is isolated

Prefer replay-backed bisect when the regression depends on request shape or
long-running serving state.

### 6. Switch tools when the boundary is clear

Switch tools after the first round of debugging narrows the fault class:

- `sglang-torch-profiler-analysis` for kernel and overlap attribution
- `debug-distributed-hang` for collective or rank-divergence hangs
- `debug-cuda-crash` for CUDA crash reproduction and kernel API logging

Do not switch tools before collecting the first bundle unless the user already has
decisive logs or dumps.

## References

Load only what the current step needs:

- [references/decision-tree.md](references/decision-tree.md)
  - symptom classification, when to switch tools, return format
- [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
  - endpoint semantics, auth notes, field interpretation
- [references/replay-trace-profile.md](references/replay-trace-profile.md)
  - request dump, crash dump, replay, trace, profiler step, bisect
- [references/moe-shared-oob-case-study.md](references/moe-shared-oob-case-study.md)
  - worked example: upstream top-k corruption, downstream MoE align shared-memory OOB
- [references/ttft-prefill-not-queue-case-study.md](references/ttft-prefill-not-queue-case-study.md)
  - worked example: TTFT spike with low queue time, request replay, and likely prefill-side ownership
- [references/communication-hang-case-study.md](references/communication-hang-case-study.md)
  - worked example: request-shaped TP hang with request replay and distributed-hang debug flow

## Scripts

- [scripts/incident_artifact_tool.py](scripts/incident_artifact_tool.py)
  - collect a read-only live bundle
  - summarize a collected bundle into a compact quick summary
  - summarize a trusted request dump or crash dump before replay
- [scripts/replay_trusted_request_dump.py](scripts/replay_trusted_request_dump.py)
  - replay a trusted request dump when `safe_pickle_load` blocks stock replay

## Output Contract

Return:

- issue type
- exact clues collected
- current best hypothesis
- what was ruled out
- next step
- user-facing risk statement

If a live bundle was collected, include its path.

If replay, trace, or profiling was chosen, state why the cheaper first clues were not
enough.
