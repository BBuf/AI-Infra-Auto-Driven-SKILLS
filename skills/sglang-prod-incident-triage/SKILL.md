---
name: sglang-prod-incident-triage
description: End-to-end production incident triage for SGLang servers and routers. Use when investigating live or recent SGLang incidents such as health-check failures, latency or throughput regressions, queue growth, request timeouts, distributed stalls, crash dumps, wrong outputs after deploys, or PD/EP/HiCache-related incidents, and when deciding whether to collect metrics, load snapshots, request dumps, OTel traces, torch profiles, or request replays before returning a concrete root-cause hypothesis with next steps.
---

# SGLang Production Incident Triage

## Overview

Use this skill to turn an SGLang production symptom into a structured triage
report with concrete evidence, a best-current hypothesis, what was ruled out,
and the next highest-signal step.

This skill is intentionally broader than the usual single-kernel or CUDA-graph
debug workflow. Those lower-level playbooks are most useful after you already
have a tight reproducer or a known failing kernel boundary. This skill is for
real serving incidents that can take minutes, hours, or days to show up under
production-like traffic:

- latency drift or throughput collapse under load
- queue growth with health checks still green
- rare crashes that depend on request mix or prompt shape
- long-tail hangs, timeouts, or distributed stalls
- wrong outputs that only appear after a deploy or topology change

The primary job here is to preserve production evidence, convert a live symptom
into a replayable incident, and only then escalate to tracing, profiling, or
kernel-level debugging.

Prefer this skill as the entry point for incidents. It should compose with
existing narrower skills instead of re-implementing them:

- use `sglang-torch-profiler-analysis` after the problem is known to be compute-side
- use `debug-distributed-hang` when the incident is clearly a TP/PP/DP/EP hang
- use `debug-cuda-crash` when replay plus coredump indicates a CUDA crash path

The reference set includes three worked examples that cover the common first
branches of this skill:

- TTFT spike with low queue time
- replay-first CUDA crash triage
- request-shaped distributed hang triage

## Core Rule

Do not start with profiling by default.

Use this escalation ladder unless the symptom itself forces a later step:

1. health and metadata
2. load and metrics
3. request dump or crash dump
4. replay
5. OTel trace
6. torch profile
7. code-level instrumentation

## Workflow

### 1. Capture the cheapest bundle first

If a live server is reachable, collect a lightweight read-only bundle before
doing anything intrusive:

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

This script collects:

- `/health`
- `/health_generate`
- `/model_info`
- `/server_info`
- `/v1/loads?include=all`
- `/v1/loads?include=core,queues,disagg,spec`
- `/metrics`
- `/hicache/storage-backend` on a best-effort basis

Then run `scripts/incident_artifact_tool.py summarize-bundle` on the collected directory.
Use its report as a first-pass readout for:

- health vs health_generate state
- live topology and runtime flags
- token capacity and memory split
- point-in-time queue and token usage
- first-pass TTFT / E2E / queue-time heuristics from Prometheus metrics

If the summary says the bundle was captured while the server was effectively
idle, do not over-read that snapshot. For intermittent incidents, move quickly
to request dump plus replay, or recollect the bundle during an active repro.

If no live server is reachable, start from the raw artifact already available:

- logs
- crash dump
- CUDA coredump
- request dump
- OTEL trace
- torch profile

### 2. Classify the symptom before choosing tools

Load [references/decision-tree.md](references/decision-tree.md) and identify the
incident class:

- server down or unhealthy
- latency or throughput regression
- wrong output or behavior regression
- intermittent timeout or hang

Do not mix these together. Pick the dominant symptom and let it determine the
next evidence source.

### 3. Read the live control-plane evidence

Load [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
when interpreting bundle contents or querying the server manually.

Use these endpoints first:

```bash
curl -s http://127.0.0.1:30000/health
curl -s http://127.0.0.1:30000/health_generate
curl -s http://127.0.0.1:30000/model_info
curl -s http://127.0.0.1:30000/server_info
curl -s "http://127.0.0.1:30000/v1/loads?include=all"
curl -s http://127.0.0.1:30000/metrics
```

Read `server_info` for:

- serving topology
- PD/EP/DP/TP settings
- speculative decoding flags
- HiCache settings
- auth or admin settings
- version and rollout identity

Read `/v1/loads?include=all` for:

- `num_running_reqs`
- `num_waiting_reqs`
- `token_usage`
- `gen_throughput`
- `cache_hit_rate`
- `speculative`
- `disaggregation`
- `queues`

### 4. Use incident-specific first responses

#### Health failure

Check:

- `/health`
- `/health_generate`
- `server_info`
- recent process logs

If `/health` is `200` but `/health_generate` fails, suspect an engine path issue,
not only an HTTP liveness issue.

#### TTFT or throughput regression

Start with:

- `server_info`
- `/v1/loads?include=all`
- `/metrics`

Look for:

- queue growth
- token-capacity saturation
- cache-hit collapse
- PD queue buildup
- speculative metrics collapse

Only escalate to trace or profile after ruling out simple load or config causes.

#### Wrong output or post-deploy behavior change

Start with:

- exact request
- `model_info`
- `server_info`
- weight version

Rule out:

- wrong model
- wrong weights
- parser/template/tooling drift
- changed runtime flags

Do not jump to kernel profiling until config drift is ruled out.

#### Timeout, stall, or hang

Start with:

- `/v1/loads?include=all`
- logs per rank or per worker
- any existing trace

If the symptom is clearly a distributed stall, switch to `debug-distributed-hang`
after the first bundle is captured.

If you need a concrete incident pattern to practice that handoff, load
[references/communication-hang-case-study.md](references/communication-hang-case-study.md).
That worked example starts from a healthy live server, triggers a request-shaped
TP hang, preserves an incident bundle while health checks time out, and only
then hands off to the specialized distributed-hang workflow.

#### Crash or CUDA exception

Start with:

- recent process logs
- crash dump folder
- launch command embedded in the crash dump
- the last healthy bundle, if one exists

If a crash dump exists, summarize it before replay:

```bash
python3 scripts/incident_artifact_tool.py summarize-dump \
  --input-file /path/to/crash_dump.pkl
```

Use that summary to answer three questions before restarting anything:

- which request or request class crashed
- whether the dump contains unfinished requests
- which launch flags must be preserved for replay

### 5. Escalate to replay, trace, or profile only when justified

Load [references/replay-trace-profile.md](references/replay-trace-profile.md) as
soon as the incident needs reproducibility or finer-grained attribution.

#### Replay

Prefer replay when:

- a crash dump exists
- a request dump exists
- the incident depends on workload shape

Use:

```bash
python3 /path/to/sglang/scripts/playground/replay_request_dump.py \
  --input-file /path/to/crash_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 128
```

If the stock replay helper is blocked by `safe_pickle_load` after recent SGLang
security hardening, and the dump is locally captured and trusted, use:

```bash
python3 scripts/replay_trusted_request_dump.py \
  --input-file /path/to/request_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 1
```

Treat `safe_pickle_load` failures on classes such as `ServerArgs` or
`GenerateReqInput` as a tooling boundary, not as evidence that the dump itself
is unusable. The trusted helper is only for locally captured or otherwise
trusted artifacts.

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

Then replay the captured dump again and inspect the resulting file with:

```bash
cuda-gdb "$(which python3)" \
  -ex "target cudacore /tmp/sglang_cuda_coredumps/cuda_coredump_<host>.<pid>.<ts>"
```

For an instruction-level clue, inspect the faulting PC neighborhood with
`x/10i <pc>` inside `cuda-gdb`, then hand off to `debug-cuda-crash` if the
failing kernel boundary is clear but the data lineage is not. This skill should
stop at incident reproduction plus failing-kernel identification, not duplicate
the full CUDA crash playbook.

If you need a concrete crash pattern to practice this flow end to end, load
[references/moe-shared-oob-case-study.md](references/moe-shared-oob-case-study.md).
That worked example intentionally corrupts one MoE routing index upstream so the
visible crash lands later in `moe_align_block_size_kernel`.

#### OTel trace

Prefer tracing when:

- request-stage timing is unclear
- router vs worker attribution is unclear
- PD prefill/decode handoff may be implicated

Use trace level changes only if tracing was enabled at startup:

```bash
curl "http://127.0.0.1:30000/set_trace_level?level=1"
curl "http://127.0.0.1:30000/set_trace_level?level=2"
```

#### Torch profile

Prefer profiling when:

- the problem is clearly compute-side
- replay already reproduced the issue
- metrics and loads cannot explain the regression

At that point, switch to `sglang-torch-profiler-analysis`. Do not duplicate its
trace-capture and table-analysis workflow here.

If you need a concrete low-noise latency pattern to practice this decision,
load
[references/ttft-prefill-not-queue-case-study.md](references/ttft-prefill-not-queue-case-study.md).
That worked example shows the cheaper path: a read-only bundle is already enough
to separate queue pressure from likely prefill-side ownership.

#### Known-good vs known-bad commit regression

If the user already knows one good commit and one bad commit, build a small
deterministic harness before doing deeper manual triage:

1. choose a stable reproducer: fixed request replay, fixed benchmark command, or a correctness checker
2. make the harness return shell success on good behavior and non-zero on bad behavior
3. run `git bisect start <bad> <good>`
4. run `git bisect run <harness>`
5. only after bisect identifies a candidate commit, return to incident triage for code-path attribution

Prefer replay-backed bisect over hand-testing prompts when the regression depends
on request shape or long-running serving state.

### 6. Hand off to a specialized skill when the boundary is clear

Switch after triage narrows the fault class:

- `sglang-torch-profiler-analysis` for kernel and overlap attribution
- `debug-distributed-hang` for collective or rank-divergence hangs
- `debug-cuda-crash` for CUDA crash reproduction and kernel API logging

Do not hand off before collecting the first incident bundle unless the user
already supplied decisive artifacts.

## References

Load only what the current step needs:

- [references/decision-tree.md](references/decision-tree.md)
  - symptom classification, escalation gates, incident report contract
- [references/endpoints-and-signals.md](references/endpoints-and-signals.md)
  - endpoint semantics, auth notes, field interpretation
- [references/replay-trace-profile.md](references/replay-trace-profile.md)
  - request dump, crash dump, replay, OTel trace, profiler handoff, regression bisect
- [references/moe-shared-oob-case-study.md](references/moe-shared-oob-case-study.md)
  - worked example: upstream top-k corruption, downstream MoE align shared-memory OOB
- [references/ttft-prefill-not-queue-case-study.md](references/ttft-prefill-not-queue-case-study.md)
  - worked example: TTFT spike with low queue time and likely prefill-side ownership
- [references/communication-hang-case-study.md](references/communication-hang-case-study.md)
  - worked example: request-shaped TP hang with healthy baseline, timed-out health checks, and distributed-hang handoff

## Scripts

- [scripts/incident_artifact_tool.py](scripts/incident_artifact_tool.py)
  - collect a read-only incident bundle
  - summarize a collected bundle into a compact first-pass report
  - summarize a trusted request dump or crash dump before replay
- [scripts/replay_trusted_request_dump.py](scripts/replay_trusted_request_dump.py)
  - replay a trusted request dump when `safe_pickle_load` blocks stock replay

## Output Contract

Return a compact report with:

- incident class
- evidence collected
- current hypothesis
- what was ruled out
- immediate operator risk
- recommended next step

If you collected a live bundle, include its path.

If you escalated to replay, trace, or profile, state why the cheaper evidence
was insufficient.
