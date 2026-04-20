# SGLang Incident Decision Tree

Use this reference when the incident is still ambiguous and the first job is to
choose the next evidence source instead of jumping into profiling.

This reference is optimized for serving incidents that often show up late under
real traffic, not for already-isolated single-request kernel bugs.

## Core Principle

Triage in this order:

1. classify the symptom
2. capture the cheapest source-backed evidence
3. decide whether the issue is load-related, correctness-related, or infra-related
4. only then escalate to trace, profile, or replay

Do not start with `torch.profiler` by default. Profiling is expensive and is
often the wrong first move for production incidents.

## Symptom Classes

### 1. Server down or unhealthy

Typical signals:

- `/health` returns `503`
- `/health_generate` times out
- the process crashed or restarted
- router sees worker unhealthy or missing

First evidence to collect:

- `/health`
- `/health_generate`
- `/server_info`
- recent stderr/stdout logs
- crash dump status if `--crash-dump-folder` is enabled

Likely directions:

- startup / weight loading failure
- deadlock or blocked scheduler
- CUDA crash / OOM
- auth or routing misconfiguration

### 2. High latency or low throughput

Typical signals:

- TTFT regressed
- TPOT / ITL regressed
- throughput dropped at same traffic level
- queue size grows while health stays green

First evidence to collect:

- `/metrics`
- `/v1/loads?include=all`
- `/server_info`
- benchmark command or production request shape

Likely directions:

- server overloaded or under-provisioned
- scheduling / queueing problem
- cache hit rate collapse
- speculative decoding disabled or ineffective
- PD / EP / HiCache topology mismatch
- kernel/backend regression

### 3. Wrong output or behavior regression

Typical signals:

- bad generations
- model suddenly answers differently
- function calling or reasoning formatting broke
- one route or one model variant regressed

First evidence to collect:

- exact request and expected vs actual output
- `/model_info`
- `/server_info`
- current weight version
- recent deploy / config delta

Likely directions:

- wrong weights or wrong revision
- changed chat template / parser / tool config
- multimodal pre/post-process drift
- quantization / kernel correctness issue

### 4. Intermittent failure, timeout, or hang

Typical signals:

- some requests never finish
- distributed jobs hang at scale
- only high-concurrency traffic fails
- retries sometimes succeed

First evidence to collect:

- `/v1/loads?include=all`
- request dumps if enabled
- OTel trace if already enabled
- worker logs per rank

Likely directions:

- distributed divergence or collective hang
- queue starvation or retraction storm
- PD transfer stall
- storage / HiCache / remote backend stall

## Cheapest Evidence Ladder

Prefer this escalation order unless the symptom itself forces a later step:

1. health endpoints and server metadata
2. load and Prometheus metrics
3. request / crash dumps
4. request replay
5. OTel trace
6. torch profile
7. custom debug instrumentation or code-level deep dive

## What To Do First By Incident Type

### TTFT spike

Start with:

- `/v1/loads?include=all`
- `/metrics`
- `/server_info`

Watch for:

- `num_waiting_reqs` growth
- `cache_hit_rate` drop
- `token_usage` saturation
- PD queue buildup

Only escalate to trace/profile if the slowdown is not already explained by load,
queue pressure, or a known config mismatch.

### Throughput collapse under load

Start with:

- `/v1/loads?include=all`
- `/metrics`
- benchmark reproduction if available

Watch for:

- low `gen_throughput`
- high queue size
- low cache hit rate
- speculative metrics collapse
- PD transfer queues or decode prealloc queues backing up

### Crash after some requests

Start with:

- crash dump folder
- stderr/stdout logs
- request dump folder if available

Then:

- replay the crash dump or recent request dump
- if replay reproduces, escalate to CUDA crash or profiling workflows

### One request class fails but others succeed

Start with:

- exact request payload
- request dump if available
- narrow reproduction request

Likely categories:

- multimodal preprocessing edge case
- parser / structured output bug
- model-specific kernel path
- tool call formatting issue

## Escalation Gates

### Use request replay when:

- you already have a crash dump or request dump
- the issue seems workload-shape dependent
- you need a stable local reproduction before deeper profiling

### Use OTel trace when:

- you need request-stage timing
- you suspect router/worker/PD handoff delay
- you need cross-thread or cross-process latency attribution

### Use torch profiler when:

- the issue is clearly inside compute rather than queueing
- you need kernel-level attribution
- you already have a reproduction or a live target you can safely profile

### Use code-level debug paths when:

- replay and tracing still leave ambiguity
- the issue looks like a crash, hang, or correctness bug in a specific kernel or distributed path

## Report Contract

Every triage should end with:

- incident class
- exact evidence collected
- current best root-cause hypothesis
- what was ruled out
- next highest-signal step
- user-facing risk statement
