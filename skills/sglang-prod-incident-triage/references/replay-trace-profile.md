# Replay, Trace, and Profile

Use this reference after the first-round triage when you need reproducible
evidence rather than just live snapshots.

It is especially useful for incidents that only become repeatable after enough
real traffic has accumulated, or that depend on workload mix rather than one
obvious prompt.

## Request Dump and Replay

### Request dump

Enable request dump on a live server:

```bash
python3 -m sglang.srt.managers.configure_logging \
  --url http://127.0.0.1:30000 \
  --dump-requests-folder /tmp/sglang_request_dump \
  --dump-requests-threshold 100
```

This is useful when:

- the incident is intermittent
- you need the exact production request shape
- you do not want to restart the server

### Crash dump

If the server already runs with:

```bash
--crash-dump-folder /tmp/crash_dump
```

then SGLang records recent requests before a crash. This is the highest-value
artifact for crash reproduction.

The crash-dump tests show the dump contains at least:

- `server_args`
- `requests`
- `launch_command` on current SGLang builds

Treat the dump as the best starting point for reproduction.
Summarize it first with:

```bash
python3 scripts/summarize_request_dump.py \
  --input-file /path/to/crash_dump.pkl
```

### Replay

Use the checked-in replay tool:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-file /path/to/crash_dump.pkl \
  --host 127.0.0.1 \
  --port 30000 \
  --parallel 128
```

Or replay a folder:

```bash
python3 scripts/playground/replay_request_dump.py \
  --input-folder /path/to/request_dump_dir \
  --file-number 10 \
  --parallel 128
```

On newer SGLang builds, `safe_pickle_load` may block some captured dump files
because they include classes such as `ServerArgs` or `GenerateReqInput`.
If the artifact is locally captured and trusted, use the skill-local helper
`scripts/replay_trusted_request_dump.py` to bypass the allowlist and replay the
same requests over HTTP.
Treat this as a trust-boundary problem in the replay helper, not as evidence
that the dump is malformed.

Use replay before profiling when:

- the incident depends on a real workload mix
- the issue appears only after some number of requests
- you need to compare two builds against the same captured traffic

### CUDA coredump restart-and-replay

If replay suggests a CUDA crash path such as illegal memory access, warp
illegal instruction, or device-side assert, restart the same build with:

```bash
SGLANG_CUDA_COREDUMP=1 \
SGLANG_CUDA_COREDUMP_DIR=/tmp/sglang_cuda_coredumps \
python -m sglang.launch_server \
  --model-path ... \
  --crash-dump-folder /tmp/sglang_crash_dump \
  ...
```

SGLang auto-injects the required `CUDA_*` variables when
`SGLANG_CUDA_COREDUMP=1`, unless the shell already provided stricter values.
After reproducing the crash via replay, inspect the generated file with:

```bash
cuda-gdb "$(which python3)" \
  -ex "target cudacore /tmp/sglang_cuda_coredumps/cuda_coredump_<host>.<pid>.<ts>"
```

Useful first commands inside `cuda-gdb`:

- `where`
- `info cuda kernels`
- `x/10i <pc>`

Treat the coredump as the place to identify the *failing* kernel, not
necessarily the *source* of the bad data. It is common for the true bug to live
one kernel earlier in the routing or preprocessing chain.

For a concrete example of this exact pattern, see
[moe-shared-oob-case-study.md](moe-shared-oob-case-study.md). That example
corrupts one MoE top-k index in `topkGatingSoftmax`, but the coredump points to
the later `moe_align_block_size_kernel` shared-memory update.

## OpenTelemetry Tracing

### Bring-up

Tracing must be enabled at startup:

```bash
python -m sglang.launch_server \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317 \
  ...
```

Optionally for the router:

```bash
python -m sglang_router.launch_router \
  --enable-trace \
  --otlp-traces-endpoint localhost:4317 \
  ...
```

Useful environment variables:

```bash
export SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS=500
export SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=64
```

### Dynamic level control

If trace was enabled, adjust verbosity without restart:

```bash
curl "http://127.0.0.1:30000/set_trace_level?level=1"
curl "http://127.0.0.1:30000/set_trace_level?level=2"
curl "http://127.0.0.1:30000/set_trace_level?level=3"
```

### What tracing is good at

Tracing is best for:

- router vs worker delay attribution
- tokenizer / scheduler / detokenizer stage timing
- PD prefill/decode transfer timing
- request lifecycle evidence across processes

Tracing is not a substitute for kernel-level profiling.

### Perfetto conversion

When you already have OTEL JSON or JSONL output:

```bash
python3 scripts/convert_otel_2_perfetto.py \
  --input /tmp/otel_trace.json \
  --output /tmp/sglang_trace_perfetto.json
```

Use this when you want a timeline view that is easier to inspect than raw OTEL.

## Torch Profiling

### Live server via HTTP control

Start profiling:

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H "Content-Type: application/json" \
  -d '{"output_dir":"/tmp/incident_profile","num_steps":1,"profile_by_stage":true,"profile_stages":["prefill"],"merge_profiles":true}'
```

Stop profiling:

```bash
curl -X POST http://127.0.0.1:30000/stop_profile
```

Common useful options:

- `output_dir`
- `start_step`
- `num_steps`
- `activities`
- `record_shapes`
- `with_stack`
- `profile_by_stage`
- `merge_profiles`

On current SGLang profile-v2 builds, a prefill-focused run may write
`*-EXTEND.trace.json.gz` rather than a file literally named `PREFILL`. A single
request can also leave DECODE traces in the same directory even when the
incident is prefill-side. For prefill regressions, inspect the rank-local
`TP-0-EXTEND` trace first and treat the merged trace as a secondary artifact.

### Client-driven profile

For a running server:

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profiles
python3 -m sglang.bench_serving \
  --backend sglang \
  --num-prompts 10 \
  --profile
```

### PD-specific rule

For PD disaggregation:

- prefill workers and decode workers must be profiled separately
- do not treat a mixed PD trace as authoritative

Use dedicated URLs or separate runs.

### When to prefer torch profiling

Use it when:

- live metrics already show a real compute regression
- replay reproduced the issue
- you need kernel families, overlap, or source attribution

Do not use it as the first move for:

- server-down incidents
- wrong-output incidents
- incidents already explained by queueing or load imbalance

## Incident-to-Tool Mapping

### Crash

Best order:

1. crash dump
2. summarize dump
3. replay
4. CUDA coredump plus `cuda-gdb`
5. CUDA crash skill or deeper instrumentation

### TTFT regression

Best order:

1. metrics and loads
2. trace if stage ownership is unclear
3. torch profiler only if compute path remains suspicious

### PD transfer stall

Best order:

1. loads with `disagg`
2. trace spans across prefill/decode
3. stage-separated profile if compute is still implicated

### Throughput regression after deploy

Best order:

1. compare `server_info`
2. compare `/metrics` and `/v1/loads`
3. replay stable workload
4. profile if needed
