# SGLang Endpoints and Signals

Use this reference when checking a live server.

## Auth

Python ingress authorization has three levels:

- `/health*`, `/ready*`, `/metrics*` and OPTIONS are always open.
- NORMAL routes need `api_key` when configured.
- ADMIN_OPTIONAL routes accept no key when neither is configured; use the API key if it is the only key, and the admin key whenever it is configured. When both are set, the API key is rejected for admin routes.

`/configure_logging`, `/start_profile`, `/stop_profile`, `/set_trace_level`,
`/hicache/*`, `/abort_request`, `/set_internal_state`, `/freeze_gc` and
expert-distribution routes are ADMIN_OPTIONAL. HiCache GET/PUT/DELETE additionally
require an admin key to be configured; clear does not add that check.
Pass the admin key to `collect-bundle --token` when both keys are configured.

```bash
curl -H "Authorization: Bearer <admin token>" ...
```

## Core Endpoints

### `/health` and `/health_generate`

Both use the same handler and run one-token generation by default. They return
503 during startup/shutdown or when no detokenizer output arrives within
`SGLANG_HEALTH_CHECK_TIMEOUT` (default 20 s). Use a client timeout of at least
25 s; transport timeout is not a server 503 verdict.
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` disables generation only for
`/health`. `SGLANG_DIAG_BYPASS_HEALTH_GENERATE=1` forces success and hides hangs.

### `/ready`

No generation. Returns 503 while paused, draining or not Up. `/ready` 503 with
`/health` 200 points to paused/draining state. Added by #37488.

### `/model_info`

Use for model identity:

- `model_path`
- `tokenizer_path`
- `is_generation`
- `weight_version`
- multimodal flags
- model type or architectures

This is the first check for wrong-output or wrong-weight problems.

### `/server_info`

Use for runtime shape:

- flattened resolved startup fields and the original `launch_command`
- scheduler info
- per-DP `internal_states`; `memory_usage.graph` maps capture phases to GiB
- SGLang version

This is usually the single best live snapshot.

## Load And Capacity

### `/v1/loads?include=all`

Best structured load endpoint for a first pass. The response is
`{timestamp, version, accelerator, num_accelerators, loads:[...]}`; there is no
`aggregate`. Sum running/waiting requests across DP ranks. Report the maximum
token usage and per-rank cache-hit/utilization values. Valid include values are
`core,memory,spec,lora,disagg,queues,all`; others return 400. Snapshots publish
every 15 decode iterations by default, so readings can lag.
`loads[*].dp_rank` identifies the rank; optional `speculative`, `memory`,
`disaggregation`, `queues` carry their nested fields.

Useful fields:

- `num_running_reqs`
- `num_waiting_reqs`
- `num_total_tokens`
- `num_used_tokens`
- `token_usage`
- `gen_throughput`
- `cache_hit_rate`
- `memory`
- `speculative`
- `disaggregation`
- `queues`

Useful queries:

```bash
curl -s http://127.0.0.1:30000/v1/loads
curl -s "http://127.0.0.1:30000/v1/loads?include=all"
curl -s "http://127.0.0.1:30000/v1/loads?include=core,queues,disagg"
curl -s "http://127.0.0.1:30000/v1/loads?format=prometheus"
```

What to look for:

- high `num_waiting_reqs` with low compute throughput usually means queueing or capacity pressure
- `token_usage` near `1.0` usually means KV or token-capacity pressure
- low `cache_hit_rate` after a deploy can explain TTFT regressions
- PD queue fields often explain transfer or prealloc bottlenecks hidden by plain queue size

### `/metrics`

Prometheus endpoint, mounted only with `--enable-metrics` (otherwise 404).
Use it when you need trends rather than one live snapshot. TTFT/E2E/TPOT
histograms carry `is_streaming` labels.

High-value metrics:

- `sglang:time_to_first_token_seconds`
- `sglang:inter_token_latency_seconds`
- `sglang:request_time_per_output_token_seconds`
- `sglang:e2e_request_latency_seconds`
- `sglang:num_running_reqs`
- `sglang:num_queue_reqs`
- `sglang:num_used_tokens`
- `sglang:cache_hit_rate`
- `sglang:gen_throughput`
- `sglang:token_usage`

Additional signals: `sglang:queue_time_seconds`, `sglang:per_stage_req_latency_seconds`,
`sglang:spec_accept_length`, `sglang:spec_accept_rate`, `sglang:num_retracted_reqs`,
`sglang:num_retracted_requests_total`, `sglang:num_paused_reqs`,
`sglang:scheduler_idle_seconds_total`, `sglang:kv_transfer_latency_ms`,
`sglang:kv_transfer_speed_gb_s`, `sglang:num_transfer_failed_reqs_total`,
`sglang:num_streaming_sessions`, `sglang:streaming_session_held_tokens`,
`sglang:get_loads_duration_seconds`.

## Request Capture

### `/configure_logging`

Used by `python -m sglang.srt.managers.configure_logging`.

Main use:

- enable request logging
- set request logging level
- enable request dump folder
- set request dump threshold

Typical payload:

```json
{
  "log_requests": true,
  "log_requests_level": 3,
  "dump_requests_folder": "/tmp/sglang_request_dump",
  "dump_requests_threshold": 1
}
```

The payload also accepts `log_level`, `log_requests_format`, `crash_dump_folder`
(runtime activation without restart), and `dump_requests_exclude_meta_keys`.
The stock CLI sends no auth, defaults its threshold to 1000 and turns request
logging off when `--log-requests` is omitted. Use authenticated curl on keyed
servers and threshold 1–10 for rare failures; dumps flush after finished requests.

Use this when the problem is ongoing and you need the next failing request
without restarting the service.

## HiCache

### `GET /hicache/storage-backend`

Returns tokenizer-side HiCache storage status:

- `hicache_storage_backend`
- `hicache_storage_backend_extra_config`
- `hicache_storage_prefetch_policy`
- `hicache_write_policy`

Use this when long-context or PD problems may involve storage-backed KV reuse.

### `PUT /hicache/storage-backend`
### `DELETE /hicache/storage-backend`

Runtime attach or detach. These are operational actions, not passive checks.

### `POST /hicache/storage-backend/clear`

Clears the contents of the attached storage backend without changing its
attachment state. This is a destructive operational action: capture the
incident bundle, relevant request dumps, and cache-status output first. Use it
only when stale or corrupt backend state is the working diagnosis and losing
the cached contents is acceptable.

## Profiling And Tracing Controls

### `/start_profile`
### `/stop_profile`

Use only after the problem is already narrowed down.

### `/set_trace_level?level=N`

Changes trace verbosity when tracing was enabled at startup.

Levels:

- `0`: disabled
- `1`: important slices
- `2`: all slices except nested ones
- `3`: all slices

## Quick Reads By Problem Type

### TTFT spike

Read:

- `/server_info`
- `/v1/loads?include=all`
- `/metrics`

Compare:

- queue size
- token usage
- cache hit rate
- PD disaggregation queues

### Hang or timeout

Read:

- `/health`
- `/health_generate`
- `/server_info`
- `/v1/loads?include=all`

If tracing is already enabled, look at trace data before heavier profiling.

### Wrong model behavior

Read:

- `/model_info`
- `/server_info`
- exact request payload and parser or template config

Do not jump to kernel profiling until config drift is ruled out.

Source audit 2026-10-05: [HTTP routes](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/entrypoints/http_server.py), [authorization](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/utils/auth.py), [load response](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/entrypoints/v1_loads.py), [metrics](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/observability/metrics_collector.py).
