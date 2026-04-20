# SGLang Endpoints and Signals

Use this reference when collecting first-round evidence from a live server.

## Authentication

Most production-safe read endpoints are public unless the server is protected by
`api_key` or `admin_api_key`.

Use:

```bash
curl -H "Authorization: Bearer <token>" ...
```

Rules:

- if only `api_key` is configured, normal protected endpoints require `api_key`
- if `admin_api_key` is configured, admin endpoints require `admin_api_key`
- some HiCache endpoints explicitly fail if `admin_api_key` is not configured at all
- `/health` and metrics-style health checks are intended to stay accessible

## Health and Metadata

### `/health`

Cheap liveness check.

- `200`: process is up enough to answer basic health
- `503`: starting, shutting down, or unhealthy

This is necessary but not sufficient. A server can return `200` and still be
latency-degraded.

### `/health_generate`

Active health check.

- tries to route a minimal generate/embedding request through the runtime
- more meaningful than `/health` for stuck schedulers or broken worker paths

Use this when requests are timing out but `/health` still returns `200`.

### `/model_info`

Returns model-path-level identity:

- `model_path`
- `tokenizer_path`
- `is_generation`
- `weight_version`
- multimodal capability flags
- model type / architectures

Use this when correctness regressions may actually be a wrong model or wrong
weight version.

### `/server_info`

Returns a broad snapshot:

- serialized `server_args`
- scheduler info
- per-DP `internal_states`
- SGLang version

Use this for almost every incident. It is the best single snapshot of the live
runtime configuration.

## Load and Capacity

### `/v1/loads?include=all`

Best structured load endpoint for triage.

Useful fields:

- `num_running_reqs`
- `num_waiting_reqs`
- `num_total_tokens`
- `num_used_tokens`
- `token_usage`
- `gen_throughput`
- `cache_hit_rate`
- optional sections:
  - `memory`
  - `speculative`
  - `lora`
  - `disaggregation`
  - `queues`

Useful query patterns:

```bash
curl -s http://127.0.0.1:30000/v1/loads
curl -s "http://127.0.0.1:30000/v1/loads?include=all"
curl -s "http://127.0.0.1:30000/v1/loads?include=core,queues,disagg"
curl -s "http://127.0.0.1:30000/v1/loads?format=prometheus"
```

Interpretation:

- high `num_waiting_reqs` with low compute throughput usually means queueing or capacity pressure
- high `token_usage` close to 1.0 suggests KV or token-capacity saturation
- low `cache_hit_rate` after a deploy can explain TTFT regressions
- PD queue fields explain transfer or prealloc bottlenecks that plain queue size hides

### `/metrics`

Prometheus endpoint.

High-value metrics during triage:

- `sglang:time_to_first_token_seconds`
- `sglang:time_per_output_token_seconds`
- `sglang:e2e_request_latency_seconds`
- `sglang:num_running_reqs`
- `sglang:num_queue_reqs`
- `sglang:num_used_tokens`
- `sglang:cache_hit_rate`
- `sglang:gen_throughput`
- `sglang:token_usage`

Use `/metrics` for trend-aware incidents. Use `/v1/loads` for a more direct
point-in-time breakdown.

## Logging and Request Capture

### `/configure_logging`

Used by `python -m sglang.srt.managers.configure_logging`.

It can:

- enable request logging
- set request logging verbosity
- enable request dump folder
- set request dump threshold

Useful payload shape:

```json
{
  "log_requests": true,
  "log_requests_level": 3,
  "dump_requests_folder": "/tmp/sglang_request_dump",
  "dump_requests_threshold": 100
}
```

Use this when the incident is ongoing and you need to capture the next failing
requests without restarting the service.

## HiCache Admin Endpoints

These are admin-sensitive. They require `admin_api_key` to be configured.

### `GET /hicache/storage-backend`

Returns tokenizer-side HiCache storage status:

- `hicache_storage_backend`
- `hicache_storage_backend_extra_config`
- `hicache_storage_prefetch_policy`
- `hicache_write_policy`

Use this when long-context or PD incidents may involve storage-backed KV reuse.

### `PUT /hicache/storage-backend`
### `DELETE /hicache/storage-backend`

Runtime attach/detach.

Do not use these casually during incident triage. They require the service to be
fully idle and are operational actions, not observation.

## Profiling and Tracing Controls

### `/start_profile`
### `/stop_profile`

Starts/stops torch profiling on the live server.

Good for:

- targeted live capture after you already narrowed the issue down

Bad first move for:

- generic latency regressions with unknown cause

### `/set_trace_level?level=N`

Controls OpenTelemetry trace verbosity when tracing was enabled at startup.

Levels:

- `0`: disabled
- `1`: important slices
- `2`: all slices except nested ones
- `3`: all slices

Use this when tracing is already configured and you need more or less span detail
without restart.

## Incident-Specific Reading Patterns

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

If tracing is enabled, then read trace data before moving to heavier profiling.

### Wrong model behavior

Read:

- `/model_info`
- `/server_info`
- exact request payload and parser/template config

Do not jump to kernel profiling until configuration drift is ruled out.
