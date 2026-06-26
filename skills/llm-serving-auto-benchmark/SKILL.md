---
name: llm-serving-auto-benchmark
description: Framework-independent LLM serving benchmark skill for comparing SGLang, vLLM, TensorRT-LLM, TokenSpeed, or another serving framework. Use when a user wants to find the best deployment command for one model across multiple serving frameworks under the same workload, GPU budget, and latency SLA.
---

# LLM Serving Auto Benchmark

## Overview

Use this skill to compare LLM serving frameworks such as SGLang, vLLM,
TensorRT-LLM, and TokenSpeed for the same model and workload.

Use a config-driven workflow:

- keep launch-only capacity choices in each framework's `base_server_flags`
- put the search knobs in `search_space`
- run the same dataset scenarios for every framework
- generate a bounded candidate list from `search_space`, with the baseline
  candidate included first
- keep failed candidates in the result file
- pick the best SLA-passing candidate after normalizing the results

For model-specific starting points, prefer the shipped configs in
`configs/cookbook-llm/`. They define a framework-neutral LLM serving cookbook
model set and translate each entry into framework-native SGLang, vLLM, and
TensorRT-LLM, and TokenSpeed server flags. Validate those configs before a real
run:

```bash
python skills/llm-serving-auto-benchmark/scripts/validate_cookbook_configs.py \
  skills/llm-serving-auto-benchmark/configs/cookbook-llm
```

If you have captured target-environment `--help` files, add
`--help-dir <artifact-help-dir>`. That check only loads configs, verifies the
server flag names, and renders candidate commands; it does not launch model
servers.

Prefer native tooling when it gives better coverage:

- SGLang: `python -m sglang.auto_benchmark` when available, otherwise
  `python -m sglang.bench_serving`
- vLLM: `vllm bench sweep serve` for server-parameter sweeps, otherwise
  `vllm serve` plus `vllm bench serve`
- TensorRT-LLM: `trtllm-serve` for the OpenAI-compatible server plus the
  TensorRT-LLM serving benchmark client or a common OpenAI-compatible benchmark
  client
- TokenSpeed: `tokenspeed serve` for the OpenAI-compatible server plus
  `tokenspeed bench serve` or the same OpenAI-compatible benchmark client used
  for the other frameworks

TensorRT-LLM has one hard scope rule in this skill: the server backend is fixed
to `trtllm-serve serve --backend pytorch`. Do not search TensorRT-LLM backend
choice. If a request, config, or candidate asks for `trt`, an engine backend, or
any other non-PyTorch TensorRT-LLM server backend, reject that candidate as
unsupported for this skill and record the reason. This does not change the
benchmark client backend; the TensorRT-LLM benchmark client still uses
OpenAI-compatible modes such as `--backend openai` or `--backend openai-chat`.

Only pick a winner after each requested framework has had its main serving knobs
tuned.

Framework selection is caller-controlled. If the caller explicitly supplies a
framework list, benchmark only those enabled frameworks. Do not silently add
TensorRT-LLM or TokenSpeed just because cookbook configs or history docs exist;
record omitted frameworks as user-excluded, not unsupported.

The parameter lists in this skill are not a compatibility contract. They are
version-sensitive candidate knob families. Before every real run, record the
exact framework version or git commit and verify the concrete CLI flag names
with `--help` in the target environment.

The default search style is framework-neutral: start from a mostly pure-TP
baseline, sweep a small set of high-impact runtime knobs, and cap the first
pass around 10 candidates per framework. Do not search memory fractions by
default.

## Validation Environment

This skill is target-agnostic. It assumes any one of the following is
available, and nothing more:

- a local GPU host with Docker/Podman and the target framework images pulled;
- a remote GPU host reached via `ssh <host>` with the framework images already
  running in a container there;
- a CI runner that can exec into a pre-built image for each framework.

Do not assume a specific operator host name inside this skill's own workflow.
The concrete SSH wiring, container names, workspace paths, and HF token plumbing
for a given box live in operator-side per-host skills; this skill only requires
that the caller can reach a shell inside a container with the requested
framework installed.

Reference files are optional and version-sensitive. Treat historical flag notes
as evidence from one image, not as a compatibility guarantee for the next run.

Additional H100 validation on `2026-05-01` used two 2-card models with a
bounded search of two SGLang memory-fraction candidates and two vLLM
memory-utilization candidates. The workload was random input `512`, output
`64`, 8 prompts, and 2 warmup requests, only to prove the search and summary
path can finish quickly.

| Model | GPUs | Best SGLang | Best vLLM | Artifact root |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen3-8B` | 2x H100, TP=2 | `sglang_mem086`, 21.64 req/s, 1385.05 output tok/s, mean TTFT 70.54 ms | `vllm_mem080`, 22.88 req/s, 1464.25 output tok/s, mean TTFT 60.56 ms | `/data/bbuf/validate/core_skill_validation_20260501/qwen3_8b/auto_benchmark` |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2x H100, TP=2 | `sglang_mem080`, 24.09 req/s, 1541.92 output tok/s, mean TTFT 61.47 ms | `vllm_mem090`, 24.76 req/s, 1584.54 output tok/s, mean TTFT 58.63 ms | `/data/bbuf/validate/core_skill_validation_20260501/mistral_7b_instruct_v03/auto_benchmark` |

## Skill Scope

This skill is a playbook plus a config+validator toolchain, not a turn-key
orchestrator. The operator still launches servers, drives workloads, and writes
one normalized JSONL row per candidate.

The `scripts/` directory contains exactly two tools:

- `validate_cookbook_configs.py`: load cookbook YAML, render bounded candidate
  server commands, and check flag names against captured `--help` snapshots
  without launching servers.
- `compare_benchmark_results.py`: turn normalized per-candidate JSONL into the
  markdown and optional CSV tables described in the Output Contract.

Cookbook configs under `configs/cookbook-llm/` must pass the validator. The
shorter [references/example-plan.yaml](references/example-plan.yaml) is a
one-off runtime-plan skeleton and is not expected to pass as-is. Use
[references/result-schema.md](references/result-schema.md) as the single source
of truth for SLA key names.

## Required Inputs

Collect these before a long run:

- model and tokenizer path, target frameworks, GPU model/count, multi-node
  allowance, precision, and quantization constraints
- endpoint shape, workload source, dataset scenarios, SLA target, search budget,
  and artifact output directory
- version manifest: framework package version or git commit, container/Python
  environment, `--help` snapshots, and whether each search parameter was
  accepted by that exact CLI

If real production traffic is the goal, use the real request distribution. A
synthetic workload is fine for bring-up and first-pass comparison, but it is not
enough for a production choice.

Record each scenario's input/output length distribution in the normalized
result rows. This is now part of the profiler handoff contract: if SGLang is
slower and `sglang-sota-humanize-loop` invokes `llm-torch-profiler-analysis`,
the profiler workload must reuse the slow SGLang benchmark scenario lengths
instead of falling back to its generic prefill `4090->1` and decode `1->2048`
defaults.

## Known Gotchas

Short list of failure modes that have bitten past validation runs. Check these
before starting a long sweep.

- SGLang `fa3` attention backends need Hopper or newer. On A100, L40S, RTX
  5090, and older GPUs, drop `fa3` from the SGLang `search_space` and keep
  `flashinfer` (or `triton` when FlashInfer is unavailable).
- SGLang `bench_serving` has two SGLang-facing backends: `--backend sglang` for
  the native `/generate` endpoint and `--backend sglang-oai` for the
  OpenAI-compatible endpoint. For cross-framework comparisons, prefer
  `sglang-oai` so every framework is measured on the same request path.
- vLLM `--enable-dbo` only works when the target vLLM image is built with a
  supported all2all backend. Keep DBO out of the default candidate list unless
  the operator has verified the image.
- vLLM `--max-num-partial-prefills > 1` is model- and runtime-gated. Keep `1`
  in the default pass; raise only after a preflight with the actual model.
- vLLM current mainline was refreshed on 2026-06-26 at
  `37ce34922f7f5e58241369511130cd99c1c50bfe` and includes PR `#46735`
  fixing CUDA graph capture in Triton / NVFP4-emulation MoE. If a target image
  predates it, treat Triton-MoE graph-capture failures or eager fallback as an
  image/runtime issue before scoring it against SGLang.
- TensorRT-LLM mainline was refreshed on 2026-06-26 at
  `0722c5f47d2cae69ac1a237da51e550dd214532c`. Keep
  `kv_cache_free_gpu_memory_fraction` in shipped configs until the target
  `trtllm-serve serve --help` proves a shorter alias is accepted.
- TensorRT-LLM current mainline includes PR `#11685` and PR `#15546`, which
  affect KV block eviction and KV block-offset host staging. If a target image
  predates them, record stale-runtime risk when cache pressure, block-offset
  races, or prefix/KV residency affect benchmark rows.
- The historical TensorRT-LLM 1.0.0 multi-GPU PyTorch-backend validation used
  `--ipc=host`, `--ulimit memlock=-1`, `--ulimit stack=67108864`,
  `--shm-size=16g`, and `NCCL_IB_DISABLE=1` (for single-node) or an equivalent
  NCCL setup. Keep these as a starting point, not as a version-independent
  requirement.
- TensorRT-LLM current mainline still exposes benchmark client backends
  `openai` and `openai-chat`, not `trtllm`. This is separate from the server
  backend, which is pinned to `pytorch` by this skill.
- `trtllm` `benchmark_serving --dataset-name random` silently falls back to
  ShareGPT sampling without `--random-ids` (or `--download-path`).
- TokenSpeed is a fast-moving engine. Current mainline checked on 2026-06-26 at
  `5aedf69d6b476baa65571011de6ea60fd5a238a8` exposes `tokenspeed serve`,
  `tokenspeed bench`, `tokenspeed env`, and `tokenspeed version`. Its server
  command is `tokenspeed serve <model>`, not a `python -m tokenspeed`
  entrypoint.
- TokenSpeed's SGLang/vLLM-compatible parameter names are not always identical
  in meaning. Prefer `--max-model-len`, `--max-num-seqs`,
  `--chunked-prefill-size`, `--max-prefill-tokens`, `--max-total-tokens`,
  `--tensor-parallel-size`, `--attn-tp-size`, `--moe-tp-size`,
  `--enable-expert-parallel`, `--attention-backend`, `--moe-backend`,
  `--kv-cache-dtype`, and speculative flags only after confirming the target
  `tokenspeed serve --help` output.
- TokenSpeed has an agentic benchmark path in-tree. When the workload is
  multi-turn or tool-heavy, add a TokenSpeed-native `tokenspeed bench serve` or
  EvalScope-style run beside the common OpenAI-compatible client and record both
  result files in the same normalized row set.
- `max_seq_len` / `max_model_len` / `context_length` candidates must cover
  `max(input_len + output_len)` across every scenario, including values inside
  `search_space`, not just the baseline. The validator checks this; do not
  bypass it.

## Secrets Hygiene

- Never print `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or any upstream API key into
  a saved artifact. Pass them through container `-e VAR` (unquoted on the right
  side so the host value is inherited) and keep them out of `server_command`
  and `benchmark_command` fields written to the result JSONL.
- When a framework echoes the full argv at startup, scrub the log or redact
  token-shaped substrings before uploading the artifact.

## Fairness Rules

Use these rules throughout the benchmark:

- Run every framework on the same GPU type, GPU count, model weights, tokenizer,
  precision, quantization policy, prompt distribution, output length target, and
  sampling settings.
- Record framework version, git commit, container image, CUDA/NCCL versions, GPU
  driver, visible GPU ids, launch command, and benchmark command.
- Warm the server before measuring. Restart or clear state between candidate
  configurations when cache effects would bias the comparison.
- Compare steady-state fixed-QPS runs separately from burst throughput runs.
- Keep failed candidates in the final results with their failure reason.
- Report both raw throughput and SLA-passing throughput. The fastest failing
  candidate is not the best deployment command.

## Workflow

### 1. Preflight

Verify SGLang plus all requested comparison frameworks before starting a search.
Run only the commands for the requested framework set:

```bash
python -m sglang.launch_server --help
python -m sglang.bench_serving --help
vllm serve --help
vllm serve --help=all
vllm bench serve --help
vllm bench serve --help=all
vllm bench sweep serve --help=all
trtllm-serve serve --help
python -m tensorrt_llm.serve.scripts.benchmark_serving --help
tokenspeed serve --help
tokenspeed bench --help
tokenspeed bench serve --help
```

Use the framework-specific `--help` output in the target environment as the
source of truth. Do not keep a stale launch flag just because it appears in an
old note.

vLLM 0.19 and newer use grouped help. Plain `vllm serve --help` only shows the
groups, so capture `--help=all` before deciding whether a search knob exists.

Save these `--help` outputs into the run artifact directory. If a listed search
knob is missing from the current CLI, remove or translate that knob before
running the benchmark. Do not silently pass unknown flags.

For TensorRT-LLM, also confirm that `trtllm-serve serve --help` accepts
`--backend pytorch`. If it does not, mark TensorRT-LLM unsupported in that
environment rather than falling back to a different server backend.

For TokenSpeed, confirm both the server and benchmark entrypoints because some
installations alias the binary as `ts`. Record the exact binary used in
`server_command`.

For each framework, launch a minimal server, confirm `/v1/models` or the native
model-info endpoint, send one streaming request, run one tiny benchmark with at
least 5 requests, then save the launch command, benchmark command, server log,
and benchmark output.

Before any GPU-backed smoke run, check the requested GPU ids directly with
`nvidia-smi`. If a requested GPU is already in use, stop and record that fact.
Do not silently borrow a different GPU count for a performance comparison. It is
fine to run a smaller one-GPU smoke only when the result is clearly labeled as a
flow check rather than a fair throughput comparison.

If the target environment runs through containers, follow
[references/container-runbook.md](references/container-runbook.md) and save image
tags, pull commands, launch/benchmark logs, and cleanup commands.

### 2. Normalize The Workload

Use one canonical workload for all frameworks. Recommended JSONL row shape:

```json
{"prompt": [{"role": "user", "content": "Summarize this text."}], "output_len": 256}
{"prompt": "Write a short explanation of CUDA graphs.", "output_len": 128}
```

Optional fields:

```json
{
  "prompt": [{"role": "user", "content": "Use low temperature."}],
  "output_len": 256,
  "extra_request_body": {"temperature": 0.0, "top_p": 0.95},
  "metadata": {"source": "prod-sample"}
}
```

When converting user data:

- inspect at least 3 rows before conversion
- preserve request-level sampling options in `extra_request_body`
- do not include the final assistant answer in the prompt when that answer is
  the target completion
- keep multimodal or tool-call payloads only if all requested frameworks support
  the chosen endpoint shape

For synthetic bring-up, use the shipped two-scenario shape:

```yaml
dataset:
  kind: random
  num_prompts: 80
  scenario_names: [chat, summarization]
  input_len: [1000, 8000]
  output_len: [1000, 1000]
```

Each aligned `input_len` / `output_len` pair is one scenario. Do not take the
cartesian product unless the user asks for that.
Name each scenario and keep the aligned pair in the artifacts. For custom
datasets, compute or record representative `input_len` and `output_len`
buckets, at least p50 and p95 when possible, so later profiler runs can match
the slow bucket rather than profiling an unrelated synthetic shape.

Before searching any sequence-length limit, compute the largest
`input_len + output_len` in the dataset. SGLang `context_length`, vLLM
`max_model_len`, TensorRT-LLM `max_seq_len`, and TokenSpeed `max_model_len`
must be at least that value for every candidate that is expected to run all
scenarios.

### 3. Pick A Search Tier

Use the smallest tier that can answer the user's question:

- Tier 1: smoke and sanity. One baseline plus a few high-impact knobs.
- Tier 2: default. A bounded sweep over the most likely server settings.
- Tier 3: exhaustive. Only when the search space is already tight and the user
  accepts a long run.

Default budget:

- `num_prompts: 80` for the default cross-framework comparison; `num_prompts:
  20` per scenario is acceptable for a smoke/flow check and must be labeled as
  such in the artifact (not as a performance result).
- `search.max_candidates_per_framework: 10` for the first useful pass
- candidate generation: baseline first, then a bounded product or ordered
  candidate list from `search_space`
- at most 5 QPS search rounds unless the user asks for more
- stop early when every candidate in one framework is clearly OOM or fails the
  basic health check

Keep these in `base_server_flags` unless the user specifically wants a capacity
or memory study:

- SGLang `mem_fraction_static`
- SGLang `schedule_policy`
- vLLM `gpu_memory_utilization`
- TensorRT-LLM `kv_cache_free_gpu_memory_fraction`
- TokenSpeed `gpu_memory_utilization`

These are real knobs, but they widen the search quickly and often turn a serving
comparison into a memory-limit study.

### 4. Tune SGLang

Prefer the SGLang auto-benchmark runner when the target checkout supports it:

```bash
python -m sglang.auto_benchmark run --config /path/to/sglang.yaml
```

Otherwise launch the server manually and benchmark with:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --output-file /path/to/sglang/results.json \
  --output-details
```

Version-sensitive SGLang knob families to verify:

- `tp_size`, `pp_size`, `dp_size`, `ep_size`
- `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`
- `sampling_backend`
- `max_running_requests`, `max_queued_requests`
- `chunked_prefill_size`, `prefill_max_requests`, `max_prefill_tokens`
- `max_total_tokens`, `page_size`
- CUDA graph and piecewise CUDA graph settings
- speculative or EAGLE settings only after the non-speculative baseline is tuned

Keep `mem_fraction_static` and `schedule_policy` pinned in the default pass,
matching the shared cookbook config style.

For quick smoke tests, it is reasonable to disable CUDA graph and piecewise CUDA
graph startup work if the goal is only to prove the framework flow. Record those
flags in the artifact. Do not carry that smoke setting into a performance winner
unless the user asked to tune eager-mode serving.

### 5. Tune vLLM

Use vLLM's sweep runner when available:

```bash
vllm bench sweep serve \
  --serve-cmd 'vllm serve <model> --port 8000' \
  --bench-cmd 'vllm bench serve --backend vllm --model <model> --port 8000 --dataset-name random --num-prompts 80' \
  --serve-params /path/to/vllm_serve_params.json \
  --bench-params /path/to/vllm_bench_params.json \
  --output-dir /path/to/vllm_results
```

If sweep support is unavailable, run `vllm serve` for each candidate and measure
with `vllm bench serve`.

Version-sensitive vLLM knob families to verify:

- tensor, pipeline, data, decode-context, and expert parallelism
- `gpu_memory_utilization`
- `max_num_seqs`
- `max_num_batched_tokens`
- `max_model_len`
- `enable_chunked_prefill`, partial prefill limits, and DBO thresholds
- KV cache dtype and block size
- dtype and quantization settings
- CUDA graph capture sizes or eager-mode toggles when relevant
- prefix cache and speculative decoding settings only when the workload needs
  those features

vLLM should get a normal sweep, not one baseline command. See
[references/framework-reference.md](references/framework-reference.md) for
native command templates and cross-framework knob families. Confirm each flag on
the target image's `--help` before a run.

Keep `gpu_memory_utilization` in the baseline for the default pass. Search it
only when the question is explicitly about fitting the model or trading capacity
against throughput.

Keep DBO and all2all backend settings out of the default pass unless the target
vLLM environment is already set up for them. They are real tuning knobs, but a
candidate can fail at startup if the required all2all backend is not available.
Also preflight concurrent partial prefill before raising
`max_num_partial_prefills` above 1; some model/runtime combinations reject it at
startup.

### 6. Tune TensorRT-LLM

Use `trtllm-serve serve` as the server entrypoint when the target environment
supports it:

```bash
trtllm-serve serve <model> \
  --backend pytorch \
  --tp_size <tp> \
  --pp_size <pp> \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --host 0.0.0.0 \
  --port 8000
```

Then benchmark the OpenAI-compatible endpoint with the TensorRT-LLM serving
benchmark client or with the same OpenAI-compatible client used for the other
frameworks.

In the historical TensorRT-LLM 1.0.0 validation image,
`benchmark_serving --dataset-name random` sampled from ShareGPT unless either
`--download-path` or `--random-ids` was passed. For a fast synthetic smoke test,
pass `--random-ids`, then confirm the behavior on the target TensorRT-LLM image.

TensorRT-LLM flag names are especially version-sensitive. In the validated
TensorRT-LLM 1.0.0 image, the KV-cache memory flag accepted by
`trtllm-serve serve` was `--kv_cache_free_gpu_memory_fraction`, not
`--free_gpu_memory_fraction`. Current mainline was rechecked at
`0722c5f47d2cae69ac1a237da51e550dd214532c` on 2026-06-26. Always verify flags
with `trtllm-serve serve --help` before running a search on any GPU target.

TensorRT-LLM backend policy for this skill:

- launch the server with `--backend pytorch`
- keep `backend: pytorch` in `base_server_flags`
- do not add `backend` to `search_space`
- reject `trt`, engine-backed serving, or any other non-PyTorch TensorRT-LLM
  server backend as unsupported for this skill

Version-sensitive TensorRT-LLM knob families to verify:

- `tp_size`, `pp_size`, and `ep_size`
- max batch size, max sequence length, max number of tokens, and KV-cache budget
- inflight batching and scheduler options
- extra LLM API options YAML used by `trtllm-serve` with the PyTorch backend

The `trtllm-serve serve` CLI exposes fewer direct runtime knobs than SGLang or
vLLM. Use direct flags when they exist, then use `--extra_llm_api_options` for
PyTorch-backend settings that are not top-level CLI flags. Keep unsupported
backend or engine requests in the failure table instead of translating them.

Keep `kv_cache_free_gpu_memory_fraction` in the baseline for the default pass.
Search `max_batch_size`, `max_num_tokens`, `max_seq_len`, and validated
PyTorch-backend config options first. The server backend remains fixed to
`pytorch`.

### 7. Tune TokenSpeed

Use TokenSpeed as a first-class comparison framework, especially for agentic or
multi-turn workloads where it may be the strongest non-SGLang baseline:

```bash
tokenspeed serve <model> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size <tp> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 12288 \
  --max-num-seqs 64 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype auto \
  --trust-remote-code
```

Benchmark either with TokenSpeed's native online serving benchmark:

```bash
tokenspeed bench serve \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80
```

or with the same OpenAI-compatible client used for the other frameworks.
When TokenSpeed is a likely leader and profiler handoff will be needed, the
native benchmark can also arm torch profiling for the same request shape:

```bash
tokenspeed bench serve \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --profile \
  --profile-num-steps 5 \
  --extra-body '{"output_dir":"/data/bbuf/profiles/tokenspeed","activities":["CPU","GPU"],"with_stack":true,"profile_id":"ts-bench"}'
```

If `output_dir` is not supplied, TokenSpeed writes under
`TOKENSPEED_PROFILER_DIR`, defaulting to `/tmp`.

For large MoE or agentic checkpoints, validate these TokenSpeed knobs before
adding them to the search:

- `tensor_parallel_size`, `attn_tp_size`, `moe_tp_size`, and
  `enable_expert_parallel`
- `attention_backend`, `drafter_attention_backend`, `moe_backend`, and
  `draft_moe_backend`
- `max_num_seqs`, `chunked_prefill_size`, `max_prefill_tokens`, and
  `max_total_tokens`
- `kv_cache_dtype`, `quantization`, and prefix-cache controls
- speculative flags such as `speculative_algorithm`, `speculative_num_steps`,
  and `speculative_num_draft_tokens`
- `comm_fusion_max_num_tokens`, `enable_allreduce_fusion`, and related
  communication-fusion flags only after a target-image smoke run

If a TokenSpeed-native agentic config exists for the same model family, run it
as an additional workload lane rather than replacing the common cross-framework
scenario. Normalize its result rows with the same schema and mark
`workload.kind` accordingly.

Keep `gpu_memory_utilization` pinned in the default pass. Search it only when
the user explicitly wants a capacity study.

### 8. Normalize Results

Write one JSONL row per candidate using the schema in
[references/result-schema.md](references/result-schema.md). Then run:

```bash
python skills/llm-serving-auto-benchmark/scripts/compare_benchmark_results.py \
  --input /path/to/candidates.jsonl \
  --output /path/to/summary.md
```

Rank candidates in this order:

1. SLA passed
2. highest request throughput or goodput
3. highest output token throughput
4. lower p50 TTFT
5. lower p50 TPOT/ITL
6. lower GPU count or simpler deployment if performance is close

Keep the SLA gate itself unchanged. In the cookbook configs and normalized
result schema, TTFT SLA still uses `max_p50_ttft_ms` and TPOT SLA still uses
`max_p50_tpot_ms`; only the default cross-candidate comparison order switches
to p50 TTFT and p50 TPOT.

## Output Contract

Return a compact report with workload/SLA, hardware and framework versions, best
deployment-command tables per framework/scenario, one cross-framework comparison
table, exact launch and benchmark commands for winners, and artifact paths for
workload, raw/normalized results, CSV or markdown summary, and server logs.

When SGLang is not the winner, include a profiler handoff note with the slow
SGLang scenario name and the exact input/output lengths or percentile bucket to
pass to `llm-torch-profiler-analysis`.

When a candidate uses speculative decoding, prefix cache, offload, or an
agentic workload, record the optional normalized fields for accept length,
pre-scheduler time, cache hit rate, and memory residency. The summary script
will display those columns when present.

Include failed or excluded candidates with reasons. Explain that this table is a
record of tried configs that were not selected: candidates that failed, were
skipped by policy, or completed but missed the SLA. Add caveats for synthetic
workloads, incomplete fair searches, or framework-specific parameter
substitutions.

Use [references/framework-reference.md](references/framework-reference.md) when
you need command templates, source links, or knob-family mappings. Use
[references/example-plan.yaml](references/example-plan.yaml) as the starting
point for a full cross-framework run plan.
