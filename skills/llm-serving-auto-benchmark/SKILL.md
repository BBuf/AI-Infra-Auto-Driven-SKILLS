---
name: llm-serving-auto-benchmark
description: Cross-framework LLM serving benchmark skill for SGLang, vLLM, and TensorRT-LLM. Use when a user wants to find the best deployment command for one model across multiple serving frameworks under the same workload, GPU budget, and latency SLA.
---

# LLM Serving Auto Benchmark

## Overview

Use this skill to compare SGLang, vLLM, and TensorRT-LLM for the same model and
workload.

Use a config-driven workflow:

- keep launch-only capacity choices in each framework's `base_server_flags`
- put the search knobs in `search_space`
- run the same dataset scenarios for every framework
- generate a bounded candidate list from `search_space`, with the baseline
  candidate included first
- keep failed candidates in the result file
- pick the best SLA-passing candidate after normalizing the results

Prefer native tooling when it gives better coverage:

- SGLang: `python -m sglang.auto_benchmark` when available, otherwise
  `python -m sglang.bench_serving`
- vLLM: `vllm bench sweep serve` for server-parameter sweeps, otherwise
  `vllm serve` plus `vllm bench serve`
- TensorRT-LLM: `trtllm-serve` for the OpenAI-compatible server plus the
  TensorRT-LLM serving benchmark client or a common OpenAI-compatible benchmark
  client

TensorRT-LLM has one hard scope rule in this skill: the server backend is fixed
to `trtllm-serve serve --backend pytorch`. Do not search TensorRT-LLM backend
choice. If a request, config, or candidate asks for `trt`, an engine backend, or
any other non-PyTorch TensorRT-LLM server backend, reject that candidate as
unsupported for this skill and record the reason. This does not change the
benchmark client backend; the TensorRT-LLM benchmark client still uses
OpenAI-compatible modes such as `--backend openai` or `--backend openai-chat`.

Only pick a winner after each requested framework has had its main serving knobs
tuned.

The parameter lists in this skill are not a compatibility contract. They are
version-sensitive candidate knob families. Before every real run, record the
exact framework version or git commit and verify the concrete CLI flag names
with `--help` in the target environment.

The default search style should stay close to SGLang auto benchmark: start from
a mostly pure-TP baseline, sweep a small set of high-impact runtime knobs, and
cap the first pass around 10 candidates per framework. Do not search memory
fractions by default.

## Required Inputs

Collect these before starting a long run:

- model path or Hugging Face repo id
- tokenizer path if it differs from the model
- target frameworks: any subset of `sglang`, `vllm`, `tensorrt-llm`
- GPU model, GPU count, and whether multi-node is allowed
- precision and quantization constraints
- endpoint shape: completions, chat completions, responses, or custom
- workload source: real traffic JSONL, ShareGPT, random synthetic, or generated
  shared-prefix synthetic
- dataset scenarios when synthetic traffic is used, for example `chat` and
  `summarization`
- SLA target: TTFT, TPOT/ITL, end-to-end latency, success rate, or goodput
- search budget: quick smoke, default search, or exhaustive search
- output directory for logs and result artifacts

Also collect a version manifest:

- framework package version and git commit when available
- container image or Python environment identifier
- `--help` snapshots for the server command and benchmark command
- whether each parameter in the search plan was accepted by that exact CLI

If real production traffic is the goal, use the real request distribution. A
synthetic workload is fine for bring-up and first-pass comparison, but it is not
enough for a production choice.

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

Verify all requested frameworks before starting a search:

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

For each framework:

1. Launch a minimal server.
2. Confirm `/v1/models` or the framework-native model-info endpoint works.
3. Send one streaming request and verify TTFT can be measured.
4. Run one tiny benchmark with at least 5 requests.
5. Save the launch command, benchmark command, server log, and benchmark output.

Before any GPU-backed smoke run, check the requested GPU ids directly with
`nvidia-smi`. If a requested GPU is already in use, stop and record that fact.
Do not silently borrow a different GPU count for a performance comparison. It is
fine to run a smaller one-GPU smoke only when the result is clearly labeled as a
flow check rather than a fair throughput comparison.

If the target environment runs through containers, follow
[references/container-runbook.md](references/container-runbook.md). Save the
image tags, pull commands, launch commands, server logs, benchmark logs, and
cleanup commands in the artifact directory.

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

For synthetic bring-up, follow the two-scenario shape used by the SGLang auto
benchmark references:

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

Before searching any sequence-length limit, compute the largest
`input_len + output_len` in the dataset. SGLang `context_length`, vLLM
`max_model_len`, and TensorRT-LLM `max_seq_len` must be at least that value for
every candidate that is expected to run all scenarios.

### 3. Pick A Search Tier

Use the smallest tier that can answer the user's question:

- Tier 1: smoke and sanity. One baseline plus a few high-impact knobs.
- Tier 2: default. A bounded sweep over the most likely server settings.
- Tier 3: exhaustive. Only when the search space is already tight and the user
  accepts a long run.

Default budget:

- `num_prompts: 80` for quick comparison
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
matching the SGLang auto benchmark cookbook style.

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
[references/parameter-coverage.md](references/parameter-coverage.md) for the
H100-verified flag families.

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

For TensorRT-LLM 1.0.0, `benchmark_serving --dataset-name random` samples from
ShareGPT unless you pass either `--download-path` or `--random-ids`. For a fast
synthetic smoke test, pass `--random-ids`.

TensorRT-LLM flag names are especially version-sensitive. In the H100
TensorRT-LLM 1.0.0 image, the KV-cache memory flag accepted by
`trtllm-serve serve` is `--kv_cache_free_gpu_memory_fraction`, not
`--free_gpu_memory_fraction`. Verify this with `trtllm-serve serve --help`
before running a search.

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

### 7. Normalize Results

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
4. lower p99 TTFT
5. lower p99 TPOT/ITL
6. lower GPU count or simpler deployment if performance is close

## Output Contract

Return a compact report with:

- workload and SLA used
- hardware and framework versions
- for each framework, one table listing the best deployment command for each
  dataset scenario and all relevant performance metrics
- one cross-framework comparison table for the selected best command per
  framework and scenario, including the command, so the deployment choice is
  clear for each dataset
- failed or excluded candidates with reasons. Explain that this table is an
  record of tried configs that were not selected: candidates that failed, were
  skipped by policy, or completed but missed the SLA.
- exact launch command and benchmark command for each winner
- artifact paths: canonical workload, raw results JSONL, normalized JSONL, CSV or
  markdown summary, and server logs needed to debug winners or failures
- a caveat if the workload was synthetic, if any framework did not complete a
  fair search, or if any framework needed framework-specific parameter
  substitutions

Use [references/framework-matrix.md](references/framework-matrix.md) when you
need command templates or source links for each framework. Use
[references/example-plan.yaml](references/example-plan.yaml) as the starting
point for a full cross-framework run plan. Use
[references/version-notes.md](references/version-notes.md) to understand which
source snapshots informed this skill and what has or has not been smoke-tested.
