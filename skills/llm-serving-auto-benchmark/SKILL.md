---
name: llm-serving-auto-benchmark
description: Cross-framework LLM serving benchmark skill for SGLang, vLLM, and TensorRT-LLM. Use when a user wants to find the best deployment command for one model across multiple serving frameworks under the same workload, GPU budget, and latency SLA.
---

# LLM Serving Auto Benchmark

## Overview

Use this skill to compare SGLang, vLLM, and TensorRT-LLM fairly for the same
model and workload.

The goal is not to run one benchmark command per framework. The goal is to run a
small, controlled search for each framework, normalize the results, and report
the best configuration under the same latency and throughput constraints.

Prefer native tooling when it gives better coverage:

- SGLang: `python -m sglang.auto_benchmark` when available, otherwise
  `python -m sglang.bench_serving`
- vLLM: `vllm bench sweep serve` for server-parameter sweeps, otherwise
  `vllm serve` plus `vllm bench serve`
- TensorRT-LLM: `trtllm-serve` for the OpenAI-compatible server plus the
  TensorRT-LLM serving benchmark client or a common OpenAI-compatible benchmark
  client

Do not declare a winner until each requested framework has had a reasonable
chance to tune its important knobs.

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
- SLA target: TTFT, TPOT/ITL, end-to-end latency, success rate, or goodput
- search budget: quick smoke, default search, or exhaustive search
- output directory for logs and result artifacts

If real production traffic is the goal, use the real request distribution. A
synthetic workload is acceptable for bring-up and broad comparison, but it is not
enough to choose a production winner.

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
python -m sglang.bench_serving --help
vllm bench serve --help
trtllm-serve --help
```

Use the framework-specific `--help` output in the target environment as the
source of truth. Do not keep a stale launch flag just because it appears in an
old note.

For each framework:

1. Launch a minimal server.
2. Confirm `/v1/models` or the framework-native model-info endpoint works.
3. Send one streaming request and verify TTFT can be measured.
4. Run one tiny benchmark with at least 5 requests.
5. Save the launch command, benchmark command, server log, and benchmark output.

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

### 3. Pick A Search Tier

Use the smallest tier that can answer the user's question:

- Tier 1: smoke and sanity. One baseline plus a few high-impact knobs.
- Tier 2: default. A bounded sweep over the most likely server settings.
- Tier 3: exhaustive. Only when the search space is already tight and the user
  accepts a long run.

Default budget:

- `num_prompts: 80` for quick comparison
- `search.max_candidates: 8` per framework for the first useful pass
- at most 5 QPS search rounds unless the user asks for more
- stop early when every candidate in one framework is clearly OOM or fails the
  basic health check

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
  --request-rate 8
```

High-impact SGLang knobs:

- `tp_size`, `pp_size`, `dp_size`, `ep_size`
- `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`
- `sampling_backend`
- `max_running_requests`, `max_queued_requests`
- `chunked_prefill_size`, `prefill_max_requests`, `max_prefill_tokens`
- `max_total_tokens`, `page_size`, `mem_fraction_static`
- CUDA graph and piecewise CUDA graph settings
- speculative or EAGLE settings only after the non-speculative baseline is tuned

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

High-impact vLLM knobs:

- tensor and pipeline parallelism
- `gpu_memory_utilization`
- `max_num_seqs`
- `max_num_batched_tokens`
- `max_model_len`
- chunked prefill settings
- KV cache dtype and block size
- dtype and quantization settings
- CUDA graph capture sizes or eager-mode toggles when relevant
- prefix cache and speculative decoding settings only when the workload needs
  those features

### 6. Tune TensorRT-LLM

Use `trtllm-serve` as the server entrypoint when the target environment supports
it:

```bash
trtllm-serve <model> --tp_size <tp> --pp_size <pp> --host 0.0.0.0 --port 8000
```

Then benchmark the OpenAI-compatible endpoint with the TensorRT-LLM serving
benchmark client or with the same OpenAI-compatible client used for the other
frameworks.

High-impact TensorRT-LLM knobs:

- `tp_size`, `pp_size`, and `ep_size`
- PyTorch backend versus TensorRT engine path when both are available
- engine build options, quantization, and plugin selections
- max batch size, max sequence length, max number of tokens, and KV-cache budget
- inflight batching and scheduler options
- extra LLM API options YAML used by `trtllm-serve`

Because TensorRT-LLM can involve engine build time, keep build artifacts and
server artifacts separate from benchmark outputs. Report build time separately
from serving performance.

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
- best candidate per framework
- overall winner under the SLA
- failed or excluded candidates with reasons
- exact launch command and benchmark command for each winner
- artifact paths: canonical workload, raw results JSONL, normalized JSONL, CSV or
  markdown summary, and key server logs
- a caveat if the workload was synthetic or if any framework did not complete a
  fair search

Use [references/framework-matrix.md](references/framework-matrix.md) when you
need command templates or source links for each framework. Use
[references/example-plan.yaml](references/example-plan.yaml) as the starting
point for a full cross-framework run plan.
