# Version Notes

This skill intentionally treats framework knobs as version-sensitive. Serving
CLIs move quickly, so a future run must capture a fresh version manifest instead
of trusting the examples blindly.

## Authored Snapshot

Last updated: 2026-04-22.

The initial skill text was informed by these local/source snapshots:

| Framework | Snapshot | Notes |
| --- | --- | --- |
| SGLang | local checkout `7044d5fe7`; H100 container package `sglang 0.5.10rc0` at repo commit `30cd2cf32` | SGLang `bench_serving` help was checked in the H100 validation container. |
| vLLM | local checkout `ed2f282bc`; H100 image `vllm/vllm-openai:latest` with `vllm 0.19.1`; official docs for `vllm bench sweep serve` and benchmark sweeps | `vllm serve` and `vllm bench serve` were smoke-tested in the H100 validation image. |
| TensorRT-LLM | H100 image `nvcr.io/nvidia/tensorrt-llm/release:latest` with `tensorrt_llm 1.0.0`; official `trtllm-serve` and serving benchmark docs current on 2026-04-22 | `trtllm-serve serve` and `tensorrt_llm.serve.scripts.benchmark_serving` were smoke-tested in the H100 validation image. |

## Update Rule

When updating this skill:

1. Refresh this table with the exact source commit, package version, or docs
   version used.
2. Re-run `--help` for each server and benchmark CLI in the target environment.
3. Move renamed or removed flags out of the example run plan.
4. Record which frameworks were actually smoke-tested and which were only
   preflighted.

## H100 Validation Notes

On 2026-04-22, validation used the `h100_sglang` host:

- `sglang_bbuf` container: `sglang 0.5.10rc0`.
- `vllm/vllm-openai:latest` image: `vllm 0.19.1`.
- `nvcr.io/nvidia/tensorrt-llm/release:latest` image: `tensorrt_llm 1.0.0`.
- Hugging Face access came from the local H100 skill and was passed as
  environment variables. Do not print the token into logs.

The first smoke attempt for `Qwen/Qwen2.5-0.5B-Instruct` and
`Qwen/Qwen2.5-1.5B-Instruct` had a bad SGLang artifact: the generated config was
empty, so `python -m sglang.auto_benchmark run` failed before launching a real
candidate. Do not count those SGLang cells as validated. The later runs below
supersede that note.

### Validated Smoke Matrix

All rows used a tiny random workload: 4 prompts, 32 input tokens, 8 output
tokens, request rate 1, and max concurrency 2. These are flow checks, not
performance numbers.

| Model | GPU shape | SGLang auto benchmark | vLLM | TensorRT-LLM |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | GPUs 6,7; TP=2 | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/qwen25_7b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | pass after using `--kv_cache_free_gpu_memory_fraction`: `/tmp/llm_serving_auto_benchmark_trt_retry_20260422_092852/qwen25_7b/results.json` |
| `google/gemma-3-4b-it` | GPUs 6,7; TP=2 for SGLang/vLLM | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/gemma3_4b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | fail in TensorRT-LLM 1.0.0: server container exited during startup, including a TP=1 retry on GPU 6 |
| `mistralai/Ministral-3-8B-Instruct-2512` | GPUs 6,7; TP=2 for SGLang/vLLM | pass: `/tmp/llm_serving_auto_benchmark_three_models_20260422_091958/ministral3_8b/sglang_auto/container_results/live_results.jsonl` | pass: `vllm serve` plus `vllm bench serve` | fail in TensorRT-LLM 1.0.0: server container exited during startup, including a TP=1 retry on GPU 6 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | GPU 6 or 7; TP=1 | pass after disabling piecewise CUDA graph: `/tmp/llm_serving_auto_benchmark_tinyllama_sglang_fixed/container_results/live_results.jsonl` | pass: `/tmp/llm_serving_auto_benchmark_more_gpu7_20260422_091149/tinyllama_1_1b/vllm/results.json` | pass: `/tmp/llm_serving_auto_benchmark_tinyllama_trt_manual/results.json` |

### Validation Lessons

- SGLang smoke configs must be real files. After writing a config through
  `docker exec`, read or checksum it before starting a long run.
- For quick SGLang flow checks, add both `disable_cuda_graph: true` and
  `disable_piecewise_cuda_graph: true`. TinyLlama hit a piecewise CUDA graph
  warmup failure without the second flag.
- In TensorRT-LLM 1.0.0, `trtllm-serve serve` accepts
  `--kv_cache_free_gpu_memory_fraction`; `--free_gpu_memory_fraction` exits with
  a CLI error.
- `benchmark_serving --dataset-name random` needs `--random-ids` for fast
  synthetic tests unless a ShareGPT `--download-path` is provided.
- When using Docker with specific GPUs, quote comma-separated device lists:
  `--gpus '"device=6,7"'`. The unquoted `--gpus device=6,7` form can fail before
  the container starts.
- Check GPU idleness before every framework run, not only at the start. Other
  jobs can appear mid-validation. If the GPU count changes, record the run as a
  smoke-only check and do not compare throughput against the earlier runs.
- Clean up by port and container name. Avoid killing raw PIDs unless they are
  proven to belong to the current validation run.
