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

On 2026-04-22, the `h100_sglang` environment had:

- GPUs 6 and 7 idle before validation: 0% utilization and 4 MiB allocated.
- `sglang_bbuf` container: `sglang 0.5.10rc0`.
- `vllm/vllm-openai:latest` image: `vllm 0.19.1`.
- `nvcr.io/nvidia/tensorrt-llm/release:latest` image: `tensorrt_llm 1.0.0`.

Smoke-tested on GPU 6 and 7 with tensor parallel size 2:

| Model | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | pass: `python -m sglang.auto_benchmark run` with one tiny candidate | pass: `vllm serve` plus `vllm bench serve` | pass: `trtllm-serve serve --backend pytorch` plus TensorRT-LLM `benchmark_serving --random-ids` |
| `Qwen/Qwen2.5-1.5B-Instruct` | pass: `python -m sglang.auto_benchmark run` with one tiny candidate | pass: `vllm serve` plus `vllm bench serve` | pass: `trtllm-serve serve --backend pytorch` plus TensorRT-LLM `benchmark_serving --random-ids` |

Cleanup behavior used in validation:

- SGLang: kill the test port with `fuser`, then kill matching
  `sglang.launch_server` command lines for the same port.
- vLLM and TensorRT-LLM: run each server in a uniquely named Docker container
  and stop it with `docker rm -f`.
- Re-check GPUs 6 and 7 after every framework. They returned to 0% utilization
  and 4 MiB allocated after the final cleanup.

TensorRT-LLM notes from validation:

- The Docker image worked when running `trtllm-serve serve` through `bash -lc`.
  Overriding the Docker entrypoint directly to `trtllm-serve` missed library
  setup in this environment and failed to load `libnvinfer.so.10`.
- For `benchmark_serving --dataset-name random`, use `--random-ids` for fast
  synthetic smoke tests. Without it, TensorRT-LLM 1.0.0 asks for a ShareGPT
  `--download-path`.
