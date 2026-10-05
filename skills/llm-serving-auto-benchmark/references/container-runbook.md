# Container Runbook

Use this runbook when the benchmark environment is container-based. It records
the exact image, command, help output, server log, benchmark log, and cleanup
step for each framework.

This runbook is target-agnostic. Every `docker run` / `docker exec` command
works on a local box, an SSH-reachable remote GPU host, or a CI runner; any
operator-side host skill only adds the SSH wrapper, container name, and
workspace path for a specific box. Substitute those values where you see
`$SGLANG_CONTAINER`, `$SGLANG_WORKSPACE`, and similar; nothing below assumes a
specific host.

## Common Setup

Pull the images that will be used:

```bash
docker pull lmsysorg/sglang:dev
docker pull vllm/vllm-openai:latest
docker pull nvcr.io/nvidia/tensorrt-llm/release:latest
# TokenSpeed images are environment-specific; set this to the image or local
# build that contains the target TokenSpeed commit.
export TOKENSPEED_IMAGE=<tokenspeed-image-with-current-source>
docker pull "$TOKENSPEED_IMAGE"
```

Use quoted Docker GPU device lists:

```bash
GPU_ARG='"device=6,7"'
docker run --gpus "$GPU_ARG" ...
```

The unquoted form `--gpus device=6,7` can be parsed incorrectly by Docker.

Mount the shared Hugging Face cache and pass tokens through environment variables
when gated models are used:

```bash
-v /data/.cache:/root/.cache \
-e HF_TOKEN \
-e HUGGINGFACE_HUB_TOKEN
```

Do not print token values into logs.

Set the run variables once and pass them into containers that need them:

```bash
export MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export TP=1
export PP=1
export PORT=8000
export RUN_DIR=/tmp/llm-serving-auto-benchmark
mkdir -p "$RUN_DIR"
```

For synthetic validation, use two aligned scenarios rather than one tiny request
shape:

```bash
# chat-like
RANDOM_INPUT_LEN=1000
RANDOM_OUTPUT_LEN=1000

# summarization-like
RANDOM_INPUT_LEN=8000
RANDOM_OUTPUT_LEN=1000
```

For a fast smoke on larger models, 20 prompts per scenario is a reasonable
minimum. Do not treat that as a performance result.

Set each framework's sequence-length limit to cover the largest scenario. For
the example above, use at least 9000 tokens for SGLang `--context-length`, vLLM
`--max-model-len`, and TensorRT-LLM `--max_seq_len`.

Before launching a server, save the help output:

```bash
python -m sglang.launch_server --help > artifacts/help/sglang_launch_server.txt
python -m sglang.benchmark.serving --help > artifacts/help/sglang_benchmark_serving.txt
vllm serve --help=all > artifacts/help/vllm_serve_all.txt
vllm bench serve --help=all > artifacts/help/vllm_bench_serve_all.txt
vllm bench sweep serve --help=all > artifacts/help/vllm_bench_sweep_serve_all.txt
trtllm-serve serve --help > artifacts/help/trtllm_serve.txt
python -m tensorrt_llm.serve.scripts.benchmark_serving --help \
  > artifacts/help/trtllm_benchmark_serving.txt
tokenspeed serve --help > artifacts/help/tokenspeed_serve.txt
vllm bench serve --help > artifacts/help/vllm_bench_serve.txt
```

## SGLang

If a prepared GPU host already has a long-running SGLang container (local or
reached via ssh; name is operator-specific), reuse it via `docker exec`
instead of creating a new container. Operator-side host skills provide the
concrete container name and workspace path for that box; this runbook assumes
the operator substitutes them:

```bash
docker exec \
  -e MODEL \
  -e TP \
  -e PORT \
  "$SGLANG_CONTAINER" bash -lc "
cd \"\$SGLANG_WORKSPACE\"
python -m sglang.launch_server \\
  --model-path \"\$MODEL\" \\
  --tp-size \"\$TP\" \\
  --host 0.0.0.0 \\
  --port \"\$PORT\"
"
```

For a fresh container:

```bash
docker run -d --name llmbench-sglang \
  --gpus "$GPU_ARG" \
  --network host \
  --ipc=host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  --entrypoint bash \
  lmsysorg/sglang:dev -lc '
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP" \
  --host 0.0.0.0 \
  --port "$PORT"
'
```

Run a tiny OpenAI-compatible smoke benchmark:




```bash
python -m sglang.benchmark.serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 8 \
  --num-prompts 4 \
  --request-rate 1 \
  --max-concurrency 2 \
  --output-file "$RUN_DIR/sglang/results.json" \
  --output-details
```

## vLLM

Server template:

```bash
docker run -d --name llmbench-vllm \
  --gpus "$GPU_ARG" \
  --network host \
  --ipc=host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  --entrypoint bash \
  vllm/vllm-openai:latest -lc '
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --trust-remote-code
'
```

Benchmark template:

```bash
docker run --rm \
  --network host \
  -v /data/.cache:/root/.cache \
  -v "$RUN_DIR:/artifacts" \
  -e MODEL \
  -e PORT \
  --entrypoint bash \
  vllm/vllm-openai:latest -lc '
vllm bench serve \
  --backend vllm \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64 \
  --save-result \
  --result-dir /artifacts/vllm \
  --result-filename results.json
'
```

Use `vllm bench sweep serve` when the target image supports it and the search
can be described with serve/bench parameter JSON files.

## TokenSpeed

Server template:

```bash
docker run -d --name llmbench-tokenspeed \
  --gpus "$GPU_ARG" \
  --network host \
  --ipc=host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  --entrypoint bash \
  "$TOKENSPEED_IMAGE" -lc '
tokenspeed serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype auto \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype auto \
  --enable-prefix-caching \
  --trust-remote-code
'
```

Benchmark template:

```bash
docker run --rm \
  --network host \
  -v /data/.cache:/root/.cache \
  -v "$RUN_DIR:/artifacts" \
  -e MODEL \
  -e PORT \
  --entrypoint bash \
  "$TOKENSPEED_IMAGE" -lc '
vllm bench serve --backend openai-chat \
  --base-url "http://127.0.0.1:$PORT" \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64 \
  --save-result \
  --result-dir /artifacts/tokenspeed
'
```

For a profiler handoff run after the plain benchmark is complete, add a writable
profile mount and arm TokenSpeed through the control port (default serving port + 1):

```bash
curl -X POST "http://127.0.0.1:${CONTROL_PORT}/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/artifacts/tokenspeed_profile","num_steps":5,"activities":["CPU","GPU"],"with_stack":true,"profile_id":"ts-bench"}'
# Drive the same workload with the common client, then POST /stop_profile if needed.
```

Some TokenSpeed images expose the binary as `ts`. If so, use `ts serve` and
the common client, then record that exact spelling in the normalized
`server_command` and `benchmark_command`.

## TensorRT-LLM

This skill supports the TensorRT-LLM PyTorch server. Current main is
PyTorch-only and deprecates `--backend`; use `--backend pytorch` only on older
images that still expose multiple backends. Do not switch the
server to `--backend trt`, an engine path, or any other backend; mark that
candidate unsupported instead.

For single-node multi-GPU TensorRT-LLM containers, keep the IPC, ulimit, shared
memory, and NCCL settings below. In a multi-GPU PyTorch-backend validation
run (captured on an H100 host; the rule is not H100-specific), the server
entered `PyTorchConfig` but failed NCCL allreduce without these container
options; the same model and candidate list passed after adding them. Expect
the same requirement on any single-node multi-GPU target.

Server template:

```bash
docker run -d --name llmbench-trtllm \
  --gpus "$GPU_ARG" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  --network host \
  -v /data/.cache:/root/.cache \
  -e MODEL \
  -e TP \
  -e PP \
  -e PORT \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  -e NCCL_IB_DISABLE=1 \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:latest -lc '
trtllm-serve serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp_size "$TP" \
  --pp_size "$PP" \
  --max_batch_size 64 \
  --max_num_tokens 8192 \
  --max_seq_len 4096 \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --trust_remote_code
'
```

Benchmark template:

```bash
docker run --rm \
  --network host \
  -v /data/.cache:/root/.cache \
  -v "$RUN_DIR:/artifacts" \
  -e MODEL \
  -e PORT \
  --entrypoint bash \
  nvcr.io/nvidia/tensorrt-llm/release:latest -lc '
python -m tensorrt_llm.serve.scripts.benchmark_serving \
  --backend openai \
  --host 127.0.0.1 \
  --port "$PORT" \
  --endpoint /v1/completions \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-ids \
  --num-prompts 80 \
  --request-rate 8 \
  --max-concurrency 64 \
  --save-result \
  --result-dir /artifacts/trtllm \
  --result-filename results.json
'
```

For TensorRT-LLM 1.0.0, the serving benchmark client `--backend` choices are
`openai` and `openai-chat`. Do not pass `--backend trtllm`. This client flag is
separate from the server backend pinned above.

## Cleanup

Use unique container names per run and clean up by name:

```bash
docker rm -f llmbench-sglang llmbench-vllm llmbench-trtllm llmbench-tokenspeed
```

If a port remains bound after container cleanup, inspect it before killing
anything:

```bash
ss -ltnp | grep ':8000'
ps -eo pid,ppid,user,etime,cmd | grep '<model-or-port>'
```

Only kill raw PIDs when the command line proves they belong to the current
validation run.

## Interface audit — 2026-10-05

SGLang's canonical client is `python -m sglang.benchmark.serving`; `bench_serving`
is a deprecated shim and the auto-benchmark module was removed (#31941). Capture
`sglang serve --help` and the client help from the target image. Current images
require CUDA 13 and a compatible host driver (#38404); the final CUDA 12 tag was
`v0.5.19-cu129`. MiniMax-M3 requires `lmsysorg/sglang:dev-minimax-m3`;
Qwen3.8-Flash-Next H200/B200 requires `lmsysorg/sglang:qwen38flashnext`.

Current SGLang uses phase-specific graph flags (#38375):
`--cuda-graph-backend-decode` / `--cuda-graph-backend-prefill` accept
`full`, `breakable`, `tc_piecewise`, `disabled`; capture sizes use
`--cuda-graph-bs-decode` / `--cuda-graph-bs-prefill` and
`--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill`. Smoke eager mode is
`--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled`.
The old size aliases were removed and `--disable-cuda-graph` is deprecated.
`--attn-dp-size` replaces the old attention-DP pair on main (#41818);
`--dp-size` remains replica DP. v0.5.21 predates this spelling: inspect image help.
Also inspect `--attn-cp-size`, `--dcp-size`, `--moe-dp-size` and the `--ep` alias.

TensorRT-LLM at `bb367fc8` is PyTorch-only after #19028. `--backend pytorch`
is a deprecated compatibility option; omit it on current main and select it
only on older images whose help lists other backends. Both
`--free_gpu_memory_fraction` (primary) and `--kv_cache_free_gpu_memory_fraction`
(alias) work. The 1.0.0 image note is historical. `--cluster_size` is deprecated
and unsupported. `--set PATH=YAML_VALUE` can override config paths after `--config`.

TokenSpeed #1236 removed its benchmark subcommand. Use the common client
`vllm bench serve --backend openai-chat` for aligned workloads. `ts` remains
an alias for `tokenspeed`. The prefix-cache off switch is
`--disable-prefix-caching`; engine flags `--api-key`, `--enable-cache-report`,
`--skip-server-warmup`, `--warmups` were removed. `--tool-call-parser` and
`--chat-template` are gateway flags. Inspect `--pipeline-parallel-size`,
`--prefill-context-parallel-size`, `--decode-context-parallel-size`,
`--lm-head-tp-size` and `--dense-gemm-backend` when tuning parallelism.

Do not force vLLM `--block-size 16`: it excludes preferred MLA/DSA backends.
Let the backend select its block size, and record its startup dispatch line.
Use the same client version across compared rows: vLLM #55508 changed chat
TTFT/E2E chunk accounting, and SGLang #39889 reports server prompt-token usage
including template tokens. Record optional client queue latency separately.
The default SGLang scheduler is FCFS; historical cookbook YAMLs no longer
force LPM on random prompts. `False` for defaults-on booleans needs an explicit
off flag and must not silently duplicate the baseline. The YAML validator
requires Python 3.10 or newer.

Source evidence: [SGLang graph flags](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/arg_groups/fields/exec_.py),
[SGLang parallel flags](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/arg_groups/fields/parallel.py),
[vLLM backend sizing](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/platforms/cuda.py),
[TensorRT-LLM serve](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/commands/serve.py),
[TokenSpeed CLI](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/cli/__main__.py).
