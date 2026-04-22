# Framework Matrix

Use this table to choose the native runner for each framework. Always verify the
actual CLI in the target container with `--help` before a long run.

| Framework | Server | Benchmark | Notes |
| --- | --- | --- | --- |
| SGLang | `python -m sglang.launch_server` | `python -m sglang.auto_benchmark` or `python -m sglang.bench_serving` | Use `auto_benchmark` when available for tiered server-flag search. `bench_serving` supports native and OpenAI-compatible endpoints. |
| vLLM | `vllm serve` | `vllm bench sweep serve` or `vllm bench serve` | `vllm bench sweep serve` can launch `vllm serve` repeatedly and sweep serve/bench parameter JSON files. |
| TensorRT-LLM | `trtllm-serve` | TensorRT-LLM serving benchmark client or a common OpenAI-compatible benchmark client | `trtllm-serve` exposes OpenAI-compatible endpoints. Separate engine build time from serving performance. |

## Source Links

- SGLang Bench Serving Guide: <https://docs.sglang.ai/developer_guide/bench_serving.html>
- vLLM benchmark sweeps: <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>
- vLLM `bench sweep serve` CLI: <https://docs.vllm.ai/en/latest/cli/bench/sweep/serve.html>
- TensorRT-LLM `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html>
- TensorRT-LLM serving benchmark tutorial: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc6/commands/trtllm-serve/run-benchmark-with-trtllm-serve.html>

## Command Templates

### SGLang

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --tp-size <tp> \
  --port 30000

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8
```

### vLLM

```bash
vllm bench sweep serve \
  --serve-cmd 'vllm serve <model> --port 8000' \
  --bench-cmd 'vllm bench serve --backend vllm --model <model> --port 8000 --dataset-name random --num-prompts 80' \
  --serve-params vllm_serve_params.json \
  --bench-params vllm_bench_params.json \
  --output-dir vllm_results
```

### TensorRT-LLM

```bash
trtllm-serve <model> \
  --tp_size <tp> \
  --pp_size <pp> \
  --host 0.0.0.0 \
  --port 8000
```

Then benchmark `http://127.0.0.1:8000/v1/completions` or
`http://127.0.0.1:8000/v1/chat/completions` with the TensorRT-LLM serving
benchmark client or the same OpenAI-compatible client used for the other
frameworks.
