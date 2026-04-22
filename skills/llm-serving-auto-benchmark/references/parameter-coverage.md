# Parameter Coverage

Do not compare SGLang, vLLM, and TensorRT-LLM by copying the same flag names
across frameworks. Compare knob families, then translate each family to the
flags that the target CLI actually accepts.

## H100 Audit Snapshot

Captured on 2026-04-22 in:

`/tmp/llm_serving_auto_benchmark_param_audit_20260422_094955`

Versions:

- SGLang: `sglang 0.5.10rc0`, repo commit
  `30cd2cf32b20355e361f7e2b3badc376fc1330af`
- vLLM image: `vllm/vllm-openai:latest`, `vllm 0.19.1`
- TensorRT-LLM image: `nvcr.io/nvidia/tensorrt-llm/release:latest`,
  `tensorrt_llm 1.0.0`

Help files captured:

- `help/sglang_launch_server.txt`
- `help/sglang_bench_serving.txt`
- `help/sglang_auto_benchmark_run.txt`
- `help/vllm_serve_all.txt`
- `help/vllm_bench_serve_all.txt`
- `help/vllm_bench_sweep_serve_all.txt`
- `help/trtllm_serve.txt`
- `help/trtllm_benchmark_serving.txt`
- `help/trtllm_bench.txt`

Observed flag counts in that environment:

| CLI | Flag Count | Note |
| --- | ---: | --- |
| `python -m sglang.launch_server --help` | 384 | Broad server surface. |
| `vllm serve --help=all` | 302 | Plain `--help` only shows config groups. |
| `vllm bench serve --help=all` | 97 | Covers random, ShareGPT, HF, custom, multimodal, and sampling knobs. |
| `vllm bench sweep serve --help=all` | 15 | Sweeps serve and bench parameter JSON files. |
| `trtllm-serve serve --help` | 23 | Direct serving surface is smaller; use backend config or build steps for deeper tuning. |
| `python -m tensorrt_llm.serve.scripts.benchmark_serving --help` | 53 | OpenAI-compatible benchmark client. |

## Knob Family Mapping

| Family | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| Parallelism | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`, `--expert-parallel-size`, `--attention-context-parallel-size` | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--decode-context-parallel-size`, `--enable-expert-parallel` | `--tp_size`, `--pp_size`, `--ep_size`, `--gpus_per_node`, `--cluster_size` |
| Memory and KV cache | `--mem-fraction-static`, `--max-total-tokens`, `--kv-cache-dtype`, `--page-size`, `--cpu-offload-gb` | `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`, `--block-size`, `--cpu-offload-gb` | `--kv_cache_free_gpu_memory_fraction`; combine with `--max_num_tokens`, `--max_seq_len`, and `--max_batch_size` |
| Batching and scheduler | `--max-running-requests`, `--max-queued-requests`, `--schedule-policy`, `--schedule-conservativeness`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--prefill-max-requests` | `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--max-num-partial-prefills`, `--max-long-partial-prefills`, `--long-prefill-token-threshold`, `--enable-dbo`, `--dbo-prefill-token-threshold`, `--dbo-decode-token-threshold` | `--max_batch_size`, `--max_num_tokens`, `--max_seq_len`; more scheduler details may require `--extra_llm_api_options` |
| Attention/backend | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`, `--sampling-backend`, `--grammar-backend` | `--attention-backend`, `--gdn-prefill-backend`, `--mm-encoder-attn-backend` | `--backend`; PyTorch backend details are mostly config-driven |
| CUDA graph and compile | `--disable-cuda-graph`, `--cuda-graph-bs`, `--cuda-graph-max-bs`, `--disable-piecewise-cuda-graph`, `--piecewise-cuda-graph-max-tokens`, `--enable-torch-compile` | `--enforce-eager`, `--compilation-config`, `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | use `--extra_llm_api_options`; server logs expose the resolved `PyTorchConfig` |
| Prefix/speculative | `--disable-radix-cache`, `--disable-chunked-prefix-cache`, `--speculative-algorithm`, `--speculative-draft-model-path`, `--speculative-num-steps`, `--speculative-num-draft-tokens` | `--enable-prefix-caching`, `--speculative-config` | usually via `--extra_llm_api_options` or a backend-specific flow |
| Dtype, quantization, loading | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code` | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code`, `--hf-token` | `--trust_remote_code`, `--tokenizer`; quantization and engine options are often outside `trtllm-serve serve` |

## Benchmark Client Notes

Both vLLM and TensorRT-LLM benchmark clients accepted these shared random-workload
flags in the H100 audit:

- `--dataset-name`
- `--random-input-len`
- `--random-output-len`
- `--num-prompts`
- `--request-rate`
- `--max-concurrency`
- `--save-result`
- `--result-dir`
- `--result-filename`
- `--ignore-eos`
- `--temperature`
- `--top-p`
- `--top-k`

TensorRT-LLM 1.0.0 additionally needs `--random-ids` for a fast synthetic random
run unless a ShareGPT download path is provided. vLLM 0.19.1 does not have that
flag.

## Search Guidance

- SGLang and vLLM both have broad runtime surfaces. Give both frameworks a real
  sweep over memory, batching, scheduler, CUDA graph/compile, and prefix cache
  options before comparing winners.
- TensorRT-LLM should still be searched, but its direct `trtllm-serve serve`
  surface is narrower. Search the exposed server flags first, then move deeper
  backend options into `--extra_llm_api_options` or an engine-build phase.
- Keep every translated flag in the artifact manifest. If a family is not
  represented for one framework, say why instead of filling the gap with an
  unrelated option.
