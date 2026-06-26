# Framework Reference

Use this file when choosing native framework commands or translating tuning
knobs across SGLang, vLLM, TensorRT-LLM, and TokenSpeed. Always verify the
concrete CLI in the target container with `--help` before a long run.

## Native Entry Points

| Framework | Server | Benchmark | Notes |
| --- | --- | --- | --- |
| SGLang | `python -m sglang.launch_server` | `python -m sglang.auto_benchmark` or `python -m sglang.bench_serving` | Use `auto_benchmark` when available for server-flag search. Use `bench_serving` for direct native or OpenAI-compatible endpoint checks. |
| vLLM | `vllm serve` | `vllm bench sweep serve` or `vllm bench serve` | Prefer `bench sweep serve` when sweeping server and benchmark parameter JSON files. |
| TensorRT-LLM | `trtllm-serve serve --backend pytorch` | TensorRT-LLM serving benchmark client or a common OpenAI-compatible client | This skill does not cover engine-backed serving or non-PyTorch server backends. |
| TokenSpeed | `tokenspeed serve` | `tokenspeed bench serve` or a common OpenAI-compatible client | Use as a first-class baseline for agentic workloads. Some images may also expose the binary as `ts`; record the actual command. |

Common source docs:

- SGLang bench serving: <https://docs.sglang.ai/developer_guide/bench_serving.html>
- vLLM benchmark sweeps: <https://docs.vllm.ai/en/latest/benchmarking/sweeps/>
- vLLM `bench sweep serve`: <https://docs.vllm.ai/en/latest/cli/bench/sweep/serve.html>
- TensorRT-LLM `trtllm-serve`: <https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html>
- TensorRT-LLM deployment guide: <https://nvidia.github.io/TensorRT-LLM/deployment-guide/index.html>
- TokenSpeed repository and docs: <https://github.com/lightseekorg/tokenspeed>

## Command Templates

### SGLang

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --tp-size <tp> \
  --port 30000

python -m sglang.bench_serving \
  --backend sglang-oai \
  --host 127.0.0.1 \
  --port 30000 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8
```

Use `--backend sglang` for SGLang-native `/generate` checks. Use
`--backend sglang-oai` when comparing against vLLM or TensorRT-LLM through an
OpenAI-compatible path.

### vLLM

```bash
vllm serve <model> \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size <tp> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80
```

### TensorRT-LLM

```bash
trtllm-serve serve <model> \
  --backend pytorch \
  --tp_size <tp> \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --config <extra-llm-api-options.yaml> \
  --host 0.0.0.0 \
  --port 8000
```

Benchmark the OpenAI-compatible endpoint with the TensorRT-LLM serving benchmark
client or the same OpenAI-compatible client used for the other frameworks. Keep
server backend choice fixed to `pytorch`. Recheck `--backend`, extra options,
and KV-cache memory aliases on the target `trtllm-serve serve --help`; this
skill still rejects non-PyTorch server candidates even when the CLI exposes
other backend choices.

### TokenSpeed

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

tokenspeed bench serve \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80
```

For profiler handoff, TokenSpeed's benchmark can drive the same server
profiler endpoints:

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

For TokenSpeed-native production-style runs, keep the server command as
`tokenspeed serve <model>` and then add only flags confirmed by the target
`tokenspeed serve --help`. Current docs show Kimi-style production knobs such as
`--kv-cache-dtype fp8`, `--quantization nvfp4`,
`--enable-expert-parallel`, `--chunked-prefill-size`,
`--attention-backend trtllm_mla`, and `--moe-backend flashinfer_trtllm`; do not
copy those backend choices to unrelated models without a smoke run.

## Knob Family Mapping

Do not copy flag names across frameworks. Compare knob families, then translate
to the target CLI.

| Family | SGLang | vLLM | TensorRT-LLM | TokenSpeed |
| --- | --- | --- | --- | --- |
| Parallelism | `--tp-size`, `--pp-size`, `--dp-size`, `--ep-size`, `--expert-parallel-size` | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` | `--tp_size`, `--pp_size`, `--ep_size`, `--gpus_per_node`, `--cluster_size` | `--tensor-parallel-size`, `--attn-tp-size`, `--dense-tp-size`, `--moe-tp-size`, `--enable-expert-parallel`, data-parallel flags |
| Memory and KV cache | `--mem-fraction-static`, `--max-total-tokens`, `--kv-cache-dtype`, `--page-size`, `--cpu-offload-gb` | `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`, `--block-size`, `--cpu-offload-gb` | `--kv_cache_free_gpu_memory_fraction`, plus `--max_num_tokens`, `--max_seq_len`, `--max_batch_size` | `--gpu-memory-utilization`, `--kv-cache-dtype`, `--max-total-tokens`, `--max-model-len`, `--max-prefill-tokens` |
| Batching and scheduler | `--max-running-requests`, `--schedule-policy`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--prefill-max-requests` | `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, partial-prefill and DBO flags | `--max_batch_size`, `--max_num_tokens`, `--max_seq_len`; extra scheduler knobs may require `--extra_llm_api_options` | `--max-num-seqs`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--max-total-tokens` |
| Attention/backend | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`, `--sampling-backend` | `--attention-backend`, `--gdn-prefill-backend`, `--mm-encoder-attn-backend` | `--backend pytorch` is fixed; do not search backend choice | `--attention-backend`, `--drafter-attention-backend`, `--moe-backend`, `--draft-moe-backend` |
| CUDA graph and compile | `--disable-cuda-graph`, `--cuda-graph-bs`, `--cuda-graph-max-bs`, `--disable-piecewise-cuda-graph`, `--enable-torch-compile` | `--enforce-eager`, `--compilation-config`, `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | use direct flags or `--extra_llm_api_options`; record resolved PyTorch config from logs | CUDA graph padding flags, runtime graph settings, and communication-fusion flags accepted by the target image |
| Prefix/speculative | `--disable-radix-cache`, `--disable-chunked-prefix-cache`, speculative decoding flags | `--enable-prefix-caching`, `--speculative-config` | only use PyTorch-backend options accepted by the target image | `--enable-prefix-caching`, `--speculative-config`, `--speculative-algorithm`, `--speculative-num-steps`, `--speculative-num-draft-tokens` |
| Dtype, quantization, loading | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code` | `--dtype`, `--quantization`, `--load-format`, `--model-loader-extra-config`, `--trust-remote-code`, `--hf-token` | `--trust_remote_code`, `--tokenizer`; engine build and non-PyTorch quantization flows are out of scope | `--dtype`, `--quantization`, `--trust-remote-code`, tokenizer/model loader flags accepted by `tokenspeed serve --help` |

## Version Rules

Framework CLIs move quickly. For every real run:

1. Record the framework package version, git commit, image tag, and help files.
2. Validate concrete flags with
   `scripts/validate_cookbook_configs.py --help-dir <artifact-help-dir>`.
3. Move renamed or removed flags out of the run plan before benchmarking.
4. Record which frameworks were model-smoked and which only passed preflight.

Historical validation from April 2026 used SGLang `0.5.10rc0`, vLLM `0.19.1`,
and TensorRT-LLM `1.0.0`. A source refresh on 2026-06-26 checked SGLang
`b91348071e6ca21f9357fd8b1adc83a18781df30`, vLLM
`c2507fb2937aa8c8e74bea15719d04fb6090befe`, TensorRT-LLM
`4164b932c6c8a14d1be85d0fd62e44b7d0171980`, and TokenSpeed
`5aedf69d6b476baa65571011de6ea60fd5a238a8`. Treat these as source evidence,
not as a substitute for target-image `--help`.
