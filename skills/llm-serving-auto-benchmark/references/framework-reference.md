# Framework Reference

Use this file when choosing native framework commands or translating tuning
knobs across SGLang, vLLM, TensorRT-LLM, and TokenSpeed. Always verify the
concrete CLI in the target container with `--help` before a long run.

## Native Entry Points

| Framework | Server | Benchmark | Notes |
| --- | --- | --- | --- |
| SGLang | `sglang serve` or `python -m sglang.launch_server` | `python -m sglang.benchmark.serving` | Current public cookbooks emit `sglang serve`. `launch_server` remains accepted. Use `benchmark.serving` for direct native or OpenAI-compatible endpoint checks. |
| vLLM | `vllm serve` | `vllm bench sweep serve` or `vllm bench serve` | Prefer `bench sweep serve` when sweeping server and benchmark parameter JSON files. |
| TensorRT-LLM | `trtllm-serve serve <model>` | TensorRT-LLM serving benchmark client or a common OpenAI-compatible client | This skill does not cover engine-backed serving or non-PyTorch server backends. |
| TokenSpeed | `tokenspeed serve` | `vllm bench serve --backend openai-chat` or a common OpenAI-compatible client | Use as a first-class baseline for agentic workloads. Some images may also expose the binary as `ts`; record the actual command. |

Common source docs:

- SGLang bench serving: <https://docs.sglang.io/docs/developer_guide/bench_serving>
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

python -m sglang.benchmark.serving \
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

vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --model <model> \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80
```

For profiler handoff, POST the control server (CONTROL_PORT defaults to serving
port + 1; override with --control-port), then drive the same workload:

```bash
curl -X POST "http://127.0.0.1:${CONTROL_PORT}/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/artifacts/tokenspeed_profile","num_steps":5,"activities":["CPU","GPU"],"with_stack":true,"profile_id":"ts-bench"}'
# Drive the same workload with the common client, then POST /stop_profile if needed.
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
| Parallelism | `--tp-size`, `--pp-size`, `--dp-size` (replicas), `--attn-dp-size`, `--attn-cp-size`, `--dcp-size`, `--moe-dp-size`, `--ep-size`, `--expert-parallel-size` | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size`, `--enable-expert-parallel` | `--tp_size`, `--pp_size`, `--ep_size`, `--gpus_per_node`, `--cluster_size` (deprecated, unsupported) | `--tensor-parallel-size`, `--attn-tp-size`, `--dense-tp-size`, `--moe-tp-size`, `--enable-expert-parallel`, data-parallel flags |
| Memory and KV cache | `--mem-fraction-static`, `--max-total-tokens`, `--kv-cache-dtype`, `--page-size`, `--cpu-offload-gb` | `--gpu-memory-utilization`, `--kv-cache-memory-bytes`, `--kv-cache-dtype`, `--block-size`, `--cpu-offload-gb` | `--kv_cache_free_gpu_memory_fraction`, plus `--max_num_tokens`, `--max_seq_len`, `--max_batch_size` | `--gpu-memory-utilization`, `--kv-cache-dtype`, `--max-total-tokens`, `--max-model-len`, `--max-prefill-tokens` |
| Batching and scheduler | `--max-running-requests`, `--schedule-policy`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--prefill-max-requests` | `--max-num-seqs`, `--max-num-batched-tokens`, `--enable-chunked-prefill`, `--long-prefill-token-threshold`, and DBO flags | `--max_batch_size`, `--max_num_tokens`, `--max_seq_len`; extra scheduler knobs may require `--extra_llm_api_options` | `--max-num-seqs`, `--chunked-prefill-size`, `--max-prefill-tokens`, `--max-total-tokens` |
| Attention/backend | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend`, `--sampling-backend` | `--attention-backend`, `--gdn-prefill-backend`, `--mm-encoder-attn-backend` | current main is PyTorch-only; `--backend` deprecated | `--attention-backend`, `--drafter-attention-backend`, `--moe-backend`, `--draft-moe-backend` |
| CUDA graph and compile | `--cuda-graph-backend-decode`, `--cuda-graph-backend-prefill`, `--cuda-graph-bs-decode`, `--cuda-graph-bs-prefill`, `--cuda-graph-max-bs-decode`, `--cuda-graph-max-bs-prefill`, `--cuda-graph-config`, `--cuda-graph-tc-compiler`, `--enable-torch-compile` | `--enforce-eager`, `--compilation-config`, `--cudagraph-capture-sizes`, `--max-cudagraph-capture-size` | use direct flags or `--extra_llm_api_options`; record resolved PyTorch config from logs | CUDA graph padding flags, runtime graph settings, and communication-fusion flags accepted by the target image |
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
and TensorRT-LLM `1.0.0`. The [source contracts](../../../docs/upstream-source-contracts.md) record the
2026-09-18 inspected source revisions. Treat these as source evidence,
not as a substitute for target-image `--help`. Since the prior refresh, vLLM PR
`#46735` changed Triton/NVFP4 MoE CUDA graph capture behavior, and
TensorRT-LLM PR `#11685` / `#15546` changed KV eviction and KV block-offset host
staging behavior. The final increment also includes vLLM `#42669` /
`#49982`, TensorRT-LLM `#16805` / `#16763`, and TokenSpeed `#821`; these widen
FA4/modeling coverage, correct disaggregated speculative accounting and
phase-1 graph cleanup, and document the Kimi K3 deployment contract
respectively. Record stale-image risk when these surfaces affect a row, but do
not treat source presence as benchmark validation.

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
