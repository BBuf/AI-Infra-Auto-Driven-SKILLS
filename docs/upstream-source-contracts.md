# Upstream source contracts

Source inspection: **2026-09-18**, snapshots frozen around 07:22 UTC. These are
source checks, not a new GPU benchmark or a promise about an installed wheel.
Record the actual image digest, package versions, framework commit and dependent
FlashInfer/CUDA/Triton versions for every run. Later commits may change dispatch.

| Source | Ref | Inspected commit |
|---|---|---|
| sglang | `main` | [`c46bf5e990bd`](https://github.com/sgl-project/sglang/commit/c46bf5e990bdd99e2c200214b04683022100e4df) |
| sglang-dsv41 | `dsv4.1` | [`f238b3cae10d`](https://github.com/sgl-project/sglang/commit/f238b3cae10db1575a632855fc69b2d1fe003142) |
| vllm | `main` | [`d5f0a6e829fa`](https://github.com/vllm-project/vllm/commit/d5f0a6e829faa69d1db289bf62b14dae136c02b2) |
| flashinfer | `main` | [`eb5f05be1f8e`](https://github.com/flashinfer-ai/flashinfer/commit/eb5f05be1f8e3ef8aa017a66dcbd5d80119b1095) |
| tensorrt-llm | `main` | [`dcc95a8bf527`](https://github.com/NVIDIA/TensorRT-LLM/commit/dcc95a8bf527583bed803606083c37585a08fc14) |
| tokenspeed | `main` | [`b41ea7d762ac`](https://github.com/lightseekorg/tokenspeed/commit/b41ea7d762ac176a63c16e8fcb7eaf52124015ca) |

## Profiler control and output

| Framework | Inspected contract | Consequence for the skill |
|---|---|---|
| SGLang | [ProfilerManager](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/managers/scheduler_components/profiler_manager.py) and [profile-v2](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/utils/profile_utils.py) | `/start_profile` carries output directory, steps, activities and stage options. Verify both actual emitted stages and nonzero GPU events. Model warmup and profiler warmup are separate. |
| vLLM | [ProfilerConfig](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/config/profiler.py) and [HTTP router](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/entrypoints/serve/profile/api_router.py) | Routes are attached only when a profiler is configured. HTTP start/stop have no per-request config body. Configure `torch_profiler_dir`, `max_iterations`, delay and warmup/active schedules at launch; a client's `num_steps` is not a universal engine step limit. CPU frontend and GPU-worker traces can coexist. |
| TensorRT-LLM | [StartProfileRequest](https://github.com/NVIDIA/TensorRT-LLM/blob/dcc95a8bf527583bed803606083c37585a08fc14/tensorrt_llm/serve/openai_protocol.py), [HTTP routes](https://github.com/NVIDIA/TensorRT-LLM/blob/dcc95a8bf527583bed803606083c37585a08fc14/tensorrt_llm/serve/openai_server.py), [PyExecutorProfileManager](https://github.com/NVIDIA/TensorRT-LLM/blob/dcc95a8bf527583bed803606083c37585a08fc14/tensorrt_llm/_torch/pyexecutor/profiling.py) | Runtime start accepts `output_dir`, `num_steps`, `start_step`, `activities`; native output is `trtllm-trace-<id>-rank-<rank>.json`. Profiling moved out of `py_executor.py`. No rank0-only source patch is needed. The inspected profiler still omits `with_stack`; missing Python attribution stays unknown, or use an explicitly recorded mapping build. |
| TokenSpeed | [worker profiling](https://github.com/lightseekorg/tokenspeed/blob/b41ea7d762ac176a63c16e8fcb7eaf52124015ca/python/tokenspeed/runtime/engine/request_handler.py), [control client](https://github.com/lightseekorg/tokenspeed/blob/b41ea7d762ac176a63c16e8fcb7eaf52124015ca/python/tokenspeed/runtime/engine/scheduler_control_client.py) and [proxy routes](https://github.com/lightseekorg/tokenspeed/blob/b41ea7d762ac176a63c16e8fcb7eaf52124015ca/python/tokenspeed/runtime/entrypoints/control_server.py) | Supports runtime start/stop and `output_dir`, `activities`, `profile_id`, `with_stack`, `record_shapes`. Torch Chrome traces and Proton's native artifacts are different formats; do not feed arbitrary Hatchet output to the Chrome parser. |

Read a trace from a shared mount or explicitly copy it back. A successful HTTP
response does not establish that a new GPU trace was written. Never silently
reuse an older file from the same directory. Preserve all ranks for a skew or
collective diagnosis; a rank0 compact view is a presentation artifact.

## Dispatch, defaults and source locations

- SGLang compiled kernels now live below `python/sglang/kernels/`: AOT Python
  bindings in `aot/python/sgl_kernel`, JIT CUDA in `jit/csrc`, Python/Triton ops
  in `ops`. Legacy traces can still contain `sgl_kernel` or earlier namespaces;
  normalize those names without claiming the old checkout path still exists.
- [SGLang main environ](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/environ.py) sets
  `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=False` after
  [#40105](https://github.com/sgl-project/sglang/pull/40105).
  [The pinned dsv4.1 branch](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/srt/environ.py)
  still sets it to `True`. This is a numerical behavior difference, not a
  cosmetic version bump. Do not transfer an accuracy result between these refs.
- [DSV4 model dispatch](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/srt/models/deepseek_v4.py)
  gates BF16 WO-A fast paths on V4.1, phase, token count, device, layout and
  exact tensor shapes. A source-level fused implementation does not establish
  that a particular target/draft path calls it.
- [vLLM pass ordering](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/compilation/passes/pass_manager.py)
  and [pass configuration](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/config/compilation.py) govern
  norm/activation/attention quantization and collective fusions. Inspect the
  backend, graph partition and dtype guards; a flag alone is not dispatch proof.
  CUDA C++ sources used by the inspected fusion families have moved into
  `csrc/libtorch_stable/`.
- [FlashInfer MXFP8 cluster split-K](https://github.com/flashinfer-ai/flashinfer/blob/eb5f05be1f8e3ef8aa017a66dcbd5d80119b1095/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py)
  supports small M up to 32, two/four K slices, `swap_ab=True`, R128C4 scales,
  and K divisibility by `128 * split_k_slices`. Peer CTAs reduce FP32 partials
  through distributed shared memory; no extra global reduction kernel is needed.
  [Tactic selection](https://github.com/flashinfer-ai/flashinfer/blob/eb5f05be1f8e3ef8aa017a66dcbd5d80119b1095/flashinfer/gemm/gemm_base.py) still
  determines whether this implementation is selected. This is not general
  NVFP4/BF16 or arbitrary-layout support.

## Memory sizing

[KV cache configurator](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/mem_cache/kv_cache_configurator.py)
reserves headroom `pre_model_load_memory * (1 - mem_fraction_static)`, subtracts
it and multimodal reservations from currently free memory, then accounts for
model-specific pools. It does **not** multiply post-weight free memory by the
fraction. [Post-capture resizing](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py)
can change capacity again. Use final rank-local pool/capacity logs; compressed
KV, Mamba state and DSPARK buffers are not ordinary dense-KV arithmetic.

## Maintaining this evidence

Read the caller and implementation at the same SHA; attach immutable links to
new conclusions. For an open PR also record head/base and diff scope. A merged
PR can be absent from another branch, or superseded by a later fix. Historical
model dossiers and GPU smoke tables retain their original dates; updating this
page does not re-audit all their PRs or rerun their models. Use
[DSV4.1 profiling lessons](../skills/llm-torch-profiler-analysis/references/dsv41-kernel-optimization.md)
for the September kernel work and
[benchmark validation](../skills/llm-serving-auto-benchmark/references/paired-validation.md)
for performance/accuracy experiments.
