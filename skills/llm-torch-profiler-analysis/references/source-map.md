# Profiler source map

Use the immutable [source contracts](../../../docs/upstream-source-contracts.md)
for the inspected revisions, HTTP payloads, output naming and version limits.
Read the installed source before changing a deployment.

| Framework | Entry / worker / attribution source |
|---|---|
| SGLang | [profiler CLI](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/profiler.py); [benchmark client](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/benchmark/serving.py); [scheduler profiler](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/managers/scheduler_components/profiler_manager.py); [stage profiler](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/observability/profile_utils.py); [distributed merge](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/python/sglang/srt/utils/profile_merger.py) |
| vLLM | [profiler config](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/config/profiler.py); [HTTP routes](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/entrypoints/serve/profile/api_router.py); [worker profiler](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/profiler/wrapper.py); [worker naming](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/v1/worker/gpu_worker.py); [compile fusion registration](https://github.com/vllm-project/vllm/blob/b022bb7a4873a47cdcdf474edf3bd499c221f40d/vllm/compilation/passes/pass_manager.py) |
| TensorRT-LLM | [request schema](https://github.com/NVIDIA/TensorRT-LLM/blob/b433473ba0d0d49764f4d2fb3d45c82716b946dc/tensorrt_llm/serve/openai_protocol.py); [HTTP routes](https://github.com/NVIDIA/TensorRT-LLM/blob/b433473ba0d0d49764f4d2fb3d45c82716b946dc/tensorrt_llm/serve/openai_server.py); [profiling manager](https://github.com/NVIDIA/TensorRT-LLM/blob/b433473ba0d0d49764f4d2fb3d45c82716b946dc/tensorrt_llm/_torch/pyexecutor/profiling.py) |
| TokenSpeed | [Rust control dispatch](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/tokenspeed-engine/crates/ts-control/src/calls.rs); [profile lifecycle](https://github.com/lightseekorg/tokenspeed/blob/7b21034e72bb45d1edef3d3b5483b50b58c4c492/python/tokenspeed/runtime/engine/profiler.py) |

Kernel attribution follows the
[general dispatch principles](heuristics.md#establish-dispatch-and-numerical-contracts)
and source families in [the fusion catalog](fuse-overlap-catalog.md). Older `sgl_kernel`
Python frames remain useful evidence, but current bindings are under
`python/sglang/kernels/aot/python/sgl_kernel`; JIT and Triton implementations are
under `python/sglang/kernels/jit` and `python/sglang/kernels/ops` respectively.

SGLang stage capture uses profiler v1 by default; opt-in v2 is stage-auto only.
Current [Rust ingress](https://github.com/sgl-project/sglang/blob/c24285bde97c3e6424cd5d8cb35788889034d8ea/rust/sglang-server/src/api_server/http/info.rs) supports
HTTP profile control; older builds may lack it. Canonical offline commands are
`sglang.benchmark.one_batch` and `sglang.benchmark.offline_throughput`; their
old aliases were removed. Cake and KDA kernels also live under
`python/sglang/kernels/{cake_kernels,kda_kernels}`.
