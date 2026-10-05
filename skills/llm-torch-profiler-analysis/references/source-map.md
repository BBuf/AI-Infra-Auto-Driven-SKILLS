# Profiler source map

Use the immutable [source contracts](../../../docs/upstream-source-contracts.md)
for the inspected revisions, HTTP payloads, output naming and version limits.
Read the installed source before changing a deployment.

| Framework | Entry / worker / attribution source |
|---|---|
| SGLang | [profiler CLI](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/profiler.py); [benchmark client](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/benchmark/serving.py); [scheduler profiler](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/managers/scheduler_components/profiler_manager.py); [stage profiler](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/utils/profile_utils.py); [distributed merge](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/utils/profile_merger.py) |
| vLLM | [profiler config](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/config/profiler.py); [HTTP routes](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/entrypoints/serve/profile/api_router.py); [worker profiler](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/profiler/wrapper.py); [worker naming](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/v1/worker/gpu_worker.py); [compile fusion registration](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/compilation/passes/pass_manager.py) |
| TensorRT-LLM | [request schema](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/serve/openai_protocol.py); [HTTP routes](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/serve/openai_server.py); [profiling manager](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/_torch/pyexecutor/profiling.py) |
| TokenSpeed | [control routes](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/runtime/entrypoints/control_server.py); [profile lifecycle](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/runtime/engine/request_handler.py) |

Kernel attribution follows the
[general dispatch principles](heuristics.md#establish-dispatch-and-numerical-contracts)
and source families in [the fusion catalog](fuse-overlap-catalog.md). Older `sgl_kernel`
Python frames remain useful evidence, but current bindings are under
`python/sglang/kernels/aot/python/sgl_kernel`; JIT and Triton implementations are
under `python/sglang/kernels/jit` and `python/sglang/kernels/ops` respectively.

SGLang stage capture uses profiler v1 by default; opt-in v2 is stage-auto only.
Rust ingress has no HTTP profiler routes. Canonical offline commands are
`sglang.benchmark.one_batch` and `sglang.benchmark.offline_throughput`; their
old aliases were removed. Cake and KDA kernels also live under
`python/sglang/kernels/{cake_kernels,kda_kernels}`.
