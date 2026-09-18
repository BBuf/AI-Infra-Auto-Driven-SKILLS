# Profiler source map

Use the immutable [source contracts](../../../docs/upstream-source-contracts.md)
for the inspected revisions, HTTP payloads, output naming and version limits.
Read the installed source before changing a deployment.

| Framework | Entry / worker / attribution source |
|---|---|
| SGLang | [profiler CLI](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/profiler.py); [benchmark client](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/bench_serving.py); [scheduler profiler](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/managers/scheduler_components/profiler_manager.py); [stage profiler](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/utils/profile_utils.py); [distributed merge](https://github.com/sgl-project/sglang/blob/c46bf5e990bdd99e2c200214b04683022100e4df/python/sglang/srt/utils/profile_merger.py) |
| vLLM | [profiler config](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/config/profiler.py); [HTTP routes](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/entrypoints/serve/profile/api_router.py); [compile fusion registration](https://github.com/vllm-project/vllm/blob/d5f0a6e829faa69d1db289bf62b14dae136c02b2/vllm/compilation/passes/pass_manager.py) |
| TensorRT-LLM | [request schema](https://github.com/NVIDIA/TensorRT-LLM/blob/dcc95a8bf527583bed803606083c37585a08fc14/tensorrt_llm/serve/openai_protocol.py); [profiling manager](https://github.com/NVIDIA/TensorRT-LLM/blob/dcc95a8bf527583bed803606083c37585a08fc14/tensorrt_llm/_torch/pyexecutor/profiling.py) |
| TokenSpeed | [control routes](https://github.com/lightseekorg/tokenspeed/blob/b41ea7d762ac176a63c16e8fcb7eaf52124015ca/python/tokenspeed/runtime/entrypoints/control_server.py); [profile lifecycle](https://github.com/lightseekorg/tokenspeed/blob/b41ea7d762ac176a63c16e8fcb7eaf52124015ca/python/tokenspeed/runtime/engine/request_handler.py) |

DSV4.1 kernel-to-model mapping and PR scope are in
[dsv41-kernel-optimization.md](dsv41-kernel-optimization.md). Older `sgl_kernel`
Python frames remain useful evidence, but current bindings are under
`python/sglang/kernels/aot/python/sgl_kernel`; JIT and Triton implementations are
under `python/sglang/kernels/jit` and `python/sglang/kernels/ops` respectively.
