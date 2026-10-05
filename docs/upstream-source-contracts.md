# Upstream source contracts

Source inspection: **2026-10-05**. Framework source heads are frozen below. These are
source checks, not a new GPU benchmark or a promise about an installed wheel.
Record the actual image digest, package versions, framework commit and dependent
FlashInfer/CUDA/Triton versions for every run. Later commits may change dispatch.

| Source | Ref | Inspected commit |
|---|---|---|
| sglang | `main` | [`b1bbd74f287f`](https://github.com/sgl-project/sglang/commit/b1bbd74f287f13ed1276b0403a01ebb55c597e93) |
| sglang-dsv41 | `dsv4.1` (historical, deleted) | [`f238b3cae10d`](https://github.com/sgl-project/sglang/commit/f238b3cae10db1575a632855fc69b2d1fe003142) |
| vllm | `main` | [`0c16eee3f1ff`](https://github.com/vllm-project/vllm/commit/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a) |
| flashinfer | `main` | [`724e238e7dec`](https://github.com/flashinfer-ai/flashinfer/commit/724e238e7deca2d6ffd5668d63ae95c428414f59) |
| tensorrt-llm | `main` | [`bb367fc8c1ad`](https://github.com/NVIDIA/TensorRT-LLM/commit/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60) |
| tokenspeed | `main` | [`6fa10840d5c3`](https://github.com/lightseekorg/tokenspeed/commit/6fa10840d5c3c23065f60428ad264fba60fa04ae) |

## Fusion and storage evidence

Apply the [general dispatch and numerical principles](../skills/llm-torch-profiler-analysis/references/heuristics.md#establish-dispatch-and-numerical-contracts)
across models. The fusion catalog carries caller eligibility and format
examples; detailed model PR reviews remain in the model-history dossiers with
their original review dates.

- [Five integration PR dossiers](../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#reviewed-kernel-integrations-2026-09-22)
  cover DeepGEMM sparse candidate logits, DSA top-k v2, dual raw/page output,
  the DeepSelect BF16 consumer and FlashMLA FP8/FP4 KV layouts. Full diffs and
  current callers were inspected separately; obsolete opt-in flags are identified.
- New-format support is not the default: main still declares
  `SGLANG_DSV4_KV_LAYOUT=v4`. FP8 528 B and FP4 288 B count payload plus scales
  per stored token before page padding; they are not whole-model KV capacity.
- [mHC fusion eligibility](../skills/llm-torch-profiler-analysis/references/fuse-overlap-catalog.md#residual-mixing-and-collective-epilogues)
  records exact row ranges, backend/parallelism guards, BF16 boundaries and
  which epilogues include RMSNorm or quantization.
- #39704 and #39957 are now merged into main; #39941 was folded into #39704.
  The [model history](../model-pr-optimization-history/sglang/deepseek-v4/README.en.md#historical-profiling-and-optimization-evidence)
  retains historical experiment context and distinguishes inspected revisions.

## Profiler control and output

| Framework | Inspected contract | Consequence for the skill |
|---|---|---|
| SGLang | [ProfilerManager](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/managers/scheduler_components/profiler_manager.py) and [opt-in profile-v2](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/utils/profile_utils.py) | `/start_profile` carries output directory, steps, activities and stage options. v1 `SchedulerProfilerManager` is default; `SGLANG_PROFILE_V2=1` supports stage-auto only: `profile_by_stage=true`, no start step, merge or stop. Rust ingress has no profiler routes. Stack and shapes default true unless disabled. Filenames are `[prefix-]<profile_id>-TP-<tp>[-DP-<d>][-PP-<p>][-EP-<e>][-EXTEND|-DECODE].trace.json.gz`; `DP` appears when `dp_size*attn_dp_size>1`. Use a labeled `profile_id` instead of a prefix for merging. Verify emitted stages and nonzero GPU events. Model warmup and profiler warmup are separate. |
| vLLM | [ProfilerConfig](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/config/profiler.py) and [HTTP router](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/entrypoints/serve/profile/api_router.py) | Routes are attached only when a profiler is configured. HTTP start/stop have no per-request config body. Configure `torch_profiler_dir`, `max_iterations`, delay and warmup/active schedules at launch; a client's `num_steps` is not a universal engine step limit. CPU frontend and GPU-worker traces can coexist. `torch_profiler_activities` selects activities; CUDA-only drops step annotations and stacks. `active_iterations` needs a warmup/wait schedule; use `max_iterations` plus `ignore_frontend=true` for a bounded worker window. Repeated rounds need [#57460](https://github.com/vllm-project/vllm/pull/57460) (merged 2026-09-21). Worker files use `[prefix_]dp<d>_pp<p>_tp<t>_dcp<c>_ep<e>_rank<g>.<ts>.pt.trace.json.gz`; frontend files use `<host>_<pid>.async_llm.<ts>.pt.trace.json.gz`. |
| TensorRT-LLM | [StartProfileRequest](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/serve/openai_protocol.py), [HTTP routes](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/serve/openai_server.py), [PyExecutorProfileManager](https://github.com/NVIDIA/TensorRT-LLM/blob/bb367fc8c1adf6e2c28c88cb1a8b46e1742a9d60/tensorrt_llm/_torch/pyexecutor/profiling.py) | Runtime start accepts `output_dir`, `num_steps`, `start_step`, `activities`; native output is `trtllm-trace-<id>-rank-<rank>.json`. Profiling moved out of `py_executor.py`. No rank0-only source patch is needed. The inspected profiler still omits `with_stack`; missing Python attribution stays unknown, or use an explicitly recorded mapping build. |
| TokenSpeed | [worker profiling](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/runtime/engine/request_handler.py), [control client](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/runtime/engine/scheduler_control_client.py) and [proxy routes](https://github.com/lightseekorg/tokenspeed/blob/6fa10840d5c3c23065f60428ad264fba60fa04ae/python/tokenspeed/runtime/entrypoints/control_server.py) | Supports runtime start/stop and `output_dir`, `activities`, `profile_id`, `with_stack`, `record_shapes`. Torch Chrome traces and Proton's native artifacts are different formats; do not feed arbitrary Hatchet output to the Chrome parser. Activities include CPU, GPU, MEM, CUDA_PROFILER, PROTON, VIZTRACER and EXPERT_LOAD (the last writes `.expert-load.pt`). Torch files use `<profile_id>-[DP<d>-]TP<t>[-EXTEND|-DECODE].trace.json.gz`. `TOKENSPEED_CUPTI_GRAPH_WARMUP=1` can leave zero GPU events. JSON-to-ProfileReq forwarding lives in external pinned wheels and was not audited here. |

Read a trace from a shared mount or explicitly copy it back. A successful HTTP
response does not establish that a new GPU trace was written. Never silently
reuse an older file from the same directory. Preserve all ranks for a skew or
collective diagnosis; a rank0 compact view is a presentation artifact.

## Dispatch, defaults and source locations

- SGLang compiled kernels now live below `python/sglang/kernels/`: AOT Python
  bindings in `aot/python/sgl_kernel`, JIT CUDA in `jit/csrc`, Python/Triton ops
  in `ops`. Legacy traces can still contain `sgl_kernel` or earlier namespaces;
  normalize those names without claiming the old checkout path still exists.
- [SGLang main environ](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/environ.py) sets
  `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=False` after
  [#40105](https://github.com/sgl-project/sglang/pull/40105).
  [The historical dsv4.1 head (branch since removed)](https://github.com/sgl-project/sglang/blob/f238b3cae10db1575a632855fc69b2d1fe003142/python/sglang/srt/environ.py)
  set it to `True`. DSV4.1 now ships from main, where it is `False`. This is a numerical behavior difference, not a
  cosmetic version bump. Do not transfer an accuracy result between these refs.
- [DSV4 model dispatch](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/models/deepseek_v4.py)
  gates BF16 WO-A fast paths on V4.1, phase, token count, device, layout and
  exact tensor shapes. A source-level fused implementation does not establish
  that a particular target/draft path calls it.
- [vLLM pass ordering](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/compilation/passes/pass_manager.py)
  and [pass configuration](https://github.com/vllm-project/vllm/blob/0c16eee3f1ff777298cc894c3eeb85f3880c6d6a/vllm/config/compilation.py) govern
  norm/activation/attention quantization and collective fusions. Inspect the
  backend, graph partition and dtype guards; a flag alone is not dispatch proof.
  CUDA C++ sources used by the inspected fusion families have moved into
  `csrc/libtorch_stable/`.
- [FlashInfer block-scaled cluster split-K](https://github.com/flashinfer-ai/flashinfer/blob/724e238e7deca2d6ffd5668d63ae95c428414f59/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm100_splitk.py)
  supports small M up to 32, two/four K slices, `swap_ab=True`, R128C4 scales,
  and K divisibility by `128 * split_k_slices`. Peer CTAs reduce FP32 partials
  through distributed shared memory; no extra global reduction kernel is needed.
  [Tactic selection](https://github.com/flashinfer-ai/flashinfer/blob/724e238e7deca2d6ffd5668d63ae95c428414f59/flashinfer/gemm/gemm_base.py) still
  determines whether this implementation is selected. At this FlashInfer main revision the block-scaled kernel serves MXFP8
  and NVFP4 `mm_fp4(backend="cute-dsl")` (since
  [#5609](https://github.com/flashinfer-ai/flashinfer/pull/5609), M ≤ 32).
  BF16 uses separate split-K implementations. Layout and tactic guards still
  decide dispatch. This 0.7.1 source head is newer than the 0.7.0.post1
  pins in SGLang and vLLM; verify the installed package before recommending it.

## Memory sizing

[KV cache configurator](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/mem_cache/kv_cache_configurator.py)
reserves headroom `pre_model_load_memory * (1 - mem_fraction_static)`, subtracts
it and multimodal reservations from currently free memory, then accounts for
model-specific pools. It does **not** multiply post-weight free memory by the
fraction. [Post-capture resizing](https://github.com/sgl-project/sglang/blob/b1bbd74f287f13ed1276b0403a01ebb55c597e93/python/sglang/srt/model_executor/model_runner_components/kv_pool_runtime.py)
can change capacity again. It is opt-in (`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=False`),
CUDA-only and skipped for draft workers; gates exclude MLA, DCP, memory saver,
incompatible Mooncake/EFA pools, insufficient graph coverage, DSV4 and
MiniMax-sparse. Unified-memory hybrid-SWA is allowed since #41961; mamba-ish
and EAGLE/standalone/DFlash speculation remain excluded there. Active mamba-ish
sizing keeps at least the pre-capture activation reserve. `Post-capture KV sizing:`
logs supersede earlier allocation lines. Use final rank-local pool/capacity logs; compressed
KV, Mamba state and DSPARK buffers are not ordinary dense-KV arithmetic.

## Maintaining this evidence

Read the caller and implementation at the same SHA; attach immutable links to
new conclusions. For an open PR also record head/base and diff scope. A merged
PR can be absent from another branch, or superseded by a later fix. Historical
model dossiers and GPU smoke tables retain their original dates; updating this
page does not re-audit all their PRs or rerun their models. Use
[general dispatch principles](../skills/llm-torch-profiler-analysis/references/heuristics.md#establish-dispatch-and-numerical-contracts)
for interpretation and
[benchmark validation](../skills/llm-serving-auto-benchmark/references/paired-validation.md)
for performance/accuracy experiments.
