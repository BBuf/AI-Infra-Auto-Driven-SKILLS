# vLLM Mixtral Quark / INT4-FP8 MoE PR History

Evidence snapshot:

- vLLM mainline checked around `0f7be0f2f76814f80f9091220a5fbbb53912ad00`
- Support status: partially supported or only adjacent architectures landed on current mainline
- Scope: Mixtral MoE, expert parallelism, FP8 / ModelOpt quantization, and EPLB in vLLM, which together form the nearest equivalent to Quark INT4-FP8 Mixtral serving.

## Landed PRs

### PR #2011 - Mixtral 8x7B support

- Link: https://github.com/vllm-project/vllm/pull/2011
- Why it mattered: Initial Mixtral model-family support.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #2090 - Optimize Mixtral with expert parallelism

- Link: https://github.com/vllm-project/vllm/pull/2090
- Why it mattered: Added early expert-parallel scaling instead of pure TP execution.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #2542 - Fused MOE for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2542
- Why it mattered: Brought fused-MoE kernels into the Mixtral serving path.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #4527 - Support MoE FP8 checkpoints for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/4527
- Why it mattered: Added the first serious FP8 checkpoint path for Mixtral MoE.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #15961 - Support ModelOpt quantization of Mixtral model

- Link: https://github.com/vllm-project/vllm/pull/15961
- Why it mattered: Extended the family to NVIDIA ModelOpt quantization flows.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

### PR #22842 - Support EPLB for Mixtral Model

- Link: https://github.com/vllm-project/vllm/pull/22842
- Why it mattered: Added expert-parallel load balancing to the Mixtral family.
- Runtime path: vllm/vllm/model_executor/models/mixtral.py, vllm/vllm/model_executor/layers/fused_moe/layer.py
- Validation / risk: re-check this PR if you touch the same loader, parser, quantization, or multimodal surface.

## Open PR Radar

- No specific open PR is pinned here; rerun GitHub PR search before asserting new support.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# VLLM Mixtral Quark INT4-FP8 MoE PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2023-12-11 | [#2011](https://github.com/vllm-project/vllm/pull/2011) | merged | Mixtral 8x7B support | model wrapper, scheduler/runtime | `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md` |
| 2023-12-13 | [#2090](https://github.com/vllm-project/vllm/pull/2090) | merged | Mixtral expert parallelism | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile` |
| 2024-01-22 | [#2542](https://github.com/vllm-project/vllm/pull/2542) | merged | Fused MOE for Mixtral | model wrapper, MoE/router, kernel, scheduler/runtime | `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp` |
| 2024-05-01 | [#4527](https://github.com/vllm-project/vllm/pull/4527) | merged | [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations) | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py` |
| 2025-04-02 | [#15961](https://github.com/vllm-project/vllm/pull/15961) | merged | Add support to modelopt quantization of Mixtral model | model wrapper, quantization, scheduler/runtime | `vllm/model_executor/models/mixtral_quant.py` |
| 2025-08-13 | [#22842](https://github.com/vllm-project/vllm/pull/22842) | merged | [EPLB] Support EPLB for Mixtral Model | model wrapper, scheduler/runtime | `vllm/model_executor/models/mixtral.py` |

## Diff Cards

### PR #2011 - Mixtral 8x7B support

- Link: https://github.com/vllm-project/vllm/pull/2011
- Status/date: `merged`, created 2023-12-11, merged 2023-12-11; author `pierrestock`.
- Diff scope read: `4` files, `+538/-0`; areas: model wrapper, scheduler/runtime; keywords: attention, cache, config, cuda, expert, kv, moe, quant, spec, topk.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunks: +# coding=utf-8; symbols: promote_scalar, MixtralAttention, __init__, forward
  - `vllm/model_executor/models/__init__.py` modified +2/-0 (2 lines); hunks: from vllm.model_executor.models.internlm import InternLMForCausalLM; "PhiForCausalLM",
  - `README.md` modified +1/-0 (1 lines); hunks: vLLM seamlessly supports many Hugging Face models, including the following archi
  - `vllm/model_executor/model_loader.py` modified +1/-0 (1 lines); hunks: "LlamaForCausalLM": LlamaForCausalLM,; symbols: has
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md`; keywords observed in patches: attention, cache, config, cuda, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #2090 - Mixtral expert parallelism

- Link: https://github.com/vllm-project/vllm/pull/2090
- Status/date: `merged`, created 2023-12-13, merged 2023-12-14; author `Yard1`.
- Diff scope read: `6` files, `+221/-334`; areas: model wrapper, scheduler/runtime, docs/config; keywords: doc, attention, cache, config, test, cuda, expert, flash, kv, moe.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunks: from torch import nn; KVCache = Tuple[torch.Tensor, torch.Tensor]; symbols: promote_scalar, MixtralMLP, __init__, forward
  - `vllm/config.py` modified +9/-7 (16 lines); hunks: def _verify_load_format(self) -> None:; symbols: _verify_load_format
  - `Dockerfile` modified +1/-13 (14 lines); hunks: ENV NVCC_THREADS=$nvcc_threads; FROM vllm-base AS vllm-openai
  - `README.md` modified +0/-4 (4 lines); hunks: Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/get
  - `vllm/model_executor/models/__init__.py` modified +3/-1 (4 lines); hunks: }
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile`; keywords observed in patches: doc, attention, cache, config, test, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #2542 - Fused MOE for Mixtral

- Link: https://github.com/vllm-project/vllm/pull/2542
- Status/date: `merged`, created 2024-01-22, merged 2024-01-30; author `pcmoritz`.
- Diff scope read: `4` files, `+115/-109`; areas: model wrapper, MoE/router, kernel, scheduler/runtime; keywords: expert, moe, cache, cuda, topk, attention, config, kv, quant, router.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunks: """Inference-only Mixtral model."""; from vllm.model_executor.input_metadata import InputMetadata; symbols: MixtralMLP, MixtralMoE, __init__, forward
  - `csrc/ops.h` modified +7/-9 (16 lines); hunks: void gptq_shuffle(; std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
  - `csrc/pybind.cpp` modified +3/-3 (6 lines); hunks: PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  - `csrc/moe_align_block_size_kernels.cu` modified +1/-1 (2 lines); hunks: void moe_align_block_size(
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp`; keywords observed in patches: expert, moe, cache, cuda, topk, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #4527 - [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)

- Link: https://github.com/vllm-project/vllm/pull/4527
- Status/date: `merged`, created 2024-05-01, merged 2024-05-04; author `mgoin`.
- Diff scope read: `2` files, `+122/-53`; areas: model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks; keywords: config, expert, moe, attention, cuda, fp8, quant, router, test.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, weight_loader, weight_loader
  - `tests/kernels/test_moe.py` modified +2/-2 (4 lines); hunks: def test_mixtral_moe(dtype: torch.dtype):; symbols: test_mixtral_moe
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py`; keywords observed in patches: config, expert, moe, attention, cuda, fp8. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15961 - Add support to modelopt quantization of Mixtral model

- Link: https://github.com/vllm-project/vllm/pull/15961
- Status/date: `merged`, created 2025-04-02, merged 2025-04-09; author `yueshen2016`.
- Diff scope read: `1` files, `+7/-1`; areas: model wrapper, quantization, scheduler/runtime; keywords: fp8, kv, quant.
- Code diff details:
  - `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunks: from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler; def load_weights(self, weights: Iterable[Tuple[str,; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral_quant.py`; keywords observed in patches: fp8, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral_quant.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22842 - [EPLB] Support EPLB for Mixtral Model

- Link: https://github.com/vllm-project/vllm/pull/22842
- Status/date: `merged`, created 2025-08-13, merged 2025-09-17; author `rouchenzi`.
- Diff scope read: `1` files, `+137/-23`; areas: model wrapper, scheduler/runtime; keywords: attention, cache, config, expert, kv, lora, moe, quant, spec.
- Code diff details:
  - `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunks: # See the License for the specific language governing permissions and; from vllm.attention import Attention; symbols: __init__, __init__, forward, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/mixtral.py`; keywords observed in patches: attention, cache, config, expert, kv, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/mixtral.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
