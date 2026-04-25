# vLLM Mixtral Quark / INT4-FP8 MoE 支持与 PR 历史

本文记录 vLLM 中与 Mixtral Quark / INT4-FP8 MoE 相关的模型支持、关键 PR、以及仍需持续跟踪的风险点。

- 状态: 当前 mainline 仅部分支持，或只有相邻架构已落地

## 核心结论

- vLLM has rich Mixtral MoE support, but not every Quark-branded checkpoint path is called out by name.
- The closest production evidence is the Mixtral fused-MoE, FP8, ModelOpt, and EPLB work already merged.

## 主要代码面

- `vllm/vllm/model_executor/models/mixtral.py`
- `vllm/vllm/model_executor/layers/fused_moe/layer.py`

## 已合入 PR

- [#2011](https://github.com/vllm-project/vllm/pull/2011) `Mixtral 8x7B support`：Initial Mixtral model-family support.
- [#2090](https://github.com/vllm-project/vllm/pull/2090) `Optimize Mixtral with expert parallelism`：Added early expert-parallel scaling instead of pure TP execution.
- [#2542](https://github.com/vllm-project/vllm/pull/2542) `Fused MOE for Mixtral`：Brought fused-MoE kernels into the Mixtral serving path.
- [#4527](https://github.com/vllm-project/vllm/pull/4527) `Support MoE FP8 checkpoints for Mixtral`：Added the first serious FP8 checkpoint path for Mixtral MoE.
- [#15961](https://github.com/vllm-project/vllm/pull/15961) `Support ModelOpt quantization of Mixtral model`：Extended the family to NVIDIA ModelOpt quantization flows.
- [#22842](https://github.com/vllm-project/vllm/pull/22842) `Support EPLB for Mixtral Model`：Added expert-parallel load balancing to the Mixtral family.

## Open PR 雷达

- 暂无固定 open PR；需要在声称新支持前重新搜索。

## 配套 skill

- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-mixtral-quark-int4fp8-moe-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `vllm-project/vllm` 的 Pull Request API 和文件级 patch 重新审计 `Mixtral Quark INT4-FP8 MoE`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2023-12-11 | [#2011](https://github.com/vllm-project/vllm/pull/2011) | merged | Mixtral 8x7B support | model wrapper, scheduler/runtime | `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md` |
| 2023-12-13 | [#2090](https://github.com/vllm-project/vllm/pull/2090) | merged | Mixtral expert parallelism | model wrapper, scheduler/runtime, docs/config | `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile` |
| 2024-01-22 | [#2542](https://github.com/vllm-project/vllm/pull/2542) | merged | Fused MOE for Mixtral | model wrapper, MoE/router, kernel, scheduler/runtime | `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp` |
| 2024-05-01 | [#4527](https://github.com/vllm-project/vllm/pull/4527) | merged | [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations) | model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks | `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py` |
| 2025-04-02 | [#15961](https://github.com/vllm-project/vllm/pull/15961) | merged | Add support to modelopt quantization of Mixtral model | model wrapper, quantization, scheduler/runtime | `vllm/model_executor/models/mixtral_quant.py` |
| 2025-08-13 | [#22842](https://github.com/vllm-project/vllm/pull/22842) | merged | [EPLB] Support EPLB for Mixtral Model | model wrapper, scheduler/runtime | `vllm/model_executor/models/mixtral.py` |

### 逐 PR 代码 diff 阅读记录

### PR #2011 - Mixtral 8x7B support

- 链接：https://github.com/vllm-project/vllm/pull/2011
- 状态/时间：`merged`，created 2023-12-11, merged 2023-12-11；作者 `pierrestock`。
- 代码 diff 已读范围：`4` 个文件，`+538/-0`；代码面：model wrapper, scheduler/runtime；关键词：attention, cache, config, cuda, expert, kv, moe, quant, spec, topk。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral.py` added +534/-0 (534 lines); hunk: +# coding=utf-8; 符号: promote_scalar, MixtralAttention, __init__, forward
  - `vllm/model_executor/models/__init__.py` modified +2/-0 (2 lines); hunk: from vllm.model_executor.models.internlm import InternLMForCausalLM; "PhiForCausalLM",
  - `README.md` modified +1/-0 (1 lines); hunk: vLLM seamlessly supports many Hugging Face models, including the following archi
  - `vllm/model_executor/model_loader.py` modified +1/-0 (1 lines); hunk: "LlamaForCausalLM": LlamaForCausalLM,; 符号: has
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md`；patch 关键词为 attention, cache, config, cuda, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral.py`, `vllm/model_executor/models/__init__.py`, `README.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #2090 - Mixtral expert parallelism

- 链接：https://github.com/vllm-project/vllm/pull/2090
- 状态/时间：`merged`，created 2023-12-13, merged 2023-12-14；作者 `Yard1`。
- 代码 diff 已读范围：`6` 个文件，`+221/-334`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：doc, attention, cache, config, test, cuda, expert, flash, kv, moe。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral.py` modified +207/-307 (514 lines); hunk: from torch import nn; KVCache = Tuple[torch.Tensor, torch.Tensor]; 符号: promote_scalar, MixtralMLP, __init__, forward
  - `vllm/config.py` modified +9/-7 (16 lines); hunk: def _verify_load_format(self) -> None:; 符号: _verify_load_format
  - `Dockerfile` modified +1/-13 (14 lines); hunk: ENV NVCC_THREADS=$nvcc_threads; FROM vllm-base AS vllm-openai
  - `README.md` modified +0/-4 (4 lines); hunk: Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/get
  - `vllm/model_executor/models/__init__.py` modified +3/-1 (4 lines); hunk: }
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile`；patch 关键词为 doc, attention, cache, config, test, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral.py`, `vllm/config.py`, `Dockerfile` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #2542 - Fused MOE for Mixtral

- 链接：https://github.com/vllm-project/vllm/pull/2542
- 状态/时间：`merged`，created 2024-01-22, merged 2024-01-30；作者 `pcmoritz`。
- 代码 diff 已读范围：`4` 个文件，`+115/-109`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime；关键词：expert, moe, cache, cuda, topk, attention, config, kv, quant, router。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral.py` modified +104/-96 (200 lines); hunk: """Inference-only Mixtral model."""; from vllm.model_executor.input_metadata import InputMetadata; 符号: MixtralMLP, MixtralMoE, __init__, forward
  - `csrc/ops.h` modified +7/-9 (16 lines); hunk: void gptq_shuffle(; std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(
  - `csrc/pybind.cpp` modified +3/-3 (6 lines); hunk: PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  - `csrc/moe_align_block_size_kernels.cu` modified +1/-1 (2 lines); hunk: void moe_align_block_size(
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp`；patch 关键词为 expert, moe, cache, cuda, topk, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral.py`, `csrc/ops.h`, `csrc/pybind.cpp` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4527 - [Kernel] Support MoE Fp8 Checkpoints for Mixtral (Static Weights with Dynamic/Static Activations)

- 链接：https://github.com/vllm-project/vllm/pull/4527
- 状态/时间：`merged`，created 2024-05-01, merged 2024-05-04；作者 `mgoin`。
- 代码 diff 已读范围：`2` 个文件，`+122/-53`；代码面：model wrapper, MoE/router, kernel, scheduler/runtime, tests/benchmarks；关键词：config, expert, moe, attention, cuda, fp8, quant, router, test。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral.py` modified +120/-51 (171 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, weight_loader, weight_loader
  - `tests/kernels/test_moe.py` modified +2/-2 (4 lines); hunk: def test_mixtral_moe(dtype: torch.dtype):; 符号: test_mixtral_moe
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py`；patch 关键词为 config, expert, moe, attention, cuda, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral.py`, `tests/kernels/test_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15961 - Add support to modelopt quantization of Mixtral model

- 链接：https://github.com/vllm-project/vllm/pull/15961
- 状态/时间：`merged`，created 2025-04-02, merged 2025-04-09；作者 `yueshen2016`。
- 代码 diff 已读范围：`1` 个文件，`+7/-1`；代码面：model wrapper, quantization, scheduler/runtime；关键词：fp8, kv, quant。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral_quant.py` modified +7/-1 (8 lines); hunk: from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler; def load_weights(self, weights: Iterable[Tuple[str,; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral_quant.py`；patch 关键词为 fp8, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral_quant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22842 - [EPLB] Support EPLB for Mixtral Model

- 链接：https://github.com/vllm-project/vllm/pull/22842
- 状态/时间：`merged`，created 2025-08-13, merged 2025-09-17；作者 `rouchenzi`。
- 代码 diff 已读范围：`1` 个文件，`+137/-23`；代码面：model wrapper, scheduler/runtime；关键词：attention, cache, config, expert, kv, lora, moe, quant, spec。
- 代码 diff 细节：
  - `vllm/model_executor/models/mixtral.py` modified +137/-23 (160 lines); hunk: # See the License for the specific language governing permissions and; from vllm.attention import Attention; 符号: __init__, __init__, forward, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `vllm/model_executor/models/mixtral.py`；patch 关键词为 attention, cache, config, expert, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `vllm/model_executor/models/mixtral.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：6；open PR 数：0。
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
