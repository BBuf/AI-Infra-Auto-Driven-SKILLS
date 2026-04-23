# SGLang Kimi K2 / K2 Thinking / K2.5 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `47c4b3825`，以及相关 merged、open、closed PR 的 patch 阅读结果整理。范围覆盖原有 `sglang-kimi-k2-k25-optimization` skill 涉及的主线，也补充了之后仍在推进的 Kimi K2.5 DeepEP、W4AFP8、AMD MXFP4 等方向。

阅读结论先放前面：截至 `47c4b3825`，Kimi K2 和 Kimi K2 Thinking 的常规 MoE 路由、Marlin W4A16 MoE、EP、PCG 已有主线支持；Kimi K2.5 已有独立多模态 wrapper、PP、DP ViT、Eagle3、PD disaggregation、EPLB 等运行时接口。Kimi K2 Thinking 的 `DeepEP + int4/Marlin` PR `#13789` 已关闭未合入；真正仍在推进的是 Kimi K2.5 W4A16 DeepEP low-latency PR `#22496`。

## 1. 时间线总览

| 创建日期   |     PR | 状态   | 主线             | 代码区域                                        | 作用                                                                        |
| ---------- | -----: | ------ | ---------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| 2025-07-14 |  #8021 | merged | Kimi K2          | `fused_moe_triton/configs`                      | 增加 H20-3e FP8 MoE tuning config。                                         |
| 2025-07-14 |  #8013 | merged | Kimi K2          | `sgl-kernel/csrc/gemm/dsv3_router_gemm_*`       | `dsv3_router_gemm` 支持 384 experts。                                       |
| 2025-07-15 |  #8047 | merged | Kimi K2          | `fused_moe_triton/configs`                      | 增加 H20 FP8 MoE tuning config。                                            |
| 2025-07-20 |  #8176 | merged | Kimi K2          | `fused_moe_triton/configs`                      | 增加 H200 TP16 Kimi K2 MoE config。                                         |
| 2025-07-20 |  #8178 | merged | Kimi K2          | `fused_moe_triton/configs`                      | 增加 B200 TP16 Kimi K2 MoE config。                                         |
| 2025-07-20 |  #8183 | merged | Kimi K2          | `fused_moe_triton/configs`                      | 修正 H200 Kimi K2 MoE config 的 expert/N 组合。                             |
| 2025-08-09 |  #9010 | merged | Kimi K2          | `fused_moe_triton/configs/triton_3_4_0`         | 增加 B200 新 Triton 版本 FP8 MoE config。                                   |
| 2025-11-12 | #13150 | merged | Kimi K2 Thinking | `python/sglang/srt/layers/moe/topk.py`          | 给 384 experts、单 expert group 的 biased top-k 加 torch.compile 优化路径。 |
| 2025-11-14 | #13287 | merged | Kimi K2 Thinking | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` | 新增 Kimi K2 专用 fused gate CUDA op。                                      |
| 2025-11-15 | #13332 | merged | Kimi K2 Thinking | `topk.py`                                       | 在 Kimi K2 Thinking 路由中接入 fused gate。                                 |
| 2025-11-16 | #13374 | merged | Kimi K2 Thinking | `kimi_k2_moe_fused_gate.cu`                     | 优化 fused gate kernel 的向量化 load 和 small-token 路径。                  |
| 2025-11-19 | #13587 | merged | Kimi K2 Thinking | `moe_align_block_size.py`                       | 删除 `sgl_moe_align_block_size` 中无意义的 padding kernel。                 |
| 2025-11-19 | #13596 | merged | Kimi K2 Thinking | `fused_marlin_moe.py`、quant method             | 避免 fake EP 下 Marlin MoE 的无用 `torch.zeros_`。                          |
| 2025-11-21 | #13725 | merged | Kimi K2 Thinking | `compressed_tensors_moe.py`                     | 给 Kimi K2 Thinking compressed-tensors MoE 增加 EP 支持。                   |
| 2025-11-23 | #13789 | closed | Kimi K2 Thinking | DeepEP + Marlin path                            | 尝试支持 K2 Thinking DeepEP，但因 illegal memory access 关闭未合入。        |
| 2025-12-14 | #15100 | merged | Kimi K2 Thinking | `fused_marlin_moe.py`、MoE runner               | 让 fused Marlin MoE 支持 piecewise CUDA graph。                             |
| 2025-12-17 | #15306 | merged | Kimi K2 Thinking | `kimi_k2_moe_fused_gate.cu`                     | 修复 PCG 下 warp illegal instruction。                                      |
| 2025-12-18 | #15347 | merged | Kimi K2 Thinking | `topk.py`                                       | 优先使用 FlashInfer `fused_topk_deepseek` 替代 Kimi fused gate。            |
| 2026-01-19 | #17325 | merged | Kimi K2 Thinking | `topk.py`                                       | 修复 biased grouped top-k 的 kernel 选择条件。                              |
| 2026-01-27 | #17789 | merged | Kimi K2.5        | `models/kimi_k25.py`、processor、parser         | 新增 Kimi K2.5 多模态模型支持。                                             |
| 2026-01-30 | #17991 | merged | Kimi K2.5        | `vision.py`、`kimi_k25.py`                      | 修复 VLM DP attention 的 double reduce。                                    |
| 2026-02-01 | #18064 | merged | Kimi K2.5        | `scheduler.py`                                  | 修复 K2.5 wrapper 下 MoE GEMM config 初始化。                               |
| 2026-02-06 | #18370 | merged | Kimi K2.5        | `modelopt_quant.py`、`kimi_k25.py`              | 修复 NVFP4 权重映射和 exclude list。                                        |
| 2026-02-08 | #18440 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 补齐 `quant_config` 保存。                                                  |
| 2026-02-08 | #18434 | merged | Kimi K2.5        | `deepseek_v2.py`、`kimi_k25.py`                 | 支持 pipeline parallel。                                                    |
| 2026-02-12 | #18689 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 增加 DP ViT encoder 支持。                                                  |
| 2026-02-23 | #19181 | merged | Kimi K2/K2.5     | `python/sglang/jit_kernel/moe_wna16_marlin.py`  | 将 Marlin MoE kernel 从 AOT 迁移到 JIT。                                    |
| 2026-02-24 | #19228 | merged | Kimi K2.5        | AMD tuning、`fused_moe_triton_config.py`        | 为 K2.5 int4 W4A16 在 AMD 上调 fused MoE config。                           |
| 2026-03-02 | #19689 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 支持 Eagle3 捕获层和 embed/head 接口。                                      |
| 2026-03-02 | #19703 | open   | Kimi K2 Thinking | `jit_kernel` fused gate                         | 将 `kimi_k2_moe_fused_gate` 迁移到 JIT，尚未合入。                          |
| 2026-03-05 | #19959 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 暴露 PP layer range，支持 PD disaggregation。                               |
| 2026-03-17 | #20747 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 修复 K2.5 piecewise CUDA graph wrapper 暴露面。                             |
| 2026-03-20 | #21004 | merged | Kimi K2.5        | `kimi_k25.py`                                   | 增加 EPLB rebalance 所需 routed expert weights 接口。                       |
| 2026-03-25 | #21391 | merged | Kimi K2.5        | `llama_eagle3.py`、test                         | 修复 DP attention + spec decoding 的 multimodal launch crash。              |
| 2026-03-31 | #21741 | open   | Kimi K2.5        | W4AFP8 MoE                                      | 通用 compressed-tensors W4AFP8 MoE 支持。                                   |
| 2026-04-06 | #22208 | open   | Kimi K2.5        | AMD Triton config                               | gfx950 small-M decode fused MoE tuning。                                    |
| 2026-04-10 | #22488 | open   | Kimi K2 Thinking | JIT fused gate                                  | 将 Kimi2 ungrouped fused gate 泛化到 GLM-5 256 experts。                    |
| 2026-04-10 | #22496 | open   | Kimi K2.5        | `deepep_moe_wna16_marlin_direct.py` 等          | K2.5 W4A16 DeepEP low-latency direct Marlin 路线。                          |
| 2026-04-14 | #22806 | open   | Kimi K2.5        | `quantization/w4afp8.py`                        | 新增 `KimiW4AFp8Config` 以加载 K2.5 W4AFP8。                                |
| 2026-04-16 | #22964 | open   | Kimi K2.5        | `KimiGPUProcessorWrapper`                       | 修复 CPU processor 输出 key 与 GPU 路径不一致。                             |
| 2026-04-19 | #23186 | open   | Kimi K2.5        | AMD MLA attention                               | 为 `amd/Kimi-K2.5-MXFP4` 增加 fused q/k RMSNorm BF16。                      |

## 2. Kimi K2 的第一阶段：384 experts 与 MoE tuning

Kimi K2 早期接入的主要矛盾不是模型 wrapper，而是 DeepSeek-V3 系 MoE 基础设施默认更熟悉 256 experts；Kimi K2 需要 384 experts，并且在 H20、H20-3e、H200、B200 上需要独立 fused MoE tuning config。

`#8013` 是这一阶段的核心代码 PR。它把 `dsv3_router_gemm` 的 expert 数从单一 256 扩展到 256/384 双形态：

- 在 `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` 等入口中增加 `DEFAULT_NUM_EXPERTS = 256`、`KIMI_K2_NUM_EXPERTS = 384`、`DEFAULT_HIDDEN_DIM = 7168` 等常量。
- runtime 根据 `mat_b.size(0)` 判断当前 expert 数，并用 `TORCH_CHECK` 明确只允许 256 或 384，避免静默走错模板实例。
- 为 token 数 `1..16`、输出 `float` / `bfloat16` 两类路径补齐 384 experts 模板实例化。
- benchmark `bench_dsv3_router_gemm.py` 和测试都扩展到 `num_experts in [256, 384]`，确保 Kimi K2 不只是能编译，而是覆盖 benchmark 和 UT。

`#8021`、`#8047`、`#8176`、`#8178`、`#8183`、`#9010` 则是 tuning config 的设备覆盖。它们新增或修正的 config 文件位于：

- `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/`
- `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/`

这些文件名编码了关键 shape，例如 `E=384` 或 `E=385`、`N=128/256`、`dtype=fp8_w8a8`、`block_shape=[128, 128]`、`device_name=NVIDIA_H20/H20-3e/H200/B200`。`E=385` 这种 config 反映了实际 MoE 路由或 shared expert 组合下的配置形态；后续调度器会按模型 config、quant config、设备名、Triton 版本去匹配这些 JSON。

## 3. Kimi K2 Thinking：top-k 路由、fused gate 与 Marlin MoE

`#13150` 首先优化了 Kimi K2 Thinking 的 biased top-k。Kimi K2 Thinking 的典型路由形态是 `num_experts == 384`、`num_expert_group == 1`、`topk` 较小。原本通用 grouped top-k 仍会保留 group masking、group score 等泛化逻辑。PR 在 `topk.py` 中新增 `kimi_k2_biased_topk_impl`：

- 直接计算 `scores.sigmoid() + correction_bias`。
- 对完整 384 experts 做 `torch.topk`，拿到 top-k expert ids。
- 通过 `torch.gather` 回取原始 sigmoid weights。
- 按需做 renormalization 和 routed scaling。
- 如果存在 logical-to-physical expert map，则把逻辑 expert id 映射为物理 expert id。
- 对 padding token mask 做过滤。
- 用 `@torch.compile` 固化这个专门路径，避免每次 decode 都走 Python 级泛化分支。

`#13287` 把上述路由进一步下沉为 CUDA op `sgl_kernel::kimi_k2_moe_fused_gate`。kernel 固定服务 Kimi K2 Thinking 的关键形态：

- `NUM_EXPERTS = 384`。
- `topk = 6`。
- `WARPS_PER_CTA = 6`。
- 初版 `VPT = 12`，面向每个 token 的 sigmoid、bias add、top-k、renorm、scaling 一次完成。
- 同时支持 small-token 和 large-token 的不同 launch 思路。
- Python wrapper 和 benchmark、test 也一起加入，测试基准是和 `kimi_k2_biased_topk_impl` 对齐。

`#13332` 在 `biased_grouped_topk_gpu` 中接入这个 kernel：当设备是 CUDA、expert 数为 384、只有一个 expert group 时，优先走 `kimi_k2_moe_fused_gate`，否则继续回退到通用路径。

`#13374` 对 fused gate 做第二轮 kernel 级优化：

- 将输入 scores 和 correction bias 明确收窄到 `float32` 路径，减少 dtype 泛化成本。
- 增加 `VEC_SIZE = 4` 的 `float4` 向量化 load。
- small-token kernel 中每个 token 用 384 个线程分别处理 384 experts。
- shared memory 保存 `selected_experts[8]`、`warp_experts`、`warp_maxs` 等中间结果。
- 减少 `__syncthreads()`，把 top-k 选择、renorm、输出写回放进更紧的 kernel。

`#13587` 删除 `sgl_moe_align_block_size` 的无用 padding kernel。这个修改看起来小，但在 MoE decode 小 batch 场景里，额外 kernel launch 和无意义 padding 会直接进入关键路径。

`#13596` 则针对 Kimi K2 Thinking 的 W4A16 Marlin MoE 做了 SGLang 侧封装 `fused_marlin_moe`。主要细节是：

- 通过 `moe_align_block_size` 整理 token/expert 对齐。
- 在 block size candidate `[8, 16, 32, 48, 64]` 中选择合适 `block_size_m`。
- 第一次调用 `moe_wna16_marlin_gemm` 做 gate/up projection。
- 调用 `silu_and_mul` 做激活融合。
- 第二次调用 `moe_wna16_marlin_gemm` 做 down projection。
- 最后 `moe_sum_reduce` 合并 top-k expert 输出。
- 原本 fake EP 路径会无条件清零中间 cache；PR 把 `torch.zeros_` 收窄到 `expert_map is not None` 才需要的场景，避免非真实 EP 下为不存在的 expert 输出付出清零成本。

当前 main 中这段 Marlin MoE 已经跟随 `#19181` 改为从 `python/sglang/jit_kernel/moe_wna16_marlin.py` 调 JIT kernel，而不是直接依赖 AOT sgl-kernel 符号。

## 4. Kimi K2 Thinking：EP、PCG 与路由 kernel 选择的演进

`#13725` 给 compressed-tensors MoE 路径补上 Kimi K2 Thinking 的 Expert Parallelism 支持。关键点是 compressed-tensors quant method 不再把 EP 信息当作假数据，而是把真实 `expert_map`、top-k ids、weights 和 runner metadata 传入 Marlin MoE。

`#15100` 让 fused Marlin MoE 支持 piecewise CUDA graph。PCG 对动态 shape、临时 tensor、custom op 以及 fake op 都很敏感；这个 PR 调整了 `fused_marlin_moe.py`、MoE runner 和 quant method 的边界，使这条路径可以被分段 CUDA graph 捕获。

`#15306` 是 PCG 后续修复，修掉 `kimi_k2_moe_fused_gate.cu` 中会触发 warp illegal instruction 的问题。这个问题出现在 fused gate 被 PCG 捕获、token shape/专家选择缓存更稳定之后，说明 kernel 内部对无效 expert 或 warp 选择状态的保护不够严。

`#15347` 改变了 Kimi K2 Thinking 的路由优先级：如果满足 FlashInfer `fused_topk_deepseek` 的条件，就优先使用这个 DSV3 优化 kernel，而不是 Kimi 专用 `moe_fused_gate`。当前 main 的 `biased_grouped_topk_gpu` 顺序大致是：

1. 如果 `fused_topk_deepseek` 可用、CUDA、expert 数是 2 的幂，并满足 group/topk 约束，则走 FlashInfer fused top-k。对 `num_expert_group == 1`，当前条件允许 `num_experts <= 384`。
2. 否则尝试通用 `moe_fused_gate`。
3. 再尝试 AITER 路径。
4. 再回退到 Kimi 384 experts 的 `kimi_k2_moe_fused_gate`。
5. 最后回退到 torch.compile 的 generic biased top-k。

`#17325` 修正了上述 kernel selection 的条件，避免在不满足 shape 或 group 限制时误选更快但不正确的路径。这个 PR 之后，Kimi fused gate 仍然存在，但它已经变成 fallback，而不是第一优先级。

`#19703` 仍 open，目标是把 `kimi_k2_moe_fused_gate` 从 AOT `sgl-kernel` 迁移到 `python/sglang/jit_kernel`。`#22488` 则进一步把 Kimi2 ungrouped fused gate 泛化到 GLM-5 的 256 experts。两者说明这条专用路由 kernel 未来更可能变成 JIT 管理的可变 expert 数 kernel，而不是 Kimi 独占 AOT 文件。

## 5. Kimi K2 Thinking DeepEP 状态：不是已经打通

`#13789` 的标题是 `[DeepEP Support] Support kimi-k2-thinking deepep`，但状态是 closed，未合入。它尝试的启动命令包括：

```bash
SGLANG_DEEPEP_BF16_DISPATCH=1 python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2-Thinking \
  --tp 8 --ep 4 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --trust-remote-code \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2
```

patch 和讨论里暴露的问题是在 `DeepEPMoE.forward_marlin_moe -> quant_method.apply_deepep_normal -> fused_marlin_moe` 一侧出现 illegal memory access。也就是说，Kimi K2 Thinking 的 `DeepEP + int4/Marlin` 不能因为 Marlin MoE JIT 化或普通 EP 支持而视为已主线打通。

和它不同，`#22496` 是 Kimi K2.5 W4A16 DeepEP low-latency 的新路线，仍 open。它没有沿用普通 `fused_marlin_moe` 的完整布局，而是新增：

- `deepep_moe_wna16_marlin_direct.py`
- `mask_silu_and_mul.py` / `.cuh`
- `marlin_direct_template.h`
- `kernel_direct.h`
- `marlin_tma_utils.h`
- 对 `moe_wna16_marlin.cuh`、`ep_moe/layer.py`、`token_dispatcher/deepep.py`、compressed-tensors quant method 的修改

这个方向的核心是给 compressed-tensors quant method 增加 `apply_deepep_normal` 和 `apply_deepep_ll`。`apply_deepep_ll` 要求 BF16 dispatch，并处理 DeepEP 输出的三维 `[E, M, K]` hidden states；它会构建和缓存 prefix/layout buffer，compact active hidden states，直接跑 Marlin gate/up 和 down 两次，中间用 `mask_silu_and_mul`，最后把结果 expand 回 DeepEP 的布局。它还加入了 `DEEPEP_LL_PROFILE_COMPUTE` profiling 日志。这个 PR 的目标是 Kimi K2.5 W4A16 DeepEP low-latency，不是已经关闭的 K2 Thinking DeepEP。

## 6. Kimi K2.5：多模态 wrapper 与运行时接口

`#17789` 是 Kimi K2.5 支持的起点。它新增 `python/sglang/srt/models/kimi_k25.py`，总体结构是：

- language model 复用 `DeepseekV3ForCausalLM`。
- vision tower 使用 MoonViT3d。
- projector 把 vision features 接到 language hidden size。
- `hf_to_sglang_mapper` 把 HF 权重里的 `language_model.layers.` 映射到 SGLang 内部的 `language_model.model.layers.`。
- processor 和 parser 接入 Kimi K2.5 的 multimodal 输入、reasoning parser 和 tool-call parser。
- `pad_input_ids` 处理 image token padding。
- `forward` 通过 `general_mm_embed_routine` 把 image embeddings 和 text embeddings 合并。

后续 K2.5 的大量 PR 都是在补齐 wrapper 对 SGLang runtime 的“透明性”：很多通用逻辑原本假设模型对象本身就是 CausalLM，而 K2.5 外层多了一层 multimodal wrapper，因此必须把底层 language model 的接口重新暴露出来。

`#18440` 补了 `self.quant_config`，否则 ModelOpt/NVFP4 等量化逻辑拿不到 wrapper 上的 quant config。`#18370` 进一步修正 NVFP4 的权重名映射和 exclude list，使量化模块知道哪些名字要穿过 `language_model` wrapper。`#18064` 则修复 scheduler 初始化 MoE GEMM config 时没有从 K2.5 `text_config` 里取 MoE 形状的问题。

`#18434` 增加 PP 支持。它让 K2.5 wrapper 能向底层 `DeepseekV3ForCausalLM` 传递 `pp_proxy_tensors`，并处理 pipeline stage 的 forward 输出。`#19959` 继续暴露 `start_layer` 和 `end_layer`，用于 PD disaggregation 等需要知道当前 PP shard 覆盖层范围的逻辑。

`#18689` 增加 DP ViT。当前 main 的 `KimiK25ForConditionalGeneration` 会读取 `get_global_server_args().mm_enable_dp_encoder`，把 `use_data_parallel` 传给 vision tower；`get_image_feature` 中如果启用 DP encoder，会走 `run_dp_sharded_mrope_vision_model`，让多模态 encoder 在 DP 维度切分执行。

`#17991` 修复 VLM DP attention double reduce，避免视觉侧 DP attention 已经 reduce 后又在上层重复 reduce。`#21391` 修复 DP attention + speculative decoding 的 launch crash：当 Eagle/spec decode 扩展 multimodal batch 时，不能重新 embed 完整 multimodal prefix，而是要复用 `forward_batch.mm_input_embeds`，只追加最后 token 的 embedding。

`#19689` 为 K2.5 增加 Eagle3 接口：`set_eagle3_layers_to_capture`、`get_embed_and_head`、`set_embed_and_head`。`#20747` 则让 wrapper 设置 `self.model = self.language_model.model`，修复 piecewise CUDA graph 对底层 model surface 的假设。

`#21004` 增加 EPLB rebalance 所需接口：当前 main 中 K2.5 的 `routed_experts_weights_of_layer` property 会返回底层 language model 的 `_routed_experts_weights_of_layer.value`。这样 EPLB 可以跨 wrapper 读取每层 routed expert 权重。

## 7. Kimi K2.5 量化与平台优化

`#19181` 把 Marlin MoE kernel 迁移到 JIT。新增 `python/sglang/jit_kernel/moe_wna16_marlin.py`，通过 `_jit_moe_wna16_marlin_module` 编译并导出 `moe_wna16_marlin_gemm`。测试覆盖：

- `m = 1` 和 `m = 123`。
- `n = 128` 和 `n = 1024`。
- `fp16` / `bf16`。
- act-order 与非 act-order。
- `uint4` / `uint4b8` 权重布局。
- JIT 和旧 AOT 结果 bitwise 对齐。

这对 Kimi 很重要，因为 Kimi K2 Thinking / K2.5 的 W4A16 MoE 会走 Marlin MoE。但它不是 DeepEP 打通的充分条件；DeepEP 还要解决 token dispatch layout、active token compact、expert buffer 和 direct Marlin 调用。

`#19228` 是 AMD 方向的 Kimi K2.5 fused MoE tuning。它让 config 读取逻辑能穿过 K2.5 `text_config`，从 quant config 里识别 int4 W4A16 的 group size 和 block shape，并为 `dtype=int4_w4a16` 生成正确 config 文件名。对于 int4 packed layout，`N` 需要按 shard intermediate size 再做 packed 折算。

`#22208` 仍 open，继续针对 AMD `gfx950` 的 small-M decode fused MoE config 做 tuning。`#23186` 也是 AMD 方向：在 MLA absorb prepare 里，如果启用 AITER 且 dtype 是 BF16，就用 `fused_qk_rmsnorm_bf16` 同时融合 q_a 和 kv_a RMSNorm，目标模型是 `amd/Kimi-K2.5-MXFP4`。

`#21741` 和 `#22806` 是 W4AFP8 方向。`#21741` 是通用 compressed-tensors W4AFP8 MoE 支持，引入 FP8 activation scale、CUTLASS W4A8 MoE 等底层能力。`#22806` 则新增 Kimi 专用 `KimiW4AFp8Config`：

- quant method 名称为 `kimi_w4afp8`。
- 解析 quant config 里的所有关键字段。
- 区分 `ignored_layers` 和 `unquantized_layers`：前者跳过 W4 但仍可能 FP8，后者如 `lm_head` 完全不量化。
- 归一化 `model.` 前缀。
- 对普通 `LinearBase` 返回 `Fp8LinearMethod` 或 `UnquantizedLinearMethod`。
- 对 `FusedMoE` 返回 `W4AFp8MoEMethod`。
- 补齐 HF 标准 `gate_proj/down_proj/up_proj` 的 expert input scale 映射。

`#22964` 修复 processor 的小坑：GPU processor `_call` 当前会输出 `image_grid_thw`，而 CPU `_cpu_call` 在某些路径输出 `grid_thws`。open PR 会把 CPU 路径也映射成 `image_grid_thw`，避免后续 multimodal feature packing 出现 key mismatch。

## 8. 当前 main 的代码形态

截至 `47c4b3825`，Kimi 相关主线可以概括为以下形态：

- `topk.py` 中 Kimi K2 Thinking 的 384 experts 路由已经不是单一路线，而是 FlashInfer `fused_topk_deepseek`、通用 `moe_fused_gate`、AITER、Kimi fused gate、torch.compile generic 的多级选择。
- `fused_marlin_moe.py` 使用 JIT `moe_wna16_marlin_gemm`，并保留 `expert_map is not None` 下的 EP 清零逻辑。
- `kimi_k25.py` 是 K2.5 的核心 wrapper，负责 language model、vision tower、projector、processor、DP ViT、PP range、Eagle3、PCG、EPLB 等运行时接口。
- K2.5 的量化和平台优化还在快速推进：NVFP4 已有主线修复，W4AFP8、K2.5 W4A16 DeepEP low-latency、AMD MXFP4 fused q/k RMSNorm 仍是 open PR 方向。
