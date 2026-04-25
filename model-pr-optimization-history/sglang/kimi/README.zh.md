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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Kimi K2 / K2.5`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-07-14 | [#8013](https://github.com/sgl-project/sglang/pull/8013) | merged | [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384 | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` |
| 2025-07-14 | [#8021](https://github.com/sgl-project/sglang/pull/8021) | merged | perf: add kimi k2 fused_moe tuning config for h30_3e | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-15 | [#8043](https://github.com/sgl-project/sglang/pull/8043) | merged | feat(function call): complete utility method for KimiK2Detector and enhance documentation | tests/benchmarks | `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py` |
| 2025-07-15 | [#8047](https://github.com/sgl-project/sglang/pull/8047) | merged | H20 tune config for Kimi | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8176](https://github.com/sgl-project/sglang/pull/8176) | merged | feat: add h200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8178](https://github.com/sgl-project/sglang/pull/8178) | merged | feat: add b200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-07-20 | [#8183](https://github.com/sgl-project/sglang/pull/8183) | merged | feat: add h200 tp 16 kimi k2 moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-08-09 | [#9010](https://github.com/sgl-project/sglang/pull/9010) | merged | [perf] add kimi-k2 b200 fused moe config | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-08-25 | [#9606](https://github.com/sgl-project/sglang/pull/9606) | merged | Fix kimi k2 function calling format | tests/benchmarks | `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-09-18 | [#10612](https://github.com/sgl-project/sglang/pull/10612) | merged | Replace the Kimi-K2 generated tool call idx with history tool call count | tests/benchmarks | `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-09-26 | [#10972](https://github.com/sgl-project/sglang/pull/10972) | merged | fix: KimiK2Detector Improve tool call ID parsing with regex | misc | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2025-11-06 | [#12759](https://github.com/sgl-project/sglang/pull/12759) | merged | [Ascend] support Kimi-K2-Thinking | model wrapper, MoE/router, quantization, scheduler/runtime | `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-12 | [#13150](https://github.com/sgl-project/sglang/pull/13150) | merged | Opt kimi_k2_thinking biased topk module | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-14 | [#13287](https://github.com/sgl-project/sglang/pull/13287) | merged | [opt kimi k2 1 / n] Add kimi k2 moe fused gate | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` |
| 2025-11-15 | [#13332](https://github.com/sgl-project/sglang/pull/13332) | merged | [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-16 | [#13374](https://github.com/sgl-project/sglang/pull/13374) | merged | [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel | MoE/router, kernel | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2025-11-18 | [#13466](https://github.com/sgl-project/sglang/pull/13466) | merged | [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking) | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-11-19 | [#13587](https://github.com/sgl-project/sglang/pull/13587) | merged | [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size | MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` |
| 2025-11-19 | [#13596](https://github.com/sgl-project/sglang/pull/13596) | merged | [kimi k2 thinking] Avoid useless torch.zeros_ | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-11-21 | [#13725](https://github.com/sgl-project/sglang/pull/13725) | merged | Add Expert Parallelism (EP) support for kimi-k2-thinking | MoE/router, quantization | `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-11-23 | [#13789](https://github.com/sgl-project/sglang/pull/13789) | closed | [DeepEP Support] Support kimi-k2-thinking deepep | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu` |
| 2025-12-14 | [#15100](https://github.com/sgl-project/sglang/pull/15100) | merged | Support piecewise cuda graph for fused marlin moe | MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks | `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` |
| 2025-12-17 | [#15306](https://github.com/sgl-project/sglang/pull/15306) | merged | Fix warp illegal instruction in kimi k2 thinking PCG | MoE/router, kernel | `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | MoE/router, kernel, tests/benchmarks | `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2026-01-19 | [#17325](https://github.com/sgl-project/sglang/pull/17325) | merged | Fix kernel selection in biased_grouped_topk_gpu | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2026-01-21 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | quantization, tests/benchmarks | `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` |
| 2026-01-23 | [#17656](https://github.com/sgl-project/sglang/pull/17656) | merged | [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases | tests/benchmarks | `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml` |
| 2026-01-27 | [#17789](https://github.com/sgl-project/sglang/pull/17789) | merged | Support Kimi-K2.5 model | model wrapper, attention/backend, multimodal/processor, docs/config | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-01-30 | [#17991](https://github.com/sgl-project/sglang/pull/17991) | merged | Fix: Avoid Double Reduce in VLM DP Attention | model wrapper, attention/backend, multimodal/processor, tests/benchmarks | `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-01 | [#18064](https://github.com/sgl-project/sglang/pull/18064) | merged | fix kimi k2.5's moe gemm config init | scheduler/runtime | `python/sglang/srt/managers/scheduler.py` |
| 2026-02-04 | [#18269](https://github.com/sgl-project/sglang/pull/18269) | merged | [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test | model wrapper, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml` |
| 2026-02-06 | [#18370](https://github.com/sgl-project/sglang/pull/18370) | merged | [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list | model wrapper, quantization | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-08 | [#18434](https://github.com/sgl-project/sglang/pull/18434) | merged | [Fix] Kimi K2.5 support pp | model wrapper | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-08 | [#18440](https://github.com/sgl-project/sglang/pull/18440) | merged | [Kimi-K2.5] Fix missing `quant_config` in `KimiK25` | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-12 | [#18689](https://github.com/sgl-project/sglang/pull/18689) | merged | Add DP ViT support for Kimi K2.5 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-02-21 | [#19120](https://github.com/sgl-project/sglang/pull/19120) | merged | fix KimiK2Detector regex patterns with re.DOTALL | misc | `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-02-23 | [#19181](https://github.com/sgl-project/sglang/pull/19181) | merged | [Kernel Slimming] Migrate marlin moe kernel to JIT | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` |
| 2026-02-24 | [#19228](https://github.com/sgl-project/sglang/pull/19228) | merged | [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning | MoE/router, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` |
| 2026-02-25 | [#19331](https://github.com/sgl-project/sglang/pull/19331) | merged | [NPU] support Kimi-K2.5 on NPU | model wrapper, MoE/router, quantization | `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2026-02-28 | [#19552](https://github.com/sgl-project/sglang/pull/19552) | merged | [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection | tests/benchmarks | `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` |
| 2026-03-02 | [#19689](https://github.com/sgl-project/sglang/pull/19689) | merged | feat: support Kimi K2.5 for Eagle3 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-02 | [#19703](https://github.com/sgl-project/sglang/pull/19703) | open | [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT | MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` |
| 2026-03-03 | [#19802](https://github.com/sgl-project/sglang/pull/19802) | merged | [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2 | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py` |
| 2026-03-05 | [#19959](https://github.com/sgl-project/sglang/pull/19959) | merged | Fix Kimi K2.5 PP layer range exposure for PD disaggregation | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-17 | [#20747](https://github.com/sgl-project/sglang/pull/20747) | merged | fix piecewise cuda graph support for Kimi-K2.5 model | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-20 | [#21004](https://github.com/sgl-project/sglang/pull/21004) | merged | [Fix] Add EPLB rebalance support for Kimi K2.5 | model wrapper | `python/sglang/srt/models/kimi_k25.py` |
| 2026-03-25 | [#21391](https://github.com/sgl-project/sglang/pull/21391) | merged | Fix Kimi K2.5 dp attention+ spec decoding launch crash | model wrapper, tests/benchmarks | `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py` |
| 2026-03-31 | [#21741](https://github.com/sgl-project/sglang/pull/21741) | open | [1/N] feat: support compressed-tensors w4afp8 MoE | MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` |
| 2026-04-06 | [#22208](https://github.com/sgl-project/sglang/pull/22208) | open | [AMD] Optimize fused MoE kernel config for small-M decode on gfx950 | MoE/router, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` |
| 2026-04-10 | [#22488](https://github.com/sgl-project/sglang/pull/22488) | open | Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation | MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py` |
| 2026-04-10 | [#22496](https://github.com/sgl-project/sglang/pull/22496) | open | [Feature] kimi k25 w4a16 support deepep low latency | MoE/router, quantization, kernel | `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` |
| 2026-04-14 | [#22806](https://github.com/sgl-project/sglang/pull/22806) | open | feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-04-16 | [#22964](https://github.com/sgl-project/sglang/pull/22964) | open | [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output | multimodal/processor | `python/sglang/srt/multimodal/processors/kimi_k25.py` |
| 2026-04-19 | [#23186](https://github.com/sgl-project/sglang/pull/23186) | merged | [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4 | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8013 - [Kimi K2] dsv3_router_gemm supports NUM_EXPERTS == 384

- 链接：https://github.com/sgl-project/sglang/pull/8013
- 状态/时间：`merged`，created 2025-07-14, merged 2025-08-01；作者 `panpan0000`。
- 代码 diff 已读范围：`5` 个文件，`+188/-30`；代码面：MoE/router, kernel, tests/benchmarks；关键词：expert, router, cuda, benchmark, quant, test。
- 代码 diff 细节：
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu` modified +50/-16 (66 lines); hunk: #include "cuda_runtime.h"; void dsv3_router_gemm(; 符号: int, int, int, int
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu` modified +50/-0 (50 lines); hunk: void invokeRouterGemmBf16Output(__nv_bfloat16* output, T const* mat_a, T const*; template void invokeRouterGemmBf16Output<__nv_bfloat16, 15, 256, 7168>(; 符号: void, void, void, void
  - `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` modified +50/-0 (50 lines); hunk: void invokeRouterGemmFloatOutput(float* output, T const* mat_a, T const* mat_b,; template void invokeRouterGemmFloatOutput<__nv_bfloat16, 15, 256, 7168>(; 符号: void, void, void, void
  - `sgl-kernel/benchmark/bench_dsv3_router_gemm.py` modified +36/-12 (48 lines); hunk: x_vals=[i + 1 for i in range(16)],; def tflops(t_ms):; 符号: benchmark_bf16_output, runner, runner, tflops
  - `sgl-kernel/tests/test_dsv3_router_gemm.py` modified +2/-2 (4 lines); hunk: @pytest.mark.parametrize("num_tokens", [i + 1 for i in range(16)]); 符号: test_dsv3_router_gemm, test_dsv3_router_gemm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu`；patch 关键词为 expert, router, cuda, benchmark, quant, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/gemm/dsv3_router_gemm_entry.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_bf16_out.cu`, `sgl-kernel/csrc/gemm/dsv3_router_gemm_float_out.cu` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8021 - perf: add kimi k2 fused_moe tuning config for h30_3e

- 链接：https://github.com/sgl-project/sglang/pull/8021
- 状态/时间：`merged`，created 2025-07-14, merged 2025-07-14；作者 `GaoYusong`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8043 - feat(function call): complete utility method for KimiK2Detector and enhance documentation

- 链接：https://github.com/sgl-project/sglang/pull/8043
- 状态/时间：`merged`，created 2025-07-15, merged 2025-07-24；作者 `CatherineSue`。
- 代码 diff 已读范围：`8` 个文件，`+205/-56`；代码面：tests/benchmarks；关键词：spec, kv, config, doc, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/base_format_detector.py` modified +70/-12 (82 lines); hunk: class BaseFormatDetector(ABC):; def parse_streaming_increment(; 符号: BaseFormatDetector, providing, __init__, parse_base_json
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +41/-16 (57 lines); hunk: class KimiK2Detector(BaseFormatDetector):; def parse_streaming_increment(; 符号: KimiK2Detector, __init__, parse_streaming_increment, parse_streaming_increment
  - `python/sglang/srt/function_call/deepseekv3_detector.py` modified +25/-10 (35 lines); hunk: class DeepSeekV3Detector(BaseFormatDetector):; def parse_streaming_increment(; 符号: DeepSeekV3Detector, __init__, parse_streaming_increment, parse_streaming_increment
  - `test/srt/test_function_call_parser.py` modified +28/-0 (28 lines); hunk: def setUp(self):; def test_deepseekv3_detector_ebnf(self):; 符号: setUp, test_pythonic_detector_ebnf, test_deepseekv3_detector_ebnf, test_kimik2_detector_ebnf
  - `python/sglang/srt/function_call/pythonic_detector.py` modified +12/-9 (21 lines); hunk: class PythonicDetector(BaseFormatDetector):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; 符号: PythonicDetector, __init__, detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py`；patch 关键词为 spec, kv, config, doc, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/base_format_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`, `python/sglang/srt/function_call/deepseekv3_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8047 - H20 tune config for Kimi

- 链接：https://github.com/sgl-project/sglang/pull/8047
- 状态/时间：`merged`，created 2025-07-15, merged 2025-07-15；作者 `artetaout`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8176 - feat: add h200 tp 16 kimi k2 moe config

- 链接：https://github.com/sgl-project/sglang/pull/8176
- 状态/时间：`merged`，created 2025-07-20, merged 2025-07-20；作者 `zhyncs`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8178 - feat: add b200 tp 16 kimi k2 moe config

- 链接：https://github.com/sgl-project/sglang/pull/8178
- 状态/时间：`merged`，created 2025-07-20, merged 2025-07-20；作者 `zhyncs`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=385,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8183 - feat: add h200 tp 16 kimi k2 moe config

- 链接：https://github.com/sgl-project/sglang/pull/8183
- 状态/时间：`merged`，created 2025-07-20, merged 2025-07-20；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_3_1/E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9010 - [perf] add kimi-k2 b200 fused moe config

- 链接：https://github.com/sgl-project/sglang/pull/9010
- 状态/时间：`merged`，created 2025-08-09, merged 2025-08-09；作者 `Alcanderian`。
- 代码 diff 已读范围：`1` 个文件，`+146/-0`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9606 - Fix kimi k2 function calling format

- 链接：https://github.com/sgl-project/sglang/pull/9606
- 状态/时间：`merged`，created 2025-08-25, merged 2025-08-26；作者 `XiaotongJiang`。
- 代码 diff 已读范围：`2` 个文件，`+117/-9`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +96/-0 (96 lines); hunk: python -m unittest discover -s tests -p "test_*unit.py" -v; async def test_unstreamed_tool_args_no_parser_data(self):; 符号: test_unstreamed_tool_args_no_parser_data, test_kimi_k2_non_streaming_tool_call_id_format, test_kimi_k2_streaming_tool_call_id_format, collect_first_tool_chunk
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +21/-9 (30 lines); hunk: def _process_tool_calls(; async def _process_tool_call_stream(; 符号: _process_tool_calls, _process_tool_call_stream
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10612 - Replace the Kimi-K2 generated tool call idx with history tool call count

- 链接：https://github.com/sgl-project/sglang/pull/10612
- 状态/时间：`merged`，created 2025-09-18, merged 2025-09-26；作者 `eraser00`。
- 代码 diff 已读范围：`2` 个文件，`+226/-15`；代码面：tests/benchmarks；关键词：test。
- 代码 diff 细节：
  - `test/srt/openai_server/basic/test_serving_chat.py` modified +175/-0 (175 lines); hunk: async def collect_first_tool_chunk():; 符号: collect_first_tool_chunk, test_kimi_k2_non_streaming_tool_call_id_with_history, test_kimi_k2_streaming_tool_call_id_with_history, collect_first_tool_chunk
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +51/-15 (66 lines); hunk: process_hidden_states_from_ret,; def _build_chat_response(; 符号: _build_chat_response, _process_response_logprobs, _process_tool_call_id, _process_tool_calls
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/openai_server/basic/test_serving_chat.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10972 - fix: KimiK2Detector Improve tool call ID parsing with regex

- 链接：https://github.com/sgl-project/sglang/pull/10972
- 状态/时间：`merged`，created 2025-09-26, merged 2025-10-01；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+17/-4`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +17/-4 (21 lines); hunk: def __init__(self):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; 符号: __init__, has_tool_call, detect_and_parse, parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/kimik2_detector.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/kimik2_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12759 - [Ascend] support Kimi-K2-Thinking

- 链接：https://github.com/sgl-project/sglang/pull/12759
- 状态/时间：`merged`，created 2025-11-06, merged 2025-11-22；作者 `zhuyijie88`。
- 代码 diff 已读范围：`4` 个文件，`+549/-170`；代码面：model wrapper, MoE/router, quantization, scheduler/runtime；关键词：expert, config, moe, quant, attention, cache, cuda, deepep, fp8, kv。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +480/-39 (519 lines); hunk: from __future__ import annotations; QuantizationConfig,; 符号: npu_wrapper_rmsnorm_init, npu_fused_experts, W8A8Int8Config, for
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +62/-130 (192 lines); hunk: _is_fp8_fnuz = is_fp8_fnuz(); def forward_npu(; 符号: forward_npu, _forward_normal, _forward_ll, _forward_ll
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-0 (6 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; 符号: load_weights, load_weights
  - `python/sglang/srt/model_executor/model_runner.py` modified +1/-1 (2 lines); hunk: def add_chunked_prefix_cache_attention_backend(backend_name):; 符号: add_chunked_prefix_cache_attention_backend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 expert, config, moe, quant, attention, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13150 - Opt kimi_k2_thinking biased topk module

- 链接：https://github.com/sgl-project/sglang/pull/13150
- 状态/时间：`merged`，created 2025-11-12, merged 2025-11-13；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+71/-14`；代码面：MoE/router；关键词：expert, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +71/-14 (85 lines); hunk: def grouped_topk_cpu(; def biased_grouped_topk_gpu(; 符号: grouped_topk_cpu, kimi_k2_biased_topk_impl, biased_grouped_topk_impl, biased_grouped_topk_gpu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 expert, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13287 - [opt kimi k2 1 / n] Add kimi k2 moe fused gate

- 链接：https://github.com/sgl-project/sglang/pull/13287
- 状态/时间：`merged`，created 2025-11-14, merged 2025-11-15；作者 `BBuf`。
- 代码 diff 已读范围：`8` 个文件，`+646/-0`；代码面：MoE/router, kernel, tests/benchmarks；关键词：moe, topk, cuda, expert, fp8, config, spec, test, benchmark, fp4。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` added +354/-0 (354 lines); hunk: +#include <ATen/cuda/CUDAContext.h>; 符号: int, int, int, int
  - `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py` added +124/-0 (124 lines); hunk: +import pytest; 符号: test_kimi_k2_moe_fused_gate, test_kimi_k2_specific_case
  - `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +117/-0 (117 lines); hunk: +import itertools; 符号: kimi_k2_biased_topk_torch_compile, kimi_k2_biased_topk_fused_kernel, benchmark
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +35/-0 (35 lines); hunk: def moe_fused_gate(; 符号: moe_fused_gate, kimi_k2_moe_fused_gate, fp8_blockwise_scaled_grouped_mm
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunk: std::vector<at::Tensor> moe_fused_gate(
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`；patch 关键词为 moe, topk, cuda, expert, fp8, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`, `sgl-kernel/tests/test_kimi_k2_moe_fused_gate.py`, `sgl-kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13332 - [opt kimi k2 2/n] apply kimi k2 thinking moe_fused_gate

- 链接：https://github.com/sgl-project/sglang/pull/13332
- 状态/时间：`merged`，created 2025-11-15, merged 2025-11-16；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+6/-9`；代码面：MoE/router；关键词：cuda, expert, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +6/-9 (15 lines); hunk: _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip; def biased_grouped_topk_gpu(; 符号: biased_grouped_topk_gpu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 cuda, expert, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13374 - [opt kimi k2 3/n] opt kimi_k2 moe_fused_gate kernel

- 链接：https://github.com/sgl-project/sglang/pull/13374
- 状态/时间：`merged`，created 2025-11-16, merged 2025-11-18；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+130/-173`；代码面：MoE/router, kernel；关键词：cuda, expert, moe, spec, topk。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +130/-173 (303 lines); hunk: #include <ATen/cuda/CUDAContext.h>; static constexpr int SMALL_TOKEN_THRESHOLD = 512;; 符号: int, int, int, int
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；patch 关键词为 cuda, expert, moe, spec, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13466 - [Piecewise CUDA Graph] Support Kimi-K2 (non-Thinking)

- 链接：https://github.com/sgl-project/sglang/pull/13466
- 状态/时间：`merged`，created 2025-11-18, merged 2025-11-21；作者 `b8zhong`。
- 代码 diff 已读范围：`1` 个文件，`+23/-0`；代码面：MoE/router；关键词：cuda, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +23/-0 (23 lines); hunk: if _is_cuda:; 符号: _kimi_k2_moe_fused_gate
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 cuda, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13587 - [opt kimi k2 4 / n] Delete useless pad kernel in sgl_moe_align_block_size

- 链接：https://github.com/sgl-project/sglang/pull/13587
- 状态/时间：`merged`，created 2025-11-19, merged 2025-11-21；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+1/-6`；代码面：MoE/router, kernel；关键词：benchmark, expert, moe, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +1/-6 (7 lines); hunk: def moe_align_block_size(; def moe_align_block_size(; 符号: moe_align_block_size, moe_align_block_size
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`；patch 关键词为 benchmark, expert, moe, topk, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13596 - [kimi k2 thinking] Avoid useless torch.zeros_

- 链接：https://github.com/sgl-project/sglang/pull/13596
- 状态/时间：`merged`，created 2025-11-19, merged 2025-11-21；作者 `BBuf`。
- 代码 diff 已读范围：`7` 个文件，`+252/-256`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：marlin, moe, quant, triton, cuda, config, expert, awq, topk, cache。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` added +239/-0 (239 lines); hunk: +import functools; 符号: get_scalar_type, fused_marlin_moe, fused_marlin_moe_fake
  - `sgl-kernel/python/sgl_kernel/fused_moe.py` modified +0/-232 (232 lines); hunk: -import functools; def moe_wna16_marlin_gemm(; 符号: get_scalar_type, moe_wna16_marlin_gemm, moe_wna16_marlin_gemm, fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +3/-12 (15 lines); hunk: from enum import Enum; from aiter.ops.shuffle import shuffle_weight; 符号: apply, apply
  - `python/sglang/srt/layers/quantization/awq.py` modified +4/-6 (10 lines); hunk: import torch_npu; def apply(; 符号: apply
  - `python/sglang/srt/layers/quantization/gptq.py` modified +4/-4 (8 lines); hunk: _is_cuda = is_cuda(); def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；patch 关键词为 marlin, moe, quant, triton, cuda, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `sgl-kernel/python/sgl_kernel/fused_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13725 - Add Expert Parallelism (EP) support for kimi-k2-thinking

- 链接：https://github.com/sgl-project/sglang/pull/13725
- 状态/时间：`merged`，created 2025-11-21, merged 2025-12-07；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+12/-0`；代码面：MoE/router, quantization；关键词：config, expert, marlin, moe, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +12/-0 (12 lines); hunk: def apply(; def apply(; 符号: apply, apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；patch 关键词为 config, expert, marlin, moe, quant, router。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13789 - [DeepEP Support] Support kimi-k2-thinking deepep

- 链接：https://github.com/sgl-project/sglang/pull/13789
- 状态/时间：`closed`，created 2025-11-23, closed 2026-04-16；作者 `BBuf`。
- 代码 diff 已读范围：`10` 个文件，`+674/-0`；代码面：MoE/router, quantization, kernel；关键词：moe, deepep, expert, marlin, quant, topk, config, cuda, triton, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +208/-0 (208 lines); hunk: def fused_marlin_moe_fake(; 符号: fused_marlin_moe_fake, batched_fused_marlin_moe
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +150/-0 (150 lines); hunk: def apply(; 符号: apply, apply_deepep_normal, apply_deepep_ll
  - `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +140/-0 (140 lines); hunk: limitations under the License.; void moe_align_block_size(; 符号: int32_t, int32_t, void
  - `python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py` modified +88/-0 (88 lines); hunk: def moe_align_block_size(; 符号: moe_align_block_size, batched_moe_align_block_size
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +36/-0 (36 lines); hunk: def run_moe_core(; def run_moe_core(; 符号: run_moe_core, run_moe_core, combine, _is_marlin_moe
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu`；patch 关键词为 moe, deepep, expert, marlin, quant, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`, `sgl-kernel/csrc/moe/moe_align_kernel.cu` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15100 - Support piecewise cuda graph for fused marlin moe

- 链接：https://github.com/sgl-project/sglang/pull/15100
- 状态/时间：`merged`，created 2025-12-14, merged 2025-12-16；作者 `ispobock`。
- 代码 diff 已读范围：`5` 个文件，`+55/-36`；代码面：MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks；关键词：expert, marlin, moe, quant, triton, config, cuda, topk, fp8, test。
- 代码 diff 细节：
  - `test/srt/test_piecewise_cuda_graph.py` modified +35/-0 (35 lines); hunk: def test_mgsm_accuracy(self):; 符号: test_mgsm_accuracy, TestPiecewiseCudaGraphGPTQ, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/gptq.py` modified +0/-29 (29 lines); hunk: def _(b_q_weight, perm, size_k, size_n, num_bits):; 符号: _, _, _
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` modified +14/-3 (17 lines); hunk: import torch; def fused_marlin_moe(; 符号: fused_marlin_moe, fused_marlin_moe_fake
  - `python/sglang/srt/layers/moe/moe_runner/marlin.py` modified +4/-2 (6 lines); hunk: def fused_experts_none_to_marlin(; def fused_experts_none_to_marlin(; 符号: fused_experts_none_to_marlin, fused_experts_none_to_marlin
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +2/-2 (4 lines); hunk: def apply(; def apply(; 符号: apply, apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py`；patch 关键词为 expert, marlin, moe, quant, triton, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_piecewise_cuda_graph.py`, `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15306 - Fix warp illegal instruction in kimi k2 thinking PCG

- 链接：https://github.com/sgl-project/sglang/pull/15306
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-18；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+12/-4`；代码面：MoE/router, kernel；关键词：expert, moe, topk。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` modified +12/-4 (16 lines); hunk: __global__ void kimi_k2_moe_fused_gate_kernel_small_token(; __global__ void kimi_k2_moe_fused_gate_kernel(; 符号: void, void
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`；patch 关键词为 expert, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15347 - Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate`

- 链接：https://github.com/sgl-project/sglang/pull/15347
- 状态/时间：`merged`，created 2025-12-18, merged 2026-01-19；作者 `leejnau`。
- 代码 diff 已读范围：`3` 个文件，`+165/-12`；代码面：MoE/router, kernel, tests/benchmarks；关键词：cuda, expert, moe, test, topk, config, flash, spec。
- 代码 diff 细节：
  - `test/registered/kernels/test_fused_topk_deepseek.py` added +97/-0 (97 lines); hunk: +import pytest; 符号: test_fused_topk_deepseek
  - `python/sglang/srt/layers/moe/topk.py` modified +66/-4 (70 lines); hunk: if _is_cuda:; def biased_grouped_topk_gpu(; 符号: biased_grouped_topk_gpu, biased_grouped_topk_gpu
  - `test/srt/test_deepseek_v3_mtp.py` modified +2/-8 (10 lines); hunk: def test_a_gsm8k(; def test_bs_1_speed(self):; 符号: test_a_gsm8k, test_bs_1_speed, test_bs_1_speed
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py`；patch 关键词为 cuda, expert, moe, test, topk, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17325 - Fix kernel selection in biased_grouped_topk_gpu

- 链接：https://github.com/sgl-project/sglang/pull/17325
- 状态/时间：`merged`，created 2026-01-19, merged 2026-01-19；作者 `yudian0504`。
- 代码 diff 已读范围：`1` 个文件，`+0/-1`；代码面：MoE/router；关键词：cuda, expert, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +0/-1 (1 lines); hunk: def biased_grouped_topk_gpu(; 符号: biased_grouped_topk_gpu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 cuda, expert, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17523 - [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI

- 链接：https://github.com/sgl-project/sglang/pull/17523
- 状态/时间：`merged`，created 2026-01-21, merged 2026-01-28；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`27` 个文件，`+1540/-43`；代码面：quantization, tests/benchmarks；关键词：test, benchmark, config, kv, attention, spec, cache, eagle, topk, cuda。
- 代码 diff 细节：
  - `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py` added +248/-0 (248 lines); hunk: +"""AMD DeepSeek-V3.2 GSM8K Completion Evaluation Test (8-GPU); 符号: ModelConfig:, __post_init__, get_display_name, get_one_example
  - `.github/workflows/nightly-test-amd.yml` modified +158/-35 (193 lines); hunk: on:; jobs:
  - `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` added +149/-0 (149 lines); hunk: +"""AMD Nightly performance benchmark for DeepSeek-V3.2 model (MTP variant).; 符号: generate_simple_markdown_report, TestNightlyDeepseekV32MTPPerformance, setUpClass, test_bench_one_batch
  - `test/registered/amd/accuracy/mi35x/test_deepseek_v32_mtp_eval_mi35x.py` added +142/-0 (142 lines); hunk: +"""MI35x DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); 符号: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/amd/accuracy/test_deepseek_v32_mtp_eval_amd.py` added +142/-0 (142 lines); hunk: +"""AMD DeepSeek-V3.2 TP+MTP GSM8K Accuracy Evaluation Test (8-GPU); 符号: TestDeepseekV32TPMTP, setUpClass, tearDownClass, test_a_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py`；patch 关键词为 test, benchmark, config, kv, attention, spec。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17656 - [AMD CI] Add moonshotai/Kimi-K2-Instruct-0905 testcases

- 链接：https://github.com/sgl-project/sglang/pull/17656
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-26；作者 `sogalin`。
- 代码 diff 已读范围：`2` 个文件，`+97/-2`；代码面：tests/benchmarks；关键词：test, attention, cache, config, mla, triton。
- 代码 diff 细节：
  - `test/registered/amd/test_kimi_k2_instruct.py` added +95/-0 (95 lines); hunk: +import os; 符号: TestKimiK2Instruct0905, setUpClass, tearDownClass, test_a_gsm8k
  - `.github/workflows/pr-test-amd.yml` modified +2/-2 (4 lines); hunk: jobs:; jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml`；patch 关键词为 test, attention, cache, config, mla, triton。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/test_kimi_k2_instruct.py`, `.github/workflows/pr-test-amd.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17789 - Support Kimi-K2.5 model

- 链接：https://github.com/sgl-project/sglang/pull/17789
- 状态/时间：`merged`，created 2026-01-27, merged 2026-01-27；作者 `yhyang201`。
- 代码 diff 已读范围：`11` 个文件，`+1053/-12`；代码面：model wrapper, attention/backend, multimodal/processor, docs/config；关键词：config, attention, vision, kv, quant, eagle, flash, lora, mla, processor。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` added +744/-0 (744 lines); hunk: +import logging; 符号: apply_rope, tpool_patch_merger, MoonViTEncoderLayer, __init__
  - `python/sglang/srt/configs/kimi_k25.py` added +171/-0 (171 lines); hunk: +"""; 符号: KimiK25VisionConfig, __init__, KimiK25Config, __init__
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` added +88/-0 (88 lines); hunk: +import re; 符号: KimiK2_5VLImageProcessor, __init__, process_mm_data_async, _process_and_collect_mm_items
  - `python/sglang/srt/parser/reasoning_parser.py` modified +21/-1 (22 lines); hunk: def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):; class ReasoningParser:; 符号: __init__, KimiK2Detector, __init__, Qwen3Detector
  - `python/sglang/srt/configs/model_config.py` modified +11/-9 (20 lines); hunk: def _derive_model_shapes(self):; def _derive_model_shapes(self):; 符号: _derive_model_shapes, _derive_model_shapes, is_generation_model
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py`；patch 关键词为 config, attention, vision, kv, quant, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/configs/kimi_k25.py`, `python/sglang/srt/multimodal/processors/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17991 - Fix: Avoid Double Reduce in VLM DP Attention

- 链接：https://github.com/sgl-project/sglang/pull/17991
- 状态/时间：`merged`，created 2026-01-30, merged 2026-02-02；作者 `yhyang201`。
- 代码 diff 已读范围：`4` 个文件，`+51/-12`；代码面：model wrapper, attention/backend, multimodal/processor, tests/benchmarks；关键词：attention, test, config, cuda, mla, quant, spec, vision。
- 代码 diff 细节：
  - `test/registered/distributed/test_dp_attention_large.py` modified +47/-0 (47 lines); hunk: import requests; from sglang.test.kits.regex_constrained_kit import TestRegexConstrainedMixin; 符号: test_gsm8k, TestDPAttentionDP2TP4VLM, setUpClass, tearDownClass
  - `python/sglang/srt/layers/attention/vision.py` modified +1/-10 (11 lines); hunk: from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm; def __init__(; 符号: __init__, forward, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-0 (3 lines); hunk: KIMIV_VT_INFER_MAX_PATCH_NUM = 16328; def __init__(; 符号: apply_rope, __init__, forward
  - `test/registered/distributed/test_dp_attention.py` modified +0/-2 (2 lines); hunk: def test_gsm8k(self):; 符号: test_gsm8k, TestDPAttentionDP2TP2VLM, setUpClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 attention, test, config, cuda, mla, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/distributed/test_dp_attention_large.py`, `python/sglang/srt/layers/attention/vision.py`, `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18064 - fix kimi k2.5's moe gemm config init

- 链接：https://github.com/sgl-project/sglang/pull/18064
- 状态/时间：`merged`，created 2026-02-01, merged 2026-02-05；作者 `cicirori`。
- 代码 diff 已读范围：`1` 个文件，`+6/-1`；代码面：scheduler/runtime；关键词：config, expert, fp4, fp8, moe, scheduler。
- 代码 diff 细节：
  - `python/sglang/srt/managers/scheduler.py` modified +6/-1 (7 lines); hunk: def init_tokenizer(self):; 符号: init_tokenizer, init_moe_gemm_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/scheduler.py`；patch 关键词为 config, expert, fp4, fp8, moe, scheduler。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/scheduler.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18269 - [AMD] Fix Janus-Pro crash and add Kimi-K2.5 nightly test

- 链接：https://github.com/sgl-project/sglang/pull/18269
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-11；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`4` 个文件，`+250/-10`；代码面：model wrapper, tests/benchmarks；关键词：config, test, attention, benchmark, cache, mla, triton, doc, fp4, processor。
- 代码 diff 细节：
  - `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py` added +106/-0 (106 lines); hunk: +"""MI35x Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU); 符号: TestKimiK25EvalMI35x, setUpClass, test_kimi_k25_gsm8k_accuracy
  - `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py` added +104/-0 (104 lines); hunk: +"""AMD Kimi-K2.5 GSM8K Completion Evaluation Test (8-GPU); 符号: TestKimiK25EvalAMD, setUpClass, tearDownClass, test_kimi_k25_gsm8k_accuracy
  - `.github/workflows/nightly-test-amd.yml` modified +39/-9 (48 lines); hunk: on:; jobs:
  - `python/sglang/srt/models/deepseek_janus_pro.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__, get_image_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`；patch 关键词为 config, test, attention, benchmark, cache, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/accuracy/mi35x/test_kimi_k25_eval_mi35x.py`, `test/registered/amd/accuracy/mi30x/test_kimi_k25_eval_amd.py`, `.github/workflows/nightly-test-amd.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18370 - [Kimi-K2.5] Fix NVFP4 Kimi-K2.5 weight mapping and exclude list

- 链接：https://github.com/sgl-project/sglang/pull/18370
- 状态/时间：`merged`，created 2026-02-06, merged 2026-02-08；作者 `mmangkad`。
- 代码 diff 已读范围：`2` 个文件，`+30/-1`；代码面：model wrapper, quantization；关键词：config, fp4, fp8, kv, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +17/-0 (17 lines); hunk: CombineInput,; def get_config_filenames(cls) -> List[str]:; 符号: get_config_filenames, get_scaled_act_names, apply_weight_name_mapper, ModelOptFp8Config
  - `python/sglang/srt/models/kimi_k25.py` modified +13/-1 (14 lines); hunk: from sglang.srt.model_loader.weight_utils import default_weight_loader; def vision_tower_forward_auto(; 符号: vision_tower_forward_auto, KimiK25ForConditionalGeneration, __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 config, fp4, fp8, kv, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18434 - [Fix] Kimi K2.5 support pp

- 链接：https://github.com/sgl-project/sglang/pull/18434
- 状态/时间：`merged`，created 2026-02-08, merged 2026-02-25；作者 `lw9527`。
- 代码 diff 已读范围：`2` 个文件，`+14/-13`；代码面：model wrapper；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-12 (23 lines); hunk: def forward(; def forward(; 符号: forward, forward
  - `python/sglang/srt/models/kimi_k25.py` modified +3/-1 (4 lines); hunk: MultimodalDataItem,; def forward(; 符号: forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18440 - [Kimi-K2.5] Fix missing `quant_config` in `KimiK25`

- 链接：https://github.com/sgl-project/sglang/pull/18440
- 状态/时间：`merged`，created 2026-02-08, merged 2026-02-08；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：config, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 config, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18689 - Add DP ViT support for Kimi K2.5

- 链接：https://github.com/sgl-project/sglang/pull/18689
- 状态/时间：`merged`，created 2026-02-12, merged 2026-02-18；作者 `yhyang201`。
- 代码 diff 已读范围：`1` 个文件，`+20/-4`；代码面：model wrapper；关键词：config, flash, kv, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +20/-4 (24 lines); hunk: from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM; class MoonViT3dPretrainedModel(nn.Module):; 符号: MoonViT3dPretrainedModel, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 config, flash, kv, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19120 - fix KimiK2Detector regex patterns with re.DOTALL

- 链接：https://github.com/sgl-project/sglang/pull/19120
- 状态/时间：`merged`，created 2026-02-21, merged 2026-02-21；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+5/-3`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +5/-3 (8 lines); hunk: def __init__(self):; def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult; 符号: __init__, detect_and_parse
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/kimik2_detector.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/kimik2_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19181 - [Kernel Slimming] Migrate marlin moe kernel to JIT

- 链接：https://github.com/sgl-project/sglang/pull/19181
- 状态/时间：`merged`，created 2026-02-23, merged 2026-02-26；作者 `celve`。
- 代码 diff 已读范围：`7` 个文件，`+3780/-4`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：expert, marlin, moe, topk, cuda, cache, processor, quant, triton, awq。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h` added +1896/-0 (1896 lines); hunk: +/*; 符号: void, void, auto, auto
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` added +1089/-0 (1089 lines); hunk: +/*; 符号: void, void, void, auto
  - `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` added +329/-0 (329 lines); hunk: +import itertools; 符号: stack_and_dev, _get_scalar_type, _setup_moe_weights, _run_single_gemm
  - `python/sglang/jit_kernel/benchmark/bench_moe_wna16_marlin.py` added +251/-0 (251 lines); hunk: +import os; 符号: stack_and_dev, _make_inputs, _run_jit, _run_aot
  - `python/sglang/jit_kernel/moe_wna16_marlin.py` added +172/-0 (172 lines); hunk: +from __future__ import annotations; 符号: _jit_moe_wna16_marlin_module, _or_empty, moe_wna16_marlin_gemm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py`；patch 关键词为 expert, marlin, moe, topk, cuda, cache。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19228 - [AMD] optimize Kimi K2.5 fused_moe_triton performance by tuning

- 链接：https://github.com/sgl-project/sglang/pull/19228
- 状态/时间：`merged`，created 2026-02-24, merged 2026-02-26；作者 `ZiguanWang`。
- 代码 diff 已读范围：`5` 个文件，`+486/-23`；代码面：MoE/router, kernel, tests/benchmarks, docs/config；关键词：config, moe, triton, benchmark, expert, fp8, quant, cuda, scheduler, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json` added +164/-0 (164 lines); hunk: +{
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +72/-12 (84 lines); hunk: ); def benchmark_config(; 符号: benchmark_config, benchmark_config, benchmark_config, get_kernel_wrapper
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +63/-6 (69 lines); hunk: ); def benchmark_config(; 符号: benchmark_config, benchmark_config, benchmark_config, run
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +23/-5 (28 lines); hunk: def get_model_config(; def get_model_config(; 符号: get_model_config, get_model_config, get_config_filename, get_config_filename
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`；patch 关键词为 config, moe, triton, benchmark, expert, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=384,N=128,device_name=,dtype=int4_w4a16_down.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19331 - [NPU] support Kimi-K2.5 on NPU

- 链接：https://github.com/sgl-project/sglang/pull/19331
- 状态/时间：`merged`，created 2026-02-25, merged 2026-02-26；作者 `khalil2ji3mp6`。
- 代码 diff 已读范围：`3` 个文件，`+23/-3`；代码面：model wrapper, MoE/router, quantization；关键词：quant, config, moe, attention, deepep, expert, topk, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +14/-2 (16 lines); hunk: from transformers import activations; from sglang.srt.models.utils import WeightsMapper; 符号: apply_rope, get_1d_sincos_pos_embed_from_grid, get_rope_shape, load_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunk: from sglang.srt.layers.moe.token_dispatcher.moriep import MoriEPNormalCombineInput; def forward_npu(; 符号: forward_npu
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +1/-0 (1 lines); hunk: def _add_fused_moe_to_target_scheme_map(self):; 符号: _add_fused_moe_to_target_scheme_map, weight_block_size
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；patch 关键词为 quant, config, moe, attention, deepep, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19552 - [feat] Enhance Kimi-K2/K2.5 function call and reasoning detection

- 链接：https://github.com/sgl-project/sglang/pull/19552
- 状态/时间：`merged`，created 2026-02-28, merged 2026-03-19；作者 `AlfredYyong`。
- 代码 diff 已读范围：`2` 个文件，`+700/-19`；代码面：tests/benchmarks；关键词：doc, spec, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_kimik2_detector.py` added +667/-0 (667 lines); hunk: +import json; 符号: _make_tool, _collect_streaming_tool_calls, TestKimiK2DetectorBasic, setUp
  - `python/sglang/srt/function_call/kimik2_detector.py` modified +33/-19 (52 lines); hunk: logger = logging.getLogger(__name__); def __init__(self):; 符号: _strip_special_tokens, KimiK2Detector, __init__, has_tool_call
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py`；patch 关键词为 doc, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_kimik2_detector.py`, `python/sglang/srt/function_call/kimik2_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19689 - feat: support Kimi K2.5 for Eagle3

- 链接：https://github.com/sgl-project/sglang/pull/19689
- 状态/时间：`merged`，created 2026-03-02, merged 2026-03-03；作者 `yefei12`。
- 代码 diff 已读范围：`1` 个文件，`+29/-0`；代码面：model wrapper；关键词：config, eagle, expert, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +29/-0 (29 lines); hunk: def get_model_config_for_expert_location(cls, config: KimiK25Config):; 符号: get_model_config_for_expert_location, set_eagle3_layers_to_capture, get_embed_and_head, set_embed_and_head
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 config, eagle, expert, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19703 - [JIT Kernel] Migrate kimi_k2_moe_fused_gate to JIT

- 链接：https://github.com/sgl-project/sglang/pull/19703
- 状态/时间：`open`，created 2026-03-02；作者 `xingsy97`。
- 代码 diff 已读范围：`5` 个文件，`+576/-1`；代码面：MoE/router, kernel, tests/benchmarks；关键词：moe, topk, cuda, expert, config, test, benchmark, cache, kv, triton。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh` added +317/-0 (317 lines); hunk: +#include <sgl_kernel/tensor.h>; 符号: int, int, int, int
  - `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py` added +111/-0 (111 lines); hunk: +import itertools; 符号: check_correctness, benchmark, fn, fn
  - `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` added +84/-0 (84 lines); hunk: +import itertools; 符号: _reference_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate, test_kimi_k2_moe_fused_gate_wrong_experts
  - `python/sglang/jit_kernel/kimi_k2_moe_fused_gate.py` added +63/-0 (63 lines); hunk: +from __future__ import annotations; 符号: _jit_kimi_k2_moe_fused_gate_module, _kimi_k2_moe_fused_gate_op, kimi_k2_moe_fused_gate
  - `python/sglang/srt/layers/moe/topk.py` modified +1/-1 (2 lines); hunk: fused_topk_deepseek = None
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py`；patch 关键词为 moe, topk, cuda, expert, config, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/moe/kimi_k2_moe_fused_gate.cuh`, `python/sglang/jit_kernel/benchmark/bench_kimi_k2_moe_fused_gate.py`, `python/sglang/jit_kernel/tests/test_kimi_k2_moe_fused_gate.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19802 - [Nightly] Add Kimi K2.5 nightly test (base + Eagle3 MTP), replace Kimi K2

- 链接：https://github.com/sgl-project/sglang/pull/19802
- 状态/时间：`merged`，created 2026-03-03, merged 2026-03-07；作者 `alisonshao`。
- 代码 diff 已读范围：`2` 个文件，`+72/-53`；代码面：model wrapper, tests/benchmarks；关键词：benchmark, cuda, test, config, eagle, spec, topk。
- 代码 diff 细节：
  - `test/registered/8-gpu-models/test_kimi_k25.py` added +72/-0 (72 lines); hunk: +import unittest; 符号: TestKimiK25, for, test_kimi_k25
  - `test/registered/8-gpu-models/test_kimi_k2.py` removed +0/-53 (53 lines); hunk: -import unittest; 符号: TestKimiK2, for, test_kimi_k2
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py`；patch 关键词为 benchmark, cuda, test, config, eagle, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/8-gpu-models/test_kimi_k25.py`, `test/registered/8-gpu-models/test_kimi_k2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19959 - Fix Kimi K2.5 PP layer range exposure for PD disaggregation

- 链接：https://github.com/sgl-project/sglang/pull/19959
- 状态/时间：`merged`，created 2026-03-05, merged 2026-03-07；作者 `yafengio`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +8/-0 (8 lines); hunk: def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):; 符号: pad_input_ids, start_layer, end_layer, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20747 - fix piecewise cuda graph support for Kimi-K2.5 model

- 链接：https://github.com/sgl-project/sglang/pull/20747
- 状态/时间：`merged`，created 2026-03-17, merged 2026-03-17；作者 `yhyang201`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：model wrapper；关键词：vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +2/-0 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21004 - [Fix] Add EPLB rebalance support for Kimi K2.5

- 链接：https://github.com/sgl-project/sglang/pull/21004
- 状态/时间：`merged`，created 2026-03-20, merged 2026-03-26；作者 `yafengio`。
- 代码 diff 已读范围：`1` 个文件，`+4/-0`；代码面：model wrapper；关键词：expert。
- 代码 diff 细节：
  - `python/sglang/srt/models/kimi_k25.py` modified +4/-0 (4 lines); hunk: def start_layer(self) -> int:; 符号: start_layer, end_layer, routed_experts_weights_of_layer, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/kimi_k25.py`；patch 关键词为 expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21391 - Fix Kimi K2.5 dp attention+ spec decoding launch crash

- 链接：https://github.com/sgl-project/sglang/pull/21391
- 状态/时间：`merged`，created 2026-03-25, merged 2026-03-26；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`2` 个文件，`+23/-2`；代码面：model wrapper, tests/benchmarks；关键词：eagle, attention, config, spec, test, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/llama_eagle3.py` modified +12/-1 (13 lines); hunk: def forward(; 符号: forward
  - `test/registered/8-gpu-models/test_kimi_k25.py` modified +11/-1 (12 lines); hunk: def test_kimi_k25(self):; def test_kimi_k25(self):; 符号: test_kimi_k25, test_kimi_k25
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py`；patch 关键词为 eagle, attention, config, spec, test, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/llama_eagle3.py`, `test/registered/8-gpu-models/test_kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21741 - [1/N] feat: support compressed-tensors w4afp8 MoE

- 链接：https://github.com/sgl-project/sglang/pull/21741
- 状态/时间：`open`，created 2026-03-31；作者 `guzekai01`。
- 代码 diff 已读范围：`13` 个文件，`+1664/-40`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：fp8, cuda, config, moe, quant, test, triton, expert, topk, benchmark。
- 代码 diff 细节：
  - `benchmark/kernels/quantization/bench_w4a8_moe_decode.py` added +887/-0 (887 lines); hunk: +"""Benchmark breakdown for CUTLASS W4A8 MoE decode (TP=8 dimensions).; 符号: init_dist, pack_int4_to_int8, pack_interleave, CUDATimer:
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py` added +315/-0 (315 lines); hunk: +"""W4AFP8 MoE scheme: INT4 group-quantized weights + FP8 dynamic activations.; 符号: _unpack_repack_int32_to_cutlass_int8, CompressedTensorsW4AFP8MoE, __init__, get_min_capability
  - `python/sglang/test/test_cutlass_w4a8_moe.py` modified +66/-23 (89 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def test_cutlass_w4a8_moe(M, N, K, E, tp_size, use_ep_moe, topk, group_size, dty; 符号: _init_single_gpu_moe_parallel, pack_int4_values_to_int8, test_cutlass_w4a8_moe, test_cutlass_w4a8_moe
  - `python/sglang/jit_kernel/csrc/gemm/per_tensor_absmax_fp8.cuh` added +86/-0 (86 lines); hunk: +#include <sgl_kernel/tensor.h> // For TensorMatcher, SymbolicSize, SymbolicDevice; 符号: size_t, void, uint32_t, size_t
  - `python/sglang/jit_kernel/tests/test_per_tensor_absmax_fp8.py` added +81/-0 (81 lines); hunk: +import itertools; 符号: reference_absmax_scale, test_absmax_correctness, test_absmax_1d, test_absmax_3d
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`；patch 关键词为 fp8, cuda, config, moe, quant, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/quantization/bench_w4a8_moe_decode.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_fp8_moe.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22208 - [AMD] Optimize fused MoE kernel config for small-M decode on gfx950

- 链接：https://github.com/sgl-project/sglang/pull/22208
- 状态/时间：`open`，created 2026-04-06；作者 `Arist12`。
- 代码 diff 已读范围：`1` 个文件，`+20/-6`；代码面：MoE/router, kernel, docs/config；关键词：benchmark, config, marlin, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` modified +20/-6 (26 lines); hunk: def get_default_config(; 符号: get_default_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py`；patch 关键词为 benchmark, config, marlin, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_config.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22488 - Extend kimi2 fused moe gate kernel to support GLM-5 (256 experts) via JIT compilation

- 链接：https://github.com/sgl-project/sglang/pull/22488
- 状态/时间：`open`，created 2026-04-10；作者 `xu-yfei`。
- 代码 diff 已读范围：`4` 个文件，`+794/-53`；代码面：MoE/router, kernel, tests/benchmarks；关键词：cuda, expert, moe, topk, cache, config, quant, spec, test。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu` added +344/-0 (344 lines); hunk: +/* Copyright 2025 SGLang Team. All Rights Reserved.; 符号: int, int, int, void
  - `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py` added +276/-0 (276 lines); hunk: +import sys; 符号: _reference_biased_topk, _call_kernel, test_moe_fused_gate_ungrouped, test_moe_fused_gate_ungrouped_shared_experts
  - `python/sglang/srt/layers/moe/topk.py` modified +94/-53 (147 lines); hunk: is_npu,; def fused_topk_deepseek(; 符号: fused_topk_deepseek, biased_grouped_topk_impl, _biased_grouped_topk_postprocess, _biased_grouped_topk_ungrouped
  - `python/sglang/jit_kernel/moe_fused_gate_ungrouped.py` added +80/-0 (80 lines); hunk: +from __future__ import annotations; 符号: _jit_moe_fused_gate_ungrouped_module, _moe_fused_gate_ungrouped_fake, moe_fused_gate_ungrouped
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 cuda, expert, moe, topk, cache, config。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/moe/moe_fused_gate_ungrouped.cu`, `python/sglang/jit_kernel/tests/test_moe_fused_gate_ungrouped.py`, `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22496 - [Feature] kimi k25 w4a16 support deepep low latency

- 链接：https://github.com/sgl-project/sglang/pull/22496
- 状态/时间：`open`，created 2026-04-10；作者 `zhangxiaolei123456`。
- 代码 diff 已读范围：`11` 个文件，`+4882/-25`；代码面：MoE/router, quantization, kernel；关键词：cuda, expert, cache, moe, config, marlin, deepep, topk, triton, fp4。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h` added +1948/-0 (1948 lines); hunk: +/*; 符号: void, void, int, auto
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh` modified +1264/-6 (1270 lines); hunk: #pragma once; __global__ void permute_cols_kernel(; 符号: void, void, void, void
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` modified +768/-16 (784 lines); hunk: _is_hip = is_hip(); def create_moe_runner(; 符号: _get_deepep_ll_direct_workspace_size, _build_active_expert_ids_kernel, _masked_silu_and_mul_fwd, _build_active_expert_ids_fwd
  - `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_tma_utils.h` added +240/-0 (240 lines); hunk: +#pragma once; 符号: uint32_t, uint32_t, alignas, alignas
  - `python/sglang/jit_kernel/mask_silu_and_mul.py` added +229/-0 (229 lines); hunk: +from __future__ import annotations; 符号: MaskedSiluAndMulKernelConfig:, threads_n, _masked_silu_and_mul_triton_kernel, _validate_kernel_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`；patch 关键词为 cuda, expert, cache, moe, config, marlin。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_direct_template.h`, `python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22806 - feat(w4afp8): add KimiW4AFp8Config for Kimi K2.5 W4AFP8 model loading

- 链接：https://github.com/sgl-project/sglang/pull/22806
- 状态/时间：`open`，created 2026-04-14；作者 `MichaelPBX`。
- 代码 diff 已读范围：`5` 个文件，`+548/-9`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, config, expert, fp8, quant, spec, triton, fp4, cuda, kv。
- 代码 diff 细节：
  - `test/registered/quant/test_kimi_w4afp8_config.py` added +363/-0 (363 lines); hunk: +"""Unit tests for KimiW4AFp8Config and related functionality.; 符号: _make_kimi_quant_config, TestKimiW4AFp8ConfigFromConfig, method, test_basic_parsing
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +155/-2 (157 lines); hunk: class W4AFp8Config(QuantizationConfig):; def get_config_filenames(cls) -> List[str]:; 符号: W4AFp8Config, for, for, __init__
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +15/-4 (19 lines); hunk: def do_load_weights(; 符号: do_load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +13/-2 (15 lines); hunk: def make_expert_params_mapping_fused_mxfp4(; 符号: make_expert_params_mapping_fused_mxfp4, make_expert_input_scale_params_mapping, set_overlap_args
  - `python/sglang/srt/layers/quantization/__init__.py` modified +2/-1 (3 lines); hunk: def override_quantization_method(self, *args, **kwargs):; def override_quantization_method(self, *args, **kwargs):; 符号: override_quantization_method, override_quantization_method
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；patch 关键词为 moe, config, expert, fp8, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/quant/test_kimi_w4afp8_config.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22964 - [fix][Kimi] fix KimiGPUProcessorWrapper _cpu_call output

- 链接：https://github.com/sgl-project/sglang/pull/22964
- 状态/时间：`open`，created 2026-04-16；作者 `litmei`。
- 代码 diff 已读范围：`1` 个文件，`+6/-1`；代码面：multimodal/processor；关键词：cuda, processor。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/kimi_k25.py` modified +6/-1 (7 lines); hunk: def _cpu_call(self, text, images, **kwargs):; 符号: _cpu_call, _get_gpu_norm_tensors
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/kimi_k25.py`；patch 关键词为 cuda, processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/kimi_k25.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23186 - [AMD] Fused qk rmsnorm bf16 for amd/Kimi-K2.5-MXFP4

- 链接：https://github.com/sgl-project/sglang/pull/23186
- 状态/时间：`merged`，created 2026-04-19, merged 2026-04-21；作者 `akao-amd`。
- 代码 diff 已读范围：`1` 个文件，`+12/-0`；代码面：model wrapper, attention/backend；关键词：attention, cache, fp8, kv, mla, quant, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +12/-0 (12 lines); hunk: def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):; def forward_absorb_prepare(; 符号: bmm_fp8, forward_absorb_prepare
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`；patch 关键词为 attention, cache, fp8, kv, mla, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：54；open PR 数：7。
- 仍需跟进的 open PR：[#19703](https://github.com/sgl-project/sglang/pull/19703), [#21741](https://github.com/sgl-project/sglang/pull/21741), [#22208](https://github.com/sgl-project/sglang/pull/22208), [#22488](https://github.com/sgl-project/sglang/pull/22488), [#22496](https://github.com/sgl-project/sglang/pull/22496), [#22806](https://github.com/sgl-project/sglang/pull/22806), [#22964](https://github.com/sgl-project/sglang/pull/22964)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
