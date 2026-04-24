# SGLang DeepSeek V3 / R1 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `929e00eea`、sgl-cookbook `origin/main` 快照 `8ec4d03`，以及 DeepSeek V3/R1 相关 merged、open、reverted PR 的 patch 阅读结果整理。范围只覆盖 DeepSeek V3、V3-0324、R1、R1-0528 以及这些模型的量化、MTP、DeepEP、LoRA 和后端优化；DeepSeek V3.1 的 parser/template 差异和 DeepSeek V3.2 的 DSA/NSA 稀疏注意力单独成文。

结论：截至 `929e00eea`，DeepSeek V3/R1 的主线入口仍是 `python/sglang/srt/models/deepseek_v2.py` 里的 `DeepseekV3ForCausalLM`，MTP 草稿模型入口是 `python/sglang/srt/models/deepseek_nextn.py` 里的 `DeepseekV3ForCausalLMNextN`。当前主线已经形成了完整的 MLA 后端选择、FP8/FP4/W4AFP8/MXFP4/MXFP8/NVFP4 加载、共享专家融合、MTP、R1 W4A8 DeepEP、DP attention、LoRA 和多硬件验证面；新增运行时内容包括 adaptive EAGLE、PCG + speculative decoding、thinking token radix-cache strip 以及 spec v2 adaptive spec。后续需要跟进的方向主要是 open PR 中的 JIT router GEMM、DeepSeek MLA 量化层 `.weight` 访问、ROCm MLA 恢复、LoRA adapter 旁路、CuteDSL EP + DP attention 双重 reduce、MUSA、DCP 和 spec v2 自适应 speculative decoding。

## 1. 时间线总览

| 创建日期 | PR | 状态 | 主线 | 代码区域 | 作用 |
| --- | ---: | --- | --- | --- | --- |
| 2024-12-26 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | AMD bring-up | Triton decode attention、fused MoE、`deepseek_v2.py` | 支持 DeepSeek V3 在 AMD 路径运行。 |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD FP8 | `deepseek_v2.py` | 修正 AMD 上 DeepSeek V3 FP8 精度。 |
| 2025-02-05 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | docs | DeepSeek 文档 | 增加 DeepSeek 使用和多机启动文档。 |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | docs | DeepSeek V3 launch docs | 修正文档里的 DeepSeek V3 启动参数。 |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | MTP | `deepseek_nextn.py`、spec decode | 给 DeepSeek V3/R1 增加 NextN/EAGLE speculative decoding。 |
| 2025-02-26 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | FP8 GEMM | benchmark、DeepGEMM | 增加 DeepGEMM 和 SGLang FP8 block-wise GEMM benchmark。 |
| 2025-03-05 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | INT8 docs | DeepSeek docs | 增加 INT8 启动示例。 |
| 2025-03-07 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGEMM | `sgl-kernel` | 把 DeepGEMM 集成进 `sgl-kernel`。 |
| 2025-03-08 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | DeepGEMM | Linear layers | 让 Linear 支持 DeepGEMM。 |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | MTP/MLA | FlashInfer MLA | 给 FlashInfer MLA backend 增加 NextN 支持。 |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | FlashMLA | attention backend | 增加初始 FlashMLA backend。 |
| 2025-03-17 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | FlashMLA graph | `flashmla_backend.py`、server args | 给 FlashMLA backend 增加 CUDA graph 支持。 |
| 2025-03-18 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | fused MoE | `moe_fused_gate.cu`、test、benchmark | 增加 DeepSeek 风格 fused group gate selection kernel。 |
| 2025-03-20 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | DeepGEMM default | server defaults | 在 Hopper 架构上默认启用 DeepGEMM。 |
| 2025-03-20 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | ROCm MTP | NextN | 在 AMD GPU 上启用 MTP/NextN。 |
| 2025-03-27 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | FA3 MLA | attention backend | 给 MLA 增加 FA3 backend。 |
| 2025-04-05 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | MoE align | `moe_align_kernel.cu`、fused MoE | 降低 `moe_align_block_size_kernel` 小 batch 开销。 |
| 2025-04-07 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | MHA chunked prefill | `flashattention_backend.py`、scheduler、`deepseek_v2.py` | 增加 `MHA_CHUNKED_KV`，支持 DeepSeek chunked prefill。 |
| 2025-04-09 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | FA3 default | server defaults | Hopper 上默认使用 FA3 MLA。 |
| 2025-04-11 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | DeepGEMM guard | defaults | 临时关闭 DeepGEMM 默认启用。 |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | DeepGEMM guard | defaults | 限制 DeepGEMM 只在 Hopper 使用。 |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | fused MoE | `deepseek_v2.py`、MoE gate | 在 DeepSeek V3/R1 应用 fused MoE gate。 |
| 2025-04-14 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | MLA kernel | `merge_attn_states.cu` | 增加更快的 `merge_state_v2` CUDA merge-attention-state kernel。 |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | RoPE | `rotary_embedding.py` | 应用 DeepSeek CUDA RoPE。 |
| 2025-04-14 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Cutlass MLA | `cutlass_mla_backend.py`、sgl-kernel attention | 增加 Cutlass MLA attention backend。 |
| 2025-04-15 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | DeepGEMM BMM | `fp8_kernel.py`、`deepseek_v2.py` | 引入 DeepGEMM `group_gemm_masked` 作为 MLA BMM 探索路径。 |
| 2025-04-16 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | FP8 quant | `fp8_kernel.py`、`fp8_utils.py` | 用 `sgl-kernel` 的 `sglang_per_token_group_quant_fp8` 替换 Triton kernel。 |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | MLA FP8 quant | `fp8_kernel.py`、`deepseek_v2.py` | 复用 zero scalar allocator，并去掉 `per_tensor_quant_mla_fp8` 中一个 kernel。 |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | shared experts | SM90 shared experts | 在 SM90 上启用 DeepSeek V3 shared-experts fusion。 |
| 2025-04-20 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | MLA copy | `deepseek_v2.py`、RoPE | 移除 DeepSeek `forward_absorb` 中一次额外 copy。 |
| 2025-04-22 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | MLA projection | `deepseek_v2.py`、loader | 融合 `q_a_proj` 和 `kv_a_proj_with_mqa`。 |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | DeepGEMM default | defaults、docs | 重新默认开启 DeepGEMM 并更新文档。 |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | MTP/fusion | R1 MTP、shared experts | 修复 R1 同时启用 MTP 和 shared-expert fusion 的组合。 |
| 2025-04-24 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | MoE tuning | Triton fused MoE config | 更新 H20 DeepSeek/R1 FP8 W8A8 fused-MoE Triton config。 |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | MoE tuning | H200 Triton fused MoE config | 更新 H200 Triton 3.2 fused-MoE config 和 warning。 |
| 2025-04-25 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | MLA KV cache | `flashattention_backend.py`、`memory_pool.py`、`deepseek_v2.py` | 融合 MLA set-KV-cache kernel，并去掉 K concat 开销。 |
| 2025-04-27 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | MTP ergonomics | server/spec args | 自动设置 MTP draft model path。 |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | MTP API | CI、docs | 更新 MTP API 变化后的测试和文档。 |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | MLA streams | `deepseek_v2.py` | 用双 stream overlap q/k norm。 |
| 2025-05-05 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | docs | MLA backend docs | 更新 MLA attention backend 文档。 |
| 2025-05-07 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | MTP/DP attention | MTP、DP attention | 支持 MTP 和 DP attention 组合。 |
| 2025-05-08 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | FlashMLA/MTP | FlashMLA、FP8 KV | 支持 FlashMLA backend + MTP + FP8 KV cache。 |
| 2025-05-09 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | hybrid attention | model_runner、cuda graph、server args | 探索 hybrid attention backend；未成为 V3/R1 主线。 |
| 2025-05-12 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | fused MoE | top-k reduce、quant methods | 把 routed scaling factor 融进 top-k reduce kernel。 |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | DeepGEMM/MLA | `fused_qkv_a_proj_with_mqa` | 用 DeepGEMM 替换该 fused projection 上的 Triton 路径。 |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | routed scaling | DeepSeek MoE | 在 DeepSeek 内融合 routed scaling factor。 |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | DeepGEMM format | per-token-group quant | 支持新 DeepGEMM 格式的 per-token-group quant。 |
| 2025-06-13 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | DeepGEMM refactor | DeepGEMM integration | 重构 DeepGEMM 集成。 |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | DeepGEMM format | SRT quant | 在 SRT 侧支持新 DeepGEMM quant 格式。 |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | DeepGEMM format | DeepSeek weights | 重新量化 DeepSeek 权重以适配新 DeepGEMM input format。 |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | DeepGEMM | new DeepGEMM path | 完成新 DeepGEMM 路径支持。 |
| 2025-06-20 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | MTP/FP4 | `deepseek_nextn.py`、spec decode | 修复 DeepSeek R1 FP4 的 MTP。 |
| 2025-07-04 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | R1 W4AFP8 | `w4afp8.py`、`cutlass_w4a8_moe.py`、EP MoE | 新增 R1 W4AFP8 配置、Cutlass W4A8 MoE 和 EP-MoE 路径。 |
| 2025-07-17 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | R1 W4AFP8 TP | Cutlass grouped W4A8 MoE | 给 R1-W4AFP8 增加 TP 模式。 |
| 2025-07-22 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | R1 W4A8 DeepEP | `token_dispatcher/deepep.py`、W4A8 MoE | 给 R1 W4A8/W4AFP8 增加 normal DeepEP。 |
| 2025-07-28 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | R1 W4A8 DeepEP LL | DeepEP low-latency | 给 R1 W4A8 增加 low-latency DeepEP。 |
| 2025-09-04 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | W4AFP8 perf | glue kernels | 优化 R1 W4AFP8 胶水 kernel。 |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | DP/compile | DP + torch compile | 修复 DeepSeek V3 DP + torch-compile GPU fault。 |
| 2025-10-12 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | FP4 default | server defaults | 更新 R1-FP4 在 Blackwell 上的默认配置。 |
| 2025-10-16 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | FP4/SM120 | backend defaults | 支持 SM120 上的 FP4 DeepSeek。 |
| 2025-10-23 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | deterministic | DeepSeek attention | 支持 DeepSeek 架构单卡 deterministic inference。 |
| 2025-10-24 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | docs | W4FP8 docs | 增加 W4FP8 用法示例。 |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | Blackwell default | `server_args.py` | 更新 DeepSeek V3 在 SM100 上的自动量化设置。 |
| 2025-11-09 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | W4AFP8 perf | W4A8 kernels | 优化 DeepSeek-V3-0324 W4AFP8 kernel。 |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | MTP/B200 | NextN、spec decode | 修复 B200 上 DeepSeek V3 MTP。 |
| 2025-11-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepEP LL | R1 W4A8 DeepEP | 让 R1 W4A8 DeepEP low-latency dispatch 使用 FP8 通信。 |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | DP accuracy | BF16 KV | 修复 DeepSeek V3 DP + BF16 KV 精度。 |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | MXFP4 | AMD EP | 修复 MXFP4 DeepSeek V3 + EP 精度。 |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | router/top-k | `topk.py` | 用 `fused_topk_deepseek` 替换通用 `moe_fused_gate` 热路径。 |
| 2025-12-20 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | PCG/FP4 | CUDA graph | 给 DeepSeek V3 FP4 增加 piecewise CUDA graph。 |
| 2026-01-07 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | loader refactor | `deepseek_common/deepseek_weight_loader.py` | 把 DeepSeek V2/V3 权重加载拆成 mixin。 |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | fused MoE configs | 增加 H20/H20-3E 的 DeepSeek 系 MoE config。 |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | eval/parser | eval choices | 从 thinking-mode choices 移除 `deepseek-r1`，因为 R1 parser 和 V3 thinking parser 不同。 |
| 2026-01-25 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | router bench | `dsv3_router_gemm` | 增加 Blackwell router GEMM benchmark。 |
| 2026-01-26 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | loader memory | weight loader | 延迟 `dict(weights)` 物化，避免大 checkpoint OOM。 |
| 2026-02-03 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | ROCm perf | MI300X | 优化 DeepSeek R1 在 MI300X 上的运行。 |
| 2026-02-08 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | AMD router | AITER router GEMM | expert 数不超过 256 时使用 `aiter_dsv3_router_gemm`。 |
| 2026-02-09 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | XPU | Intel GPU | 支持 R1 在 Intel GPU 上推理。 |
| 2026-02-11 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | AMD MTP | TP4 MTP | 修复 AMD TP4 DeepSeek V3 MTP 精度。 |
| 2026-02-22 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | `deepseek_common/attention_forward_methods/` | 把 DeepSeek MLA forward 拆到共享 forward-method 模块。 |
| 2026-02-26 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | R1 MXFP4 | NextN loading | 修复 R1-0528-MXFP4 权重加载 shape mismatch。 |
| 2026-03-04 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | AMD CI | MI35x lanes | 增加 DeepSeek KV FP8 和 all-reduce fusion 的 MI35x 测试。 |
| 2026-03-04 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | AMD perf | AITER FP8 top-k | AITER FP8 路由中保留 BF16 correction bias，避免 runtime dtype conversion。 |
| 2026-03-18 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | DP bugfix | DeepSeek R1 DP | 修复 DeepSeek R1 DP GPU fault。 |
| 2026-03-24 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | MXFP8 | routed MoE | 增加 MXFP8 DeepSeek V3 routed MoE 支持。 |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | server args、EAGLE runtime、spec workers | 给 EAGLE top-k=1 增加自适应 `speculative_num_steps`。 |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | revert | DeepEP LL | 回滚 `#14162`。 |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | `model_runner.py`、PCG runner、server args | 允许 piecewise CUDA graph 和 speculative decoding 同时使用。 |
| 2026-04-08 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | reland | DeepEP LL | 重新合入 R1 W4A8 DeepEP low-latency FP8 通信。 |
| 2026-04-08 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | LoRA | quant info、MLA LoRA | 重构 LoRA quant info，并增加 DeepSeek V3 MLA LoRA 支持。 |
| 2026-04-16 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | CPU shared expert | CPU MoE | 扩展无 scaling factor 时的 CPU shared-expert 接口，偏 CPU parity 而非 H200 吞吐。 |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | model config、scheduler、radix cache、reasoning parser | 探索 parser-gated 两阶段 reasoning radix-cache stripping，未成为当前主线。 |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | quant bugfix | `DeepseekV2AttentionMLA` | 给 AWQ/compressed-tensors 层的 `.weight` 访问加保护。 |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | MoE/DP | CuteDSL EP + DP attention | 修复 `DeepseekV2MoE` 在 CuteDSL EP + DP attention 下的 double-reduce。 |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | `schedule_batch.py`、`mem_cache/common.py`、`server_args.py` | 增加可选的 thinking token radix-cache strip。 |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | scheduler output processor、EAGLE v2 workers | 把 adaptive speculative decoding 扩展到 spec v2。 |

## 2. 单机 H200 优化资料覆盖

单机 H200 优化资料列出了 `#4514`、`#4530`、`#5086`、`#5113`、`#5381`、`#5385`、`#5390`、`#5432`、`#5473`、`#5549`、`#5578`、`#5619`、`#5716`、`#5740`、`#5748`、`#5977`、`#6034`、`#6151`、`#6220`。这些 PR 已进入时间线，并按当前 main 状态标注为默认路径、可选 backend、探索路径或 closed 方向。

这批 PR 的主线可以拆成四条。

第一条是 FP8 Block GEMM / DeepGEMM。`#3893` 先用 benchmark 把 DeepGEMM 和 SGLang FP8 block-wise GEMM 放到同一比较面；`#4165` 把 DeepGEMM 接进 `sgl-kernel`；`#4199` 让 Linear 支持 DeepGEMM。之后 `#4613`、`#5263`、`#5310`、`#5628` 说明默认策略不是一次性固定的，而是在 Hopper 默认、临时关闭、限制架构、重新开启之间迭代。`#5432` 的 DeepGEMM `group_gemm_masked` BMM 和 MLA FP8 quant kernel 属于探索路径，不能直接写成当前 H200 默认；`#5473` 把 per-token-group FP8 quantization 从 Triton 换成 `sgl-kernel`，`#5549` 通过 zero scalar allocator 复用去掉 `per_tensor_quant_mla_fp8` 里一个 kernel。后续 `#6890` 和 `#7146`、`#7150`、`#7155`、`#7156`、`#7172` 又把 fused projection 和 DeepSeek 权重量化迁到新的 DeepGEMM input format。

第二条是 Fused MoE。`#4530` 增加 `moe_fused_gate.cu`、绑定、benchmark 和测试，专门处理 DeepSeek biased grouped top-k / group gate selection；`#5086` 降低 `moe_align_block_size_kernel` 小 batch 开销；`#5371` 把 fused MoE gate 接进 DeepSeek V3/R1；`#5571` 在 SM90 上打开 shared-experts fusion；`#5716` 和 `#5740` 分别补 H20/H200 fused-MoE Triton config；`#6220` 把 routed scaling factor 融进 top-k reduce kernel，`#6970` 又把同类 scaling 直接融进 DeepSeek 路径。读当前 main 时，要同时看 `topk.py`、`fused_moe_triton/fused_moe.py`、`sgl-kernel/csrc/moe/moe_fused_gate.cu`、`moe_align_kernel.cu` 和 `sgl_kernel_ops.h`。

第三条是 MLA/attention backend。FlashMLA 先由 `#4472` 接入，`#4514` 加 CUDA graph，`#6109` 再补 MTP 和 FP8 KV cache；FA3 MLA 由 `#4831` 接入，`#5210` 让 Hopper 默认走 FA3 MLA；Cutlass MLA 是 `#5390`，后续 `#6034` 用文档把 FA3、FlashMLA、Cutlass MLA 等 backend 的选择边界写清楚。模型文件热路径方面，`#5113` 增加 `MHA_CHUNKED_KV`，`#5381` 增加 `merge_state_v2` CUDA kernel，`#5385` 接 DeepSeek CUDA RoPE，`#5578` 移除 `forward_absorb` copy，`#5619` 融合 `q_a_proj` / `kv_a_proj_with_mqa`，`#5748` 融合 MLA set-KV-cache kernel，`#5977` 用双 stream overlap q/k norm。

第四条是 MTP 和 backend 组合。`#3582` 是 V3/R1 NextN/EAGLE 起点，`#4218` 支持 FlashInfer MLA + NextN，`#5707` 修 R1 的 MTP + shared-expert fusion 组合，`#5793` 自动设置 draft model path，`#5952` 更新 MTP API 的测试和文档，`#6081` 支持 MTP + DP attention，`#6109` 又把 FlashMLA、MTP、FP8 KV cache 连接起来。`#6151` 是 closed 的 hybrid attention backend 探索，应作为历史背景记录，但不能算当前主线支持。

## 2.1 MTP/PCG 与 Thinking Radix Cache

SGLang main 快照为 `929e00eea`，sgl-cookbook 快照为 `8ec4d03`。cookbook 更新未改变 DeepSeek 文档或模型条目；相关增量集中在 SGLang 运行时 PR。

`#21599` 把 EAGLE top-k=1 的 `speculative_num_steps` 做成自适应路径，改动覆盖 `server_args.py`、spec runtime state/params、EAGLE workers 和 runner。它会影响 V3/R1 MTP 性能调参：不应再假设 draft step 数是静态常量。`#22128` 允许 piecewise CUDA graph 与 speculative decoding 共存，相关逻辑在 `model_runner.py`、`piecewise_cuda_graph_runner.py` 和 server flag gate；因此排查 PCG + MTP 时不应再归类为“不支持组合”。

`#22950` 是 closed 的 reasoning radix-cache strip 早期方案，涉及 model config、scheduler、radix cache 和 `reasoning_parser.py`；当前主线应以 merged `#23315` 为准。`#23315` 在 `server_args.py` 增加 opt-in 参数，并在 `schedule_batch.py` / `mem_cache/common.py` 里支持把 thinking tokens 从 radix-cache entry 中剥离，避免 `<think>` / `</think>` 这类 reasoning token 变成后续请求可复用 prefix。open `#23336` 则把 adaptive spec 推到 spec v2 的 `scheduler_output_processor_mixin.py`、`managers/utils.py`、`eagle_worker_v2.py` 和 `multi_layer_eagle_worker_v2.py`。

## 3. 当前主线代码形态

DeepSeek V3/R1 的当前主线不是一个新文件，而是复用 DeepSeek V2 时代演进出来的 `deepseek_v2.py`。`DeepseekV3ForCausalLM` 继承 `DeepseekV2ForCausalLM`，核心层包括 `DeepseekV2AttentionMLA`、`DeepseekV2MoE`、`DeepseekV2DecoderLayer` 和 `DeepseekV2Model`。因此，很多命名为 `deepseek_v2` 的修复实际会影响 V3、R1、V3.1 甚至 V3.2。

当前主线最重要的共享模块是：

- `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`：处理 stacked qkv/gate_up、expert 参数映射、`kv_b_proj` 后处理、W4AFP8 scale 映射、DeepGEMM BMM 所需权重变换。
- `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`：根据 backend、deterministic、PCG、MHA/MLA 选择 forward 方法。
- `python/sglang/srt/models/deepseek_common/attention_forward_methods/`：承接 `#19122` 后的 MLA/MHA forward 逻辑。
- `python/sglang/srt/models/deepseek_nextn.py`：MTP/NextN 草稿层。
- `python/sglang/srt/parser/reasoning_parser.py`：`deepseek-r1` 和 `deepseek-v3` reasoning parser 的分界。
- `python/sglang/srt/function_call/deepseekv3_detector.py`：V3/R1 tool call parser。
- `python/sglang/srt/managers/schedule_batch.py` 和 `python/sglang/srt/mem_cache/common.py`：thinking token radix-cache strip 的当前主线路径。
- `python/sglang/srt/server_args.py`：DeepSeek 系默认 attention backend、KV cache dtype、量化 backend、DeepEP/DP attention guard 的主要入口。

`server_args.py` 现在对 DeepSeek V3/R1 做了几类自动选择：在 Blackwell SM100 上，如果没有手动指定 MLA backend，会默认选 `trtllm_mla`；官方 FP8 或 ModelOpt FP8/FP4 量化在合适条件下会倾向 `flashinfer_trtllm` MoE runner；如果启用 piecewise CUDA graph，V3/R1 会记录 “use MLA for prefill” 的路径；ROCm 则会走 AITER 相关 allreduce fusion 和 FP4/EAGLE 的 backend 默认值。这些默认值意味着排查性能问题时不能只看模型文件，必须把启动参数和 server-side 自动改写一起看。

## 4. MLA 与权重加载：从基础支持到 Backend-Aware

DeepSeek V3/R1 的注意力主线是 MLA。`DeepseekV2AttentionMLA` 会根据 `q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、`v_head_dim` 构造 q/k/v latent projection。当前 main 中，`q_a_proj` 和 `kv_a_proj_with_mqa` 可以融合为 `fused_qkv_a_proj_with_mqa`，`kv_b_proj` 则在加载后被拆出 `w_kc`、`w_vc` 等 backend 需要的组件。

`#16649` 把权重加载抽成 `DeepseekV2WeightLoaderMixin` 后，DeepSeek 系后续模型都复用这套逻辑。它的关键细节包括：

- `gate_proj/up_proj` 堆叠到 `gate_up_proj`，q/k/v 相关权重按 MLA 结构做特殊映射。
- expert 参数通过 `make_expert_params_mapping` 和 W4AFP8 input scale mapping 映射。
- shared expert fusion 时，`mlp.shared_experts` 可以映射到 `mlp.experts.256`。
- `kv_b_proj` 对 AWQ、FP8 block scale、DeepGEMM BMM 都有 post-load 处理。
- R1 MXFP4 / NextN 某些 checkpoint 使用 `model.layers.61*` 命名，需要在 `deepseek_nextn.py` 做特殊处理。

`#17744` 属于实用的 OOM 修复：它避免在加载大模型时直接 `dict(weights)`，减少内存尖峰。`#23195` 仍 open，它提示当前量化层可能没有 `.weight` 属性；如果 AWQ 或 compressed-tensors checkpoint 出现 MLA 初始化报错，应先检查这个方向，而不是误判为权重缺失。

`#19122` 之后，MLA forward 被拆到 `attention_forward_methods`。这让 backend 切换更清楚，但也带来兼容性回归风险，因此 open `#22938` 仍在恢复 MI300X 的 DeepSeek MLA 路径，open `#21530` 仍在修 ROCm fused MLA decode RoPE。

## 5. MoE 路由、共享专家与通信边界

DeepSeek V3/R1 的 MoE 由 256 个 routed experts 加 shared experts 组成。当前 main 的 `DeepseekV2MoE` 会根据 config 和 server args 计算 `num_fused_shared_experts`。当共享专家融合启用时，loader 会把 `mlp.shared_experts` 重映射到 `mlp.experts.256`，让普通 routed expert 和 shared expert 进入同一个 fused MoE 计算表面。

这个融合不是无条件开启：

- TBO/SBO 会关闭融合。
- DeepEP 默认关闭融合，除非显式设置 `--enforce-shared-experts-fusion`。
- W4AFP8 会关闭融合，因为 routed experts 和 shared experts 可能使用不同量化方法。
- 架构、expert 数、后端能力不匹配时也会关闭。

DeepEP 下共享专家融合更复杂。普通融合是 `256 + 1`；DeepEP 融合会把本地 expert layout 扩成 `256 + EP_size`，TopK 需要处理 shared expert 在 EP rank 上的交错和映射。这个地方的 bug 往往表现为输出正确性或 double reduce，而不是单个 kernel 速度慢。`#23257` 仍 open，正是修 CuteDSL EP + DP attention 组合下 MoE 内部 allreduce 和外层 DP attention reduce 重叠的问题。

路由侧的主线变化是 `#15347`。DeepSeek 的 biased grouped top-k 不再优先走通用 `moe_fused_gate`，而是在满足条件时用 `fused_topk_deepseek`。`#17707` 增加了 Blackwell router benchmark，`#22933` 扩展了无 scaling factor 时的 CPU shared-expert 接口，属于 CPU parity 清理而不是 H200 GPU 吞吐优化；open `#21531` 则把 `dsv3_router_gemm` 从 AOT sgl-kernel 迁到 JIT，是未来 router 维护性和部署便利性的重点。

## 6. MTP / NextN：草稿层是独立运行面

DeepSeek V3/R1 的 MTP 通过 EAGLE 和 `DeepseekV3ForCausalLMNextN` 实现。它不是简单复用主模型最后一层，而是有独立的 NextN layer、共享 embed/head、独立加载逻辑和可能不同的量化配置。

当前 `deepseek_nextn.py` 的关键约束是：

- 只支持一个 NextN layer。
- target model 是 `DeepseekV3ForCausalLM`，draft model 是 `DeepseekV3ForCausalLMNextN`。
- draft 层可能是 BF16，也可能有与 target 不同的量化处理。
- AMD R1 MXFP4 有特殊命名和 shape 修复。
- 某些 DeepEP BF16 dispatch 环境变量需要围绕 NextN 执行切换。

`#7376` 修了 R1 FP4 MTP，`#13548` 修了 B200 V3 MTP，`#18607` 修了 AMD TP4 V3 MTP 精度，`#19425` 修了 R1-0528-MXFP4 的 draft loading shape。当前验证面里，H200 V3 MTP 注册测试要求 GSM8K 高于 `0.935`，平均 spec accept length 高于 `2.8`，并且 batch-size-1 吞吐超过普通 V3 lane。

较新的 spec 线还补充了两个需要单独记录的约束：`#21599` 让 EAGLE top-k=1 的 draft step 数可以自适应，`#22128` 让 PCG 可以和 speculative decoding 共存。open `#23336` 还会继续改 spec v2 worker 的自适应路径。写 skill 或排查性能时，要把 target model、draft model、spec v1/v2、PCG 和 DP attention 一起记录。

## 7. R1 W4AFP8 / W4A8 DeepEP：单独的量化优化梯子

R1 W4AFP8 不能按普通 FP8 来理解。`#7762` 引入的 `W4AFp8Config` 会根据 quant config 判断 mixed precision，把普通 Linear 映射到 FP8 或非量化方法，把 MoE experts 映射到 W4A8。`cutlass_w4a8_moe.py` 处理 packed int4 expert weight、FP8 activation、input scale 和 grouped MoE runner。

后续几个 PR 让这条路径完整起来：

- `#8118` 给 R1-W4AFP8 加 TP mode。
- `#8247` 加 normal DeepEP，核心是让 DeepEP dispatch metadata 能进入 W4A8 MoE 的 `apply_deepep_normal`。
- `#8464` 加 low-latency DeepEP。
- `#10027` 和 `#12921` 优化 W4AFP8 glue kernel 和 DeepSeek-V3-0324 的 W4AFP8 性能。

low-latency DeepEP 的 FP8 通信线必须按 revert history 阅读：`#14162` 合入过 R1 W4A8 DeepEP LL FP8 communication，`#21719` 回滚，`#22316` 重新合入。只读 `#14162` 会得到错误结论，当前 main 的状态要以 `#22316` 之后的代码为准。

## 8. 量化、平台和 parser 支持

DeepSeek V3/R1 当前量化面很宽：

- 官方 V3 FP8：不需要再手动声明 `--quantization fp8`，server 会识别量化配置。
- FP4/NVFP4：`#11512`、`#11708`、`#12778` 让 Blackwell/SM120 默认值和 backend 选择更合理。
- W4AFP8/W4A8：以 `w4afp8.py`、`cutlass_w4a8_moe.py`、DeepEP normal/LL 为中心。
- MXFP4：`#15304` 修 AMD EP 精度，`#19425` 修 R1-0528-MXFP4 draft loading，open `#21529` 继续推进 ROCm Quark W4A4。
- MXFP8：`#21280` 给 routed MoE 增加 MXFP8 支持。
- LoRA：`#22323` 重构 LoRA quant info 并支持 DeepSeek V3 MLA LoRA；open `#22268` 指向 `prepare_qkv_latent` 可能绕过 adapter 的问题。

parser 也要分清：V3/R1 tool calling 使用 `--tool-call-parser deepseekv3`，V3-style thinking 使用 `--reasoning-parser deepseek-v3`，R1 使用 `--reasoning-parser deepseek-r1`。R1 parser 会处理没有起始 `<think>` 的场景，在遇到 `</think>` 前强制当作 reasoning；这和 V3/V3.1 的 Qwen3-style thinking parser 不同。

thinking parser 之外还要单独看 radix cache。`#23315` 的 opt-in strip 是 cache 层行为：它决定 thinking tokens 是否可以作为 prefix cache 内容复用；它不是 `deepseekv3_detector.py` 或 `reasoning_parser.py` 的 parser 格式变化。遇到多轮 reasoning 输出异常时，要同时记录 parser、strip flag 和缓存命中情况。

## 9. 当前验证面与未合入方向

当前 main 里的验证面包括：

- `test/registered/8-gpu-models/test_deepseek_v3_basic.py`：H200 V3 基础精度和性能，GSM8K 阈值高于 `0.935`。
- `test/registered/8-gpu-models/test_deepseek_v3_mtp.py`：H200 V3 MTP，检查 avg spec accept length 和吞吐。
- `test/registered/amd/test_deepseek_v3_basic.py`、`test/registered/amd/test_deepseek_v3_mtp.py`、`test/registered/amd/test_deepseek_r1_mxfp4_8gpu.py`：AMD 基础、MTP、R1 MXFP4。
- `test/registered/backends/test_deepseek_r1_fp8_trtllm_backend.py`：R1 FP8 TRTLLM backend。
- `test/registered/quant/test_deepseek_v3_fp4_4gpu.py`、`test/registered/quant/test_w4a8_deepseek_v3.py`：FP4 和 W4A8。
- `test/registered/mla/test_mla_deepseek_v3.py`、`test/registered/mla/test_mla_int8_deepseek_v3.py`：MLA 和 INT8 MLA。
- `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`：LoRA logprob regression。
- `test/registered/kernels/test_fused_topk_deepseek.py`：DeepSeek fused top-k。

需要跟进的 open PR：

- `#14194`：DeepSeek V2/V3 DCP。
- `#15315`、`#15380`：DeepSeek-R1-W4AFP8 group GEMM。
- `#18892`：DeepSeek V3 GEMM JIT。
- `#21526`：ROCm AITER router GEMM regression。
- `#21529`：ROCm DeepSeek 架构 MXFP4/Quark W4A4。
- `#21530`：ROCm fused MLA decode RoPE。
- `#21531`：`dsv3_router_gemm` JIT 迁移。
- `#22268`：DeepSeek MLA LoRA adapter 旁路。
- `#22774`：MUSA backend，已于 `2026-04-24T01:59:51Z` 合入；diff 在 DeepSeek MHA/MLA 中加入 `_is_musa` 分支、MUSA FP8 MLA 权重 BF16 fallback，以及 shared expert fusion 的 MUSA capability 检查。
- `#22938`：MLA refactor 后 MI300X DeepSeek path 恢复。
- `#23195`：量化层 `.weight` 访问保护。
- `#23257`：CuteDSL EP + DP attention double reduce。
- `#23336`：spec v2 adaptive speculative decoding。
