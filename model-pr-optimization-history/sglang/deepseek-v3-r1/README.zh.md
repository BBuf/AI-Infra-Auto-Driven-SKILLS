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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3 / R1`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2024-12-26 | [#2601](https://github.com/sgl-project/sglang/pull/2601) | merged | [Feature, Hardware] Enable DeepseekV3 on AMD GPUs | attention/backend, MoE/router, kernel | `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` |
| 2024-12-30 | [#2667](https://github.com/sgl-project/sglang/pull/2667) | merged | AMD DeepSeek_V3 FP8 Numerical fix | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-02-05 | [#3314](https://github.com/sgl-project/sglang/pull/3314) | merged | Feature/docs deepseek usage and add multi-node | misc |  |
| 2025-02-12 | [#3522](https://github.com/sgl-project/sglang/pull/3522) | merged | refine deepseek_v3 launch server doc | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-02-14 | [#3582](https://github.com/sgl-project/sglang/pull/3582) | merged | Support NextN (MTP) speculative decoding for DeepSeek-V3/R1 | model wrapper, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py` |
| 2025-02-26 | [#3893](https://github.com/sgl-project/sglang/pull/3893) | merged | add deepgemm and sglang fp8 block-wise gemm benchmark | quantization, kernel, tests/benchmarks | `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md` |
| 2025-03-05 | [#4079](https://github.com/sgl-project/sglang/pull/4079) | merged | add INT8 example into dsv3 README | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-03-07 | [#4165](https://github.com/sgl-project/sglang/pull/4165) | merged | DeepGemm integrate to sgl-kernel | kernel, tests/benchmarks | `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh` |
| 2025-03-08 | [#4199](https://github.com/sgl-project/sglang/pull/4199) | merged | linear support deepgemm | quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py` |
| 2025-03-09 | [#4218](https://github.com/sgl-project/sglang/pull/4218) | merged | Support nextn for flashinfer mla attention backend | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-03-16 | [#4472](https://github.com/sgl-project/sglang/pull/4472) | merged | Support FlashMLA backend | attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-03-17 | [#4514](https://github.com/sgl-project/sglang/pull/4514) | merged | Support FlashMLA backend cuda graph | attention/backend | `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2025-03-18 | [#4530](https://github.com/sgl-project/sglang/pull/4530) | merged | Add deepseek style fused moe group gate selection kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py` |
| 2025-03-20 | [#4613](https://github.com/sgl-project/sglang/pull/4613) | merged | Set deepgemm to the default value in the hopper architecture. | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` |
| 2025-03-20 | [#4631](https://github.com/sgl-project/sglang/pull/4631) | merged | [ROCm] Enable MTP (NextN) on AMD GPU | attention/backend, kernel, tests/benchmarks | `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py` |
| 2025-03-27 | [#4831](https://github.com/sgl-project/sglang/pull/4831) | merged | [Feature] Support FA3 backend for MLA | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-05 | [#5086](https://github.com/sgl-project/sglang/pull/5086) | merged | reduce moe_align_block_size_kernel small batch mode overhead | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| 2025-04-07 | [#5113](https://github.com/sgl-project/sglang/pull/5113) | merged | Support MHA with chunked prefix cache for DeepSeek chunked prefill | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2025-04-09 | [#5210](https://github.com/sgl-project/sglang/pull/5210) | merged | feat: use fa3 mla by default on hopper | attention/backend, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py` |
| 2025-04-11 | [#5263](https://github.com/sgl-project/sglang/pull/5263) | merged | [Fix] Turn off DeepGEMM by default | quantization, kernel, docs/config | `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2025-04-12 | [#5310](https://github.com/sgl-project/sglang/pull/5310) | merged | fix: use deepgemm only on hopper | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py` |
| 2025-04-14 | [#5371](https://github.com/sgl-project/sglang/pull/5371) | merged | apply fused moe gate in ds v3/r1 | MoE/router | `python/sglang/srt/layers/moe/topk.py` |
| 2025-04-14 | [#5381](https://github.com/sgl-project/sglang/pull/5381) | merged | kernel: support slightly faster merge_state_v2 cuda kernel | attention/backend, kernel, tests/benchmarks | `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py` |
| 2025-04-14 | [#5385](https://github.com/sgl-project/sglang/pull/5385) | merged | Apply deepseek cuda rope | misc | `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-04-14 | [#5390](https://github.com/sgl-project/sglang/pull/5390) | merged | Add Cutlass MLA attention backend | attention/backend, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2025-04-15 | [#5432](https://github.com/sgl-project/sglang/pull/5432) | merged | [perf] introduce deep gemm group_gemm_masked as bmm | model wrapper, quantization, kernel, tests/benchmarks | `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-16 | [#5473](https://github.com/sgl-project/sglang/pull/5473) | merged | use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-04-19 | [#5549](https://github.com/sgl-project/sglang/pull/5549) | merged | Remove one kernel in per_tensor_quant_mla_fp8 | model wrapper, quantization, kernel | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` |
| 2025-04-20 | [#5571](https://github.com/sgl-project/sglang/pull/5571) | merged | enable DeepSeek V3 shared_experts_fusion in sm90 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-20 | [#5578](https://github.com/sgl-project/sglang/pull/5578) | merged | Remove extra copy in deepseek forward absorb | model wrapper, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-04-22 | [#5619](https://github.com/sgl-project/sglang/pull/5619) | merged | Fuse q_a_proj and kv_a_proj for DeepSeek models | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-22 | [#5628](https://github.com/sgl-project/sglang/pull/5628) | merged | Turn on DeepGemm By Default and Update Doc | quantization, docs/config | `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py` |
| 2025-04-24 | [#5707](https://github.com/sgl-project/sglang/pull/5707) | merged | [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1 | model wrapper | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-04-24 | [#5716](https://github.com/sgl-project/sglang/pull/5716) | merged | perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` |
| 2025-04-25 | [#5740](https://github.com/sgl-project/sglang/pull/5740) | merged | update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions. | MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` |
| 2025-04-25 | [#5748](https://github.com/sgl-project/sglang/pull/5748) | merged | Fuse MLA set kv cache kernel | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py` |
| 2025-04-27 | [#5793](https://github.com/sgl-project/sglang/pull/5793) | merged | Auto set draft model path for MTP | model wrapper, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` |
| 2025-05-01 | [#5952](https://github.com/sgl-project/sglang/pull/5952) | merged | Update ci test and doc for MTP api change | attention/backend, tests/benchmarks, docs/config | `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md` |
| 2025-05-02 | [#5977](https://github.com/sgl-project/sglang/pull/5977) | merged | Overlap qk norm with two streams | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2025-05-05 | [#6034](https://github.com/sgl-project/sglang/pull/6034) | merged | Update doc for MLA attention backends | docs/config | `docs/references/deepseek.md`, `docs/backend/server_arguments.md` |
| 2025-05-07 | [#6081](https://github.com/sgl-project/sglang/pull/6081) | merged | feat: mtp support dp-attention | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks | `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` |
| 2025-05-08 | [#6109](https://github.com/sgl-project/sglang/pull/6109) | merged | [Feat] Support FlashMLA backend with MTP and FP8 KV cache | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py` |
| 2025-05-09 | [#6151](https://github.com/sgl-project/sglang/pull/6151) | closed | [Feat] optimize Qwen3 on H20 by hybrid Attention Backend | kernel, scheduler/runtime | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py` |
| 2025-05-12 | [#6220](https://github.com/sgl-project/sglang/pull/6220) | merged | Fuse routed scaling factor in topk_reduce kernel | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-05 | [#6890](https://github.com/sgl-project/sglang/pull/6890) | merged | Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa | quantization | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2025-06-08 | [#6970](https://github.com/sgl-project/sglang/pull/6970) | merged | Fuse routed scaling factor in deepseek | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-06-13 | [#7146](https://github.com/sgl-project/sglang/pull/7146) | merged | Support new DeepGEMM format in per token group quant | quantization, kernel, tests/benchmarks | `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h` |
| 2025-06-13 | [#7150](https://github.com/sgl-project/sglang/pull/7150) | merged | Refactor DeepGEMM integration | model wrapper, MoE/router, quantization, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-13 | [#7155](https://github.com/sgl-project/sglang/pull/7155) | merged | Support new DeepGEMM format in per token group quant (part 2: srt) | quantization, kernel | `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py` |
| 2025-06-13 | [#7156](https://github.com/sgl-project/sglang/pull/7156) | merged | Re-quantize DeepSeek model weights to support DeepGEMM new input format | model wrapper, quantization | `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py` |
| 2025-06-14 | [#7172](https://github.com/sgl-project/sglang/pull/7172) | merged | Support new DeepGEMM | model wrapper, MoE/router, quantization, kernel, docs/config | `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-06-20 | [#7376](https://github.com/sgl-project/sglang/pull/7376) | merged | Fix MTP with Deepseek R1 Fp4 | model wrapper, MoE/router, kernel | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-07-04 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` |
| 2025-07-17 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config | `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` |
| 2025-07-22 | [#8247](https://github.com/sgl-project/sglang/pull/8247) | merged | [1/N]Support DeepSeek-R1 w4a8 normal deepep | MoE/router, quantization, tests/benchmarks | `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-07-28 | [#8464](https://github.com/sgl-project/sglang/pull/8464) | merged | [2/N]Support DeepSeek-R1 w4a8 low latency deepep | MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` |
| 2025-09-04 | [#10027](https://github.com/sgl-project/sglang/pull/10027) | merged | [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-09-12 | [#10361](https://github.com/sgl-project/sglang/pull/10361) | merged | Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile | attention/backend, multimodal/processor | `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py` |
| 2025-10-12 | [#11512](https://github.com/sgl-project/sglang/pull/11512) | merged | Update DeepSeek-R1-FP4 default config on blackwell | misc | `python/sglang/srt/server_args.py` |
| 2025-10-16 | [#11708](https://github.com/sgl-project/sglang/pull/11708) | merged | Support running FP4 Deepseek on SM120. | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py` |
| 2025-10-23 | [#12000](https://github.com/sgl-project/sglang/pull/12000) | merged | [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU | model wrapper | `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-24 | [#12057](https://github.com/sgl-project/sglang/pull/12057) | merged | [doc] add example of using w4fp8 for Deepseek | tests/benchmarks | `benchmark/deepseek_v3/README.md` |
| 2025-11-06 | [#12778](https://github.com/sgl-project/sglang/pull/12778) | merged | Update dsv3 quantization auto setting for sm100 | misc | `python/sglang/srt/server_args.py` |
| 2025-11-09 | [#12921](https://github.com/sgl-project/sglang/pull/12921) | merged | [perf]optimize w4afp8 kernel on deepseek-v3-0324 | MoE/router, quantization, kernel | `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py` |
| 2025-11-19 | [#13548](https://github.com/sgl-project/sglang/pull/13548) | merged | [Fix] Fix DeepSeek V3 MTP on B200 | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2025-11-30 | [#14162](https://github.com/sgl-project/sglang/pull/14162) | merged | DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-12-11 | [#14897](https://github.com/sgl-project/sglang/pull/14897) | merged | Fix dsv3 dp accuracy issue when using bf16-kv | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2025-12-17 | [#15304](https://github.com/sgl-project/sglang/pull/15304) | merged | Fix the accuracy issue when running mxfp4 dsv3 model and enable ep | MoE/router, quantization | `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py` |
| 2025-12-18 | [#15347](https://github.com/sgl-project/sglang/pull/15347) | merged | Use dsv3 optimized routing `fused_topk_deepseek` instead of `moe_fused_gate` | MoE/router, kernel, tests/benchmarks | `test/registered/kernels/test_fused_topk_deepseek.py`, `python/sglang/srt/layers/moe/topk.py`, `test/srt/test_deepseek_v3_mtp.py` |
| 2025-12-20 | [#15531](https://github.com/sgl-project/sglang/pull/15531) | merged | Support piecewise cuda graph for dsv3 fp4 | model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-07 | [#16649](https://github.com/sgl-project/sglang/pull/16649) | merged | [Refactor] Split out deepseek v2 weight loader function into mixin | model wrapper | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py` |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-01-16 | [#17178](https://github.com/sgl-project/sglang/pull/17178) | merged | Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py | tests/benchmarks | `python/sglang/test/run_eval.py` |
| 2026-01-25 | [#17707](https://github.com/sgl-project/sglang/pull/17707) | merged | Add dsv3 router gemm benchmark on blackwell | model wrapper, MoE/router, kernel, tests/benchmarks | `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-26 | [#17744](https://github.com/sgl-project/sglang/pull/17744) | merged | Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization | model wrapper | `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` |
| 2026-02-04 | [#18242](https://github.com/sgl-project/sglang/pull/18242) | merged | [ROCm] Optimize Deepseek R1 on MI300X | model wrapper, attention/backend, quantization | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-02-08 | [#18451](https://github.com/sgl-project/sglang/pull/18451) | merged | [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256. | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-09 | [#18461](https://github.com/sgl-project/sglang/pull/18461) | merged | [Intel GPU] Enable DeepSeek R1 inference on XPU | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py` |
| 2026-02-11 | [#18607](https://github.com/sgl-project/sglang/pull/18607) | merged | [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile` |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-02-26 | [#19425](https://github.com/sgl-project/sglang/pull/19425) | merged | [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4 | model wrapper, quantization | `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py` |
| 2026-03-04 | [#19834](https://github.com/sgl-project/sglang/pull/19834) | merged | [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek) | quantization, tests/benchmarks | `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` |
| 2026-03-04 | [#19843](https://github.com/sgl-project/sglang/pull/19843) | merged | [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-18 | [#20841](https://github.com/sgl-project/sglang/pull/20841) | merged | Fix gpu-fault issue when run deepseek-r1 and enable dp | attention/backend | `python/sglang/srt/layers/attention/aiter_backend.py` |
| 2026-03-24 | [#21280](https://github.com/sgl-project/sglang/pull/21280) | merged | [RL] Support mxfp8 DeepSeek V3 | attention/backend, MoE/router, quantization, kernel, scheduler/runtime | `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2026-03-27 | [#21529](https://github.com/sgl-project/sglang/pull/21529) | open | Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm | model wrapper, MoE/router, quantization, kernel, tests/benchmarks | `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py` |
| 2026-03-27 | [#21531](https://github.com/sgl-project/sglang/pull/21531) | open | [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel | model wrapper, MoE/router, kernel, tests/benchmarks | `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py` |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | kernel, scheduler/runtime, tests/benchmarks, docs/config | `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-03-31 | [#21719](https://github.com/sgl-project/sglang/pull/21719) | merged | Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication" | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | kernel, scheduler/runtime, tests/benchmarks | `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-04-07 | [#22268](https://github.com/sgl-project/sglang/pull/22268) | open | [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3 | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-04-08 | [#22316](https://github.com/sgl-project/sglang/pull/22316) | merged | [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication | MoE/router, quantization, kernel | `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2026-04-08 | [#22323](https://github.com/sgl-project/sglang/pull/22323) | merged | [Lora] Lora quat info re-factor and support deepseekv3 mla lora | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py` |
| 2026-04-16 | [#22933](https://github.com/sgl-project/sglang/pull/22933) | merged | [CPU] expand the interface of shared_expert without scaling factor | MoE/router, quantization, kernel, tests/benchmarks | `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp` |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | scheduler/runtime, tests/benchmarks, docs/config | `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| 2026-04-20 | [#23195](https://github.com/sgl-project/sglang/pull/23195) | open | [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors | model wrapper, attention/backend, tests/benchmarks | `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` |
| 2026-04-20 | [#23257](https://github.com/sgl-project/sglang/pull/23257) | open | Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention | model wrapper, attention/backend, MoE/router, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | multimodal/processor, scheduler/runtime | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |

### 逐 PR 代码 diff 阅读记录

### PR #2601 - [Feature, Hardware] Enable DeepseekV3 on AMD GPUs

- 链接：https://github.com/sgl-project/sglang/pull/2601
- 状态/时间：`merged`，created 2024-12-26, merged 2025-01-03；作者 `BruceXcluding`。
- 代码 diff 已读范围：`2` 个文件，`+9/-5`；代码面：attention/backend, MoE/router, kernel；关键词：triton, attention, config, expert, fp8, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +5/-5 (10 lines); hunk: def invoke_fused_moe_kernel(; def get_default_config(; 符号: invoke_fused_moe_kernel, get_default_config, get_default_config, get_default_config
  - `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` modified +4/-0 (4 lines); hunk: def _decode_grouped_att_m_fwd(; 符号: _decode_grouped_att_m_fwd
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py`；patch 关键词为 triton, attention, config, expert, fp8, moe。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #2667 - AMD DeepSeek_V3 FP8 Numerical fix

- 链接：https://github.com/sgl-project/sglang/pull/2667
- 状态/时间：`merged`，created 2024-12-30, merged 2024-12-30；作者 `HaiShaw`。
- 代码 diff 已读范围：`1` 个文件，`+34/-7`；代码面：model wrapper；关键词：attention, config, flash, fp8, kv, lora, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-7 (41 lines); hunk: from sglang.srt.layers.quantization.fp8_utils import (; from sglang.srt.managers.schedule_batch import global_server_args_dict; 符号: forward_absorb, forward_absorb, load_weights, load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, config, flash, fp8, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #3314 - Feature/docs deepseek usage and add multi-node

- 链接：https://github.com/sgl-project/sglang/pull/3314
- 状态/时间：`merged`，created 2025-02-05, merged 2025-02-07；作者 `lycanlancelot`。
- 代码 diff 已读范围：`0` 个文件，`+0/-0`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - GitHub 未返回文件级 patch。
- 支持/优化点判断：该 PR 的实际 diff 主要落在 未返回 patch 文件；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 未返回 patch 文件 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #3522 - refine deepseek_v3 launch server doc

- 链接：https://github.com/sgl-project/sglang/pull/3522
- 状态/时间：`merged`，created 2025-02-12, merged 2025-02-12；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+7/-0`；代码面：tests/benchmarks；关键词：attention, benchmark, config, doc, fp8, kv, mla。
- 代码 diff 细节：
  - `benchmark/deepseek_v3/README.md` modified +7/-0 (7 lines); hunk: python3 -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-r; python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --tp 16 -
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/deepseek_v3/README.md`；patch 关键词为 attention, benchmark, config, doc, fp8, kv。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/deepseek_v3/README.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #3582 - Support NextN (MTP) speculative decoding for DeepSeek-V3/R1

- 链接：https://github.com/sgl-project/sglang/pull/3582
- 状态/时间：`merged`，created 2025-02-14, merged 2025-02-14；作者 `ispobock`。
- 代码 diff 已读范围：`7` 个文件，`+437/-7`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：spec, config, eagle, kv, attention, cuda, expert, mla, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` added +295/-0 (295 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN
  - `scripts/export_deepseek_nextn.py` added +113/-0 (113 lines); hunk: +"""; 符号: get_nexn_layer_id, update_and_save_config, copy_non_safetensors_files, export_nextn_layer_parameters
  - `python/sglang/srt/speculative/spec_info.py` modified +11/-1 (12 lines); hunk: class SpeculativeAlgorithm(IntEnum):; 符号: SpeculativeAlgorithm, is_none, is_eagle, is_nextn
  - `python/sglang/srt/server_args.py` modified +6/-3 (9 lines); hunk: def __post_init__(self):; def add_cli_args(parser: argparse.ArgumentParser):; 符号: __post_init__, add_cli_args
  - `python/sglang/srt/speculative/eagle_worker.py` modified +7/-2 (9 lines); hunk: fast_topk,; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py`；patch 关键词为 spec, config, eagle, kv, attention, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py`, `scripts/export_deepseek_nextn.py`, `python/sglang/srt/speculative/spec_info.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #3893 - add deepgemm and sglang fp8 block-wise gemm benchmark

- 链接：https://github.com/sgl-project/sglang/pull/3893
- 状态/时间：`merged`，created 2025-02-26, merged 2025-03-02；作者 `BBuf`。
- 代码 diff 已读范围：`2` 个文件，`+320/-0`；代码面：quantization, kernel, tests/benchmarks；关键词：benchmark, fp8, triton, config, cuda, quant, spec, test。
- 代码 diff 细节：
  - `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py` added +314/-0 (314 lines); hunk: +import itertools; 符号: per_token_cast_to_fp8, per_block_cast_to_fp8, fp8_gemm_deepgemm, fp8_gemm_sglang
  - `benchmark/kernels/deepseek/README.md` added +6/-0 (6 lines); hunk: +## DeepSeek kernels benchmark
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md`；patch 关键词为 benchmark, fp8, triton, config, cuda, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/deepseek/benchmark_deepgemm_fp8_gemm.py`, `benchmark/kernels/deepseek/README.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4079 - add INT8 example into dsv3 README

- 链接：https://github.com/sgl-project/sglang/pull/4079
- 状态/时间：`merged`，created 2025-03-05, merged 2025-03-13；作者 `laixinn`。
- 代码 diff 已读范围：`1` 个文件，`+16/-2`；代码面：tests/benchmarks；关键词：awq, benchmark, quant。
- 代码 diff 细节：
  - `benchmark/deepseek_v3/README.md` modified +16/-2 (18 lines); hunk: AWQ does not support BF16, so add the `--dtype half` flag if AWQ is used for qua
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/deepseek_v3/README.md`；patch 关键词为 awq, benchmark, quant。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/deepseek_v3/README.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4165 - DeepGemm integrate to sgl-kernel

- 链接：https://github.com/sgl-project/sglang/pull/4165
- 状态/时间：`merged`，created 2025-03-07, merged 2025-03-10；作者 `laixinn`。
- 代码 diff 已读范围：`6` 个文件，`+324/-5`；代码面：kernel, tests/benchmarks；关键词：cuda, flash, cache, doc, fp8, test。
- 代码 diff 细节：
  - `sgl-kernel/tests/test_deep_gemm.py` added +263/-0 (263 lines); hunk: +import os; 符号: per_token_cast_to_fp8, per_block_cast_to_fp8, construct, construct_grouped
  - `sgl-kernel/setup.py` modified +52/-1 (53 lines); hunk: # ==============================================================================; def _get_version():; 符号: _get_version, _get_version, CustomBuildPy, run
  - `sgl-kernel/build.sh` modified +4/-3 (7 lines); hunk: else; docker run --rm \
  - `.gitmodules` modified +3/-0 (3 lines); hunk: [submodule "sgl-kernel/3rdparty/flashinfer"]
  - `sgl-kernel/pyproject.toml` modified +1/-1 (2 lines); hunk: [build-system]
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh`；patch 关键词为 cuda, flash, cache, doc, fp8, test。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/tests/test_deep_gemm.py`, `sgl-kernel/setup.py`, `sgl-kernel/build.sh` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4199 - linear support deepgemm

- 链接：https://github.com/sgl-project/sglang/pull/4199
- 状态/时间：`merged`，created 2025-03-08, merged 2025-03-11；作者 `sleepcoo`。
- 代码 diff 已读范围：`3` 个文件，`+76/-44`；代码面：quantization, kernel, tests/benchmarks；关键词：fp8, quant, cuda, test, config, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +36/-28 (64 lines); hunk: _is_cuda = torch.cuda.is_available() and torch.version.cuda; def grid(META):; 符号: _per_token_group_quant_fp8, grid
  - `python/sglang/test/test_block_fp8.py` modified +39/-15 (54 lines); hunk: import itertools; w8a8_block_fp8_matmul,; 符号: native_per_token_group_quant_fp8, native_w8a8_block_fp8_matmul, TestW8A8BlockFP8Matmul, setUpClass
  - `test/srt/test_fp8_kernel.py` modified +1/-1 (2 lines); hunk: def setUpClass(cls):; 符号: setUpClass, _make_A
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py`；patch 关键词为 fp8, quant, cuda, test, config, triton。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_block_fp8.py`, `test/srt/test_fp8_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4218 - Support nextn for flashinfer mla attention backend

- 链接：https://github.com/sgl-project/sglang/pull/4218
- 状态/时间：`merged`，created 2025-03-09, merged 2025-03-09；作者 `Fridge003`。
- 代码 diff 已读范围：`5` 个文件，`+393/-58`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config；关键词：flash, mla, eagle, spec, topk, cache, attention, cuda, kv, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +317/-57 (374 lines); hunk: from dataclasses import dataclass; from sglang.srt.layers.dp_attention import get_attention_tp_size; 符号: FlashInferMLAAttnBackend, __init__, __init__, init_forward_metadata
  - `test/srt/test_mla_flashinfer.py` modified +63/-0 (63 lines); hunk: import unittest; def test_gsm8k(self):; 符号: test_gsm8k, TestFlashinferMLAMTP, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +10/-0 (10 lines); hunk: def init_attention_backend(self):; 符号: init_attention_backend
  - `docs/references/deepseek.md` modified +1/-1 (2 lines); hunk: Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/be
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-0 (2 lines); hunk: def no_absorb() -> bool:; 符号: no_absorb
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py`；patch 关键词为 flash, mla, eagle, spec, topk, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`, `test/srt/test_mla_flashinfer.py`, `python/sglang/srt/speculative/eagle_worker.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4472 - Support FlashMLA backend

- 链接：https://github.com/sgl-project/sglang/pull/4472
- 状态/时间：`merged`，created 2025-03-16, merged 2025-03-16；作者 `sleepcoo`。
- 代码 diff 已读范围：`6` 个文件，`+209/-1`；代码面：attention/backend, scheduler/runtime, tests/benchmarks；关键词：flash, mla, cache, spec, attention, triton, cuda, kv, config, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashmla_backend.py` added +128/-0 (128 lines); hunk: +from __future__ import annotations; 符号: FlashMLABackend, __init__, forward_decode
  - `python/sglang/srt/layers/attention/utils.py` modified +54/-0 (54 lines); hunk: def create_flashinfer_kv_indices_triton(; def create_flashinfer_kv_indices_triton(; 符号: create_flashinfer_kv_indices_triton, create_flashinfer_kv_indices_triton, create_flashmla_kv_indices_triton
  - `python/sglang/srt/model_executor/model_runner.py` modified +8/-0 (8 lines); hunk: def __init__(; def model_specific_adjustment(self):; 符号: __init__, model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunk: class ServerArgs:; def __post_init__(self):; 符号: ServerArgs:, __post_init__, add_cli_args
  - `python/sglang/srt/managers/schedule_batch.py` modified +5/-1 (6 lines); hunk: "speculative_accept_threshold_single": ServerArgs.speculative_accept_threshold_single,; def merge_batch(self, other: "ScheduleBatch"):; 符号: merge_batch, get_model_worker_batch
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 flash, mla, cache, spec, attention, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/layers/attention/utils.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4514 - Support FlashMLA backend cuda graph

- 链接：https://github.com/sgl-project/sglang/pull/4514
- 状态/时间：`merged`，created 2025-03-17, merged 2025-03-19；作者 `sleepcoo`。
- 代码 diff 已读范围：`3` 个文件，`+188/-32`；代码面：attention/backend；关键词：flash, mla, attention, cuda, kv, triton, cache, config, eagle, lora。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +184/-30 (214 lines); hunk: from __future__ import annotations; from sglang.srt.layers.radix_attention import RadixAttention; 符号: FlashMLADecodeMetadata:, __init__, FlashMLABackend, __init__
  - `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunk: def __post_init__(self):; 符号: __post_init__
  - `python/sglang/srt/layers/attention/utils.py` modified +0/-1 (1 lines); hunk: def create_flashmla_kv_indices_triton(; 符号: create_flashmla_kv_indices_triton
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py`；patch 关键词为 flash, mla, attention, cuda, kv, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashmla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4530 - Add deepseek style fused moe group gate selection kernel

- 链接：https://github.com/sgl-project/sglang/pull/4530
- 状态/时间：`merged`，created 2025-03-18, merged 2025-03-29；作者 `qingquansong`。
- 代码 diff 已读范围：`9` 个文件，`+616/-1`；代码面：MoE/router, kernel, tests/benchmarks；关键词：moe, topk, expert, cuda, quant, spec, config, fp8, test, benchmark。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/moe_fused_gate.cu` added +447/-0 (447 lines); hunk: +#include <ATen/cuda/CUDAContext.h>; 符号: versions:, int, int, int
  - `sgl-kernel/benchmark/bench_moe_fused_gate.py` added +74/-0 (74 lines); hunk: +import itertools; 符号: biased_grouped_topk_org, biased_grouped_topk_org_kernel, benchmark
  - `sgl-kernel/tests/test_moe_fused_gate.py` added +72/-0 (72 lines); hunk: +import pytest; 符号: test_moe_fused_gate_combined
  - `sgl-kernel/python/sgl_kernel/moe.py` modified +12/-0 (12 lines); hunk: def topk_softmax(; 符号: topk_softmax, moe_fused_gate
  - `sgl-kernel/csrc/torch_extension.cc` modified +5/-0 (5 lines); hunk: TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py`；patch 关键词为 moe, topk, expert, cuda, quant, spec。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/moe_fused_gate.cu`, `sgl-kernel/benchmark/bench_moe_fused_gate.py`, `sgl-kernel/tests/test_moe_fused_gate.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4613 - Set deepgemm to the default value in the hopper architecture.

- 链接：https://github.com/sgl-project/sglang/pull/4613
- 状态/时间：`merged`，created 2025-03-20, merged 2025-03-21；作者 `sleepcoo`。
- 代码 diff 已读范围：`2` 个文件，`+16/-3`；代码面：quantization, kernel；关键词：cuda, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-3 (12 lines); hunk: direct_register_custom_op,; import deep_gemm # `pip install "sgl-kernel>=0.0.4.post3"`; 符号: grid
  - `python/sglang/srt/utils.py` modified +7/-0 (7 lines); hunk: def get_amdgpu_memory_capacity():; 符号: get_amdgpu_memory_capacity, get_device_sm, get_nvgpu_memory_capacity
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`；patch 关键词为 cuda, fp8, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4631 - [ROCm] Enable MTP (NextN) on AMD GPU

- 链接：https://github.com/sgl-project/sglang/pull/4631
- 状态/时间：`merged`，created 2025-03-20, merged 2025-03-24；作者 `alexsun07`。
- 代码 diff 已读范围：`7` 个文件，`+43/-4`；代码面：attention/backend, kernel, tests/benchmarks；关键词：cuda, spec, eagle, topk, cache, expert, kv, mla, moe, test。
- 代码 diff 细节：
  - `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h` added +20/-0 (20 lines); hunk: +#include <torch/library.h>
  - `sgl-kernel/csrc/torch_extension_rocm.cc` modified +12/-0 (12 lines); hunk: TORCH_LIBRARY_EXPAND(sgl_kernel, m) {
  - `python/sglang/srt/speculative/build_eagle_tree.py` modified +2/-2 (4 lines); hunk: import torch
  - `python/sglang/srt/speculative/eagle_utils.py` modified +3/-1 (4 lines); hunk: from sglang.srt.mem_cache.memory_pool import TokenToKVPoolAllocator; tree_speculative_sampling_target_only,
  - `sgl-kernel/csrc/speculative/eagle_utils.cu` modified +4/-0 (4 lines); hunk: #include <ATen/ATen.h>
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py`；patch 关键词为 cuda, spec, eagle, topk, cache, expert。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/speculative/pytorch_extension_utils_rocm.h`, `sgl-kernel/csrc/torch_extension_rocm.cc`, `python/sglang/srt/speculative/build_eagle_tree.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #4831 - [Feature] Support FA3 backend for MLA

- 链接：https://github.com/sgl-project/sglang/pull/4831
- 状态/时间：`merged`，created 2025-03-27, merged 2025-03-29；作者 `Fridge003`。
- 代码 diff 已读范围：`3` 个文件，`+180/-74`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, flash, mla, triton, cache, config, cuda, fp8, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +171/-73 (244 lines); hunk: import torch; def __init__(; 符号: __init__, init_forward_metadata, forward_extend, forward_extend
  - `python/sglang/srt/model_executor/model_runner.py` modified +5/-1 (6 lines); hunk: def model_specific_adjustment(self):; def init_attention_backend(self):; 符号: model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/models/deepseek_v2.py` modified +4/-0 (4 lines); hunk: def __init__(; def no_absorb(self, forward_batch: ForwardBatch) -> bool:; 符号: __init__, no_absorb, no_absorb
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, flash, mla, triton, cache, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5086 - reduce moe_align_block_size_kernel small batch mode overhead

- 链接：https://github.com/sgl-project/sglang/pull/5086
- 状态/时间：`merged`，created 2025-04-05, merged 2025-04-10；作者 `BBuf`。
- 代码 diff 已读范围：`4` 个文件，`+143/-56`；代码面：MoE/router, kernel, tests/benchmarks；关键词：expert, moe, topk, triton, test, benchmark, config, quant。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/moe_align_kernel.cu` modified +111/-44 (155 lines); hunk: __global__ void moe_align_block_size_kernel(; __global__ void moe_align_block_size_kernel(; 符号: void, void, void
  - `sgl-kernel/benchmark/bench_moe_align_block_size.py` modified +31/-10 (41 lines); hunk: def calculate_diff(num_tokens, num_experts=256, block_size=128, topk=8):; def benchmark(num_tokens, num_experts, topk, provider):; 符号: calculate_diff, benchmark, sgl_moe_align_block_size_with_empty, benchmark
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-1 (2 lines); hunk: def moe_align_block_size(; 符号: moe_align_block_size
  - `sgl-kernel/tests/test_moe_align.py` modified +0/-1 (1 lines); hunk: def moe_align_block_size_triton(; 符号: moe_align_block_size_triton, test_moe_align_block_size_compare_implementations
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`；patch 关键词为 expert, moe, topk, triton, test, benchmark。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/moe_align_kernel.cu`, `sgl-kernel/benchmark/bench_moe_align_block_size.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5113 - Support MHA with chunked prefix cache for DeepSeek chunked prefill

- 链接：https://github.com/sgl-project/sglang/pull/5113
- 状态/时间：`merged`，created 2025-04-07, merged 2025-04-16；作者 `Fridge003`。
- 代码 diff 已读范围：`10` 个文件，`+734/-46`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, mla, attention, cuda, flash, kv, expert, moe, triton, deepep。
- 代码 diff 细节：
  - `python/sglang/test/attention/test_prefix_chunk_info.py` added +224/-0 (224 lines); hunk: +import unittest; 符号: MockForwardBatch, __init__, get_max_chunk_capacity, MockReqToTokenPool:
  - `python/sglang/srt/models/deepseek_v2.py` modified +174/-9 (183 lines); hunk: import logging; _is_cuda = is_cuda(); 符号: AttnForwardMethod, DeepseekV2MLP, __init__, __init__
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +181/-0 (181 lines); hunk: class ForwardBatch:; def _compute_mrope_positions(; 符号: ForwardBatch:, _compute_mrope_positions, get_max_chunk_capacity, set_prefix_chunk_idx
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +80/-34 (114 lines); hunk: from sglang.srt.layers.radix_attention import RadixAttention; def forward_extend(; 符号: forward_extend, forward_decode
  - `test/srt/test_fa3.py` modified +53/-2 (55 lines); hunk: from sglang.srt.utils import get_device_sm, kill_process_tree; """; 符号: test_gsm8k, TestFlashAttention3MLASpeculativeDecode, get_server_args, test_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py`；patch 关键词为 cache, mla, attention, cuda, flash, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/attention/test_prefix_chunk_info.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/model_executor/forward_batch_info.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5210 - feat: use fa3 mla by default on hopper

- 链接：https://github.com/sgl-project/sglang/pull/5210
- 状态/时间：`merged`，created 2025-04-09, merged 2025-04-12；作者 `zhyncs`。
- 代码 diff 已读范围：`3` 个文件，`+42/-11`；代码面：attention/backend, scheduler/runtime；关键词：cuda, attention, cache, flash, fp8, kv, mla, spec, topk, config。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunk: is_cuda,; def model_specific_adjustment(self):; 符号: model_specific_adjustment, model_specific_adjustment, init_attention_backend
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +12/-7 (19 lines); hunk: def init_forward_metadata(self, forward_batch: ForwardBatch):; def forward_extend(; 符号: init_forward_metadata, forward_extend, forward_decode, init_forward_metadata_capture_cuda_graph
  - `python/sglang/srt/utils.py` modified +9/-0 (9 lines); hunk: def fast_topk(values, topk, dim):; 符号: fast_topk, is_hopper_with_cuda_12_3
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py`；patch 关键词为 cuda, attention, cache, flash, fp8, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5263 - [Fix] Turn off DeepGEMM by default

- 链接：https://github.com/sgl-project/sglang/pull/5263
- 状态/时间：`merged`，created 2025-04-11, merged 2025-04-15；作者 `Fridge003`。
- 代码 diff 已读范围：`2` 个文件，`+6/-2`；代码面：quantization, kernel, docs/config；关键词：fp8, quant, attention, doc, eagle, spec。
- 代码 diff 细节：
  - `docs/references/deepseek.md` modified +3/-1 (4 lines); hunk: With data parallelism attention enabled, we have achieved up to **1.9x** decoding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +3/-1 (4 lines); hunk: from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_quant_fp8
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py`；patch 关键词为 fp8, quant, attention, doc, eagle, spec。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/fp8_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5310 - fix: use deepgemm only on hopper

- 链接：https://github.com/sgl-project/sglang/pull/5310
- 状态/时间：`merged`，created 2025-04-12, merged 2025-04-12；作者 `zhyncs`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：quantization, kernel；关键词：fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +1/-1 (2 lines); hunk: from sgl_kernel import sgl_per_token_group_quant_fp8, sgl_per_token_quant_fp8
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_kernel.py`；patch 关键词为 fp8, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5371 - apply fused moe gate in ds v3/r1

- 链接：https://github.com/sgl-project/sglang/pull/5371
- 状态/时间：`merged`，created 2025-04-14, merged 2025-04-14；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+37/-16`；代码面：MoE/router；关键词：cuda, expert, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/topk.py` modified +37/-16 (53 lines); hunk: # limitations under the License.; _is_cuda = is_cuda(); 符号: biased_grouped_topk_impl, is_power_of_two, biased_grouped_topk, biased_grouped_topk
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/topk.py`；patch 关键词为 cuda, expert, moe, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/topk.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5381 - kernel: support slightly faster merge_state_v2 cuda kernel

- 链接：https://github.com/sgl-project/sglang/pull/5381
- 状态/时间：`merged`，created 2025-04-14, merged 2025-04-15；作者 `DefTruth`。
- 代码 diff 已读范围：`7` 个文件，`+638/-4`；代码面：attention/backend, kernel, tests/benchmarks；关键词：attention, cuda, mla, cache, kv, triton, fp8, test。
- 代码 diff 细节：
  - `sgl-kernel/tests/test_merge_state_v2.py` added +396/-0 (396 lines); hunk: +from typing import Optional; 符号: merge_state_kernel, merge_state_triton, merge_state_torch, generate_markdown_table
  - `sgl-kernel/csrc/attention/merge_attn_states.cu` added +201/-0 (201 lines); hunk: +#include <ATen/cuda/CUDAContext.h>; 符号: void, uint
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +35/-4 (39 lines); hunk: -from typing import Tuple; def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):; 符号: lightning_attention_decode, merge_state, merge_state_v2, cutlass_mla_decode
  - `sgl-kernel/csrc/common_extension.cc` modified +2/-0 (2 lines); hunk: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-0 (2 lines); hunk: void lightning_attention_decode(
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py`；patch 关键词为 attention, cuda, mla, cache, kv, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/tests/test_merge_state_v2.py`, `sgl-kernel/csrc/attention/merge_attn_states.cu`, `sgl-kernel/python/sgl_kernel/attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5385 - Apply deepseek cuda rope

- 链接：https://github.com/sgl-project/sglang/pull/5385
- 状态/时间：`merged`，created 2025-04-14, merged 2025-04-14；作者 `ispobock`。
- 代码 diff 已读范围：`1` 个文件，`+12/-1`；代码面：misc；关键词：cache, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/rotary_embedding.py` modified +12/-1 (13 lines); hunk: def _compute_cos_sin_cache(self) -> torch.Tensor:; 符号: _compute_cos_sin_cache, forward, forward_hip, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 cache, cuda。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5390 - Add Cutlass MLA attention backend

- 链接：https://github.com/sgl-project/sglang/pull/5390
- 状态/时间：`merged`，created 2025-04-14, merged 2025-04-28；作者 `trevor-m`。
- 代码 diff 已读范围：`7` 个文件，`+305/-3`；代码面：attention/backend, kernel, scheduler/runtime, docs/config；关键词：attention, mla, flash, triton, kv, spec, cache, cuda, config, doc。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/cutlass_mla_backend.py` added +278/-0 (278 lines); hunk: +from __future__ import annotations; 符号: CutlassMLADecodeMetadata:, __init__, CutlassMLABackend, __init__
  - `python/sglang/srt/server_args.py` modified +14/-1 (15 lines); hunk: def __post_init__(self):; def add_cli_args(parser: argparse.ArgumentParser):; 符号: __post_init__, add_cli_args
  - `python/sglang/srt/model_executor/model_runner.py` modified +7/-0 (7 lines); hunk: def model_specific_adjustment(self):; def init_attention_backend(self):; 符号: model_specific_adjustment, init_attention_backend
  - `sgl-kernel/python/sgl_kernel/attention.py` modified +3/-0 (3 lines); hunk: def cutlass_mla_decode(; def cutlass_mla_decode(; 符号: cutlass_mla_decode, cutlass_mla_decode, cutlass_mla_get_workspace_size
  - `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunk: Please consult the documentation below to learn more about the parameters you ma
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 attention, mla, flash, triton, kv, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/cutlass_mla_backend.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5432 - [perf] introduce deep gemm group_gemm_masked as bmm

- 链接：https://github.com/sgl-project/sglang/pull/5432
- 状态/时间：`merged`，created 2025-04-15, merged 2025-04-20；作者 `Alcanderian`。
- 代码 diff 已读范围：`3` 个文件，`+361/-20`；代码面：model wrapper, quantization, kernel, tests/benchmarks；关键词：cuda, fp8, mla, quant, triton, moe, awq, config, expert, flash。
- 代码 diff 细节：
  - `python/sglang/test/test_block_fp8.py` modified +167/-0 (167 lines); hunk: from sglang.srt.layers.activation import SiluAndMul; def test_per_tensor_quant_mla_fp8(self):; 符号: test_per_tensor_quant_mla_fp8, TestPerTokenGroupQuantMlaDeepGemmMaskedFP8, setUpClass, _per_token_group_quant_mla_deep_gemm_masked_fp8
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +108/-4 (112 lines); hunk: fp8_min = -fp8_max; ); 符号: per_tensor_quant_mla_fp8, _per_token_group_quant_mla_deep_gemm_masked_fp8, per_tensor_quant_mla_deep_gemm_masked_fp8, scaled_fp8_quant
  - `python/sglang/srt/models/deepseek_v2.py` modified +86/-16 (102 lines); hunk: from sglang.srt.layers.moe.fused_moe_triton import FusedMoE; _is_cuda = is_cuda(); 符号: __init__, forward_absorb, forward_absorb, post_load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 cuda, fp8, mla, quant, triton, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/test_block_fp8.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5473 - use sglang_per_token_group_quant_fp8 from sgl-kernel instead of trion kernel

- 链接：https://github.com/sgl-project/sglang/pull/5473
- 状态/时间：`merged`，created 2025-04-16, merged 2025-04-18；作者 `strgrb`。
- 代码 diff 已读范围：`2` 个文件，`+25/-6`；代码面：quantization, kernel；关键词：fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +24/-5 (29 lines); hunk: def sglang_per_token_group_quant_fp8(; 符号: sglang_per_token_group_quant_fp8
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-1 (2 lines); hunk: def apply_w8a8_block_fp8_linear(; 符号: apply_w8a8_block_fp8_linear
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 fp8, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5549 - Remove one kernel in per_tensor_quant_mla_fp8

- 链接：https://github.com/sgl-project/sglang/pull/5549
- 状态/时间：`merged`，created 2025-04-19, merged 2025-04-19；作者 `fzyzcjy`。
- 代码 diff 已读范围：`4` 个文件，`+62/-18`；代码面：model wrapper, quantization, kernel；关键词：cuda, fp8, mla, quant, attention, config, deepep, expert, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +32/-9 (41 lines); hunk: from sglang.srt.managers.schedule_batch import global_server_args_dict; class AttnForwardMethod(IntEnum):; 符号: AttnForwardMethod, forward, forward, forward_normal
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +9/-7 (16 lines); hunk: ):; def _per_tensor_quant_mla_fp8_stage2(; 符号: deep_gemm_fp8_fp8_bf16_nt, _per_tensor_quant_mla_fp8_stage2, per_tensor_quant_mla_fp8, per_tensor_quant_mla_fp8
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunk: def is_fa3_default_architecture(hf_config):; 符号: is_fa3_default_architecture, BumpAllocator:, __init__, allocate
  - `python/sglang/srt/models/deepseek_nextn.py` modified +8/-2 (10 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; 符号: forward, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py`；patch 关键词为 cuda, fp8, mla, quant, attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5571 - enable DeepSeek V3 shared_experts_fusion in sm90

- 链接：https://github.com/sgl-project/sglang/pull/5571
- 状态/时间：`merged`，created 2025-04-20, merged 2025-04-20；作者 `BBuf`。
- 代码 diff 已读范围：`1` 个文件，`+12/-0`；代码面：model wrapper；关键词：config, cuda, deepep, expert, fp8, kv, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-0 (12 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, cuda, deepep, expert, fp8, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5578 - Remove extra copy in deepseek forward absorb

- 链接：https://github.com/sgl-project/sglang/pull/5578
- 状态/时间：`merged`，created 2025-04-20, merged 2025-04-22；作者 `ispobock`。
- 代码 diff 已读范围：`3` 个文件，`+18/-21`；代码面：model wrapper, tests/benchmarks；关键词：cache, doc, kv, lora, mla, test。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-13 (22 lines); hunk: def forward_absorb(; def forward_absorb(; 符号: forward_absorb, forward_absorb
  - `.github/workflows/pr-test-amd.yml` modified +7/-7 (14 lines); hunk: jobs:; jobs:
  - `python/sglang/srt/layers/rotary_embedding.py` modified +2/-1 (3 lines); hunk: def forward_native(; def forward_native(; 符号: forward_native, forward_native, Llama3RotaryEmbedding
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 cache, doc, kv, lora, mla, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `.github/workflows/pr-test-amd.yml`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5619 - Fuse q_a_proj and kv_a_proj for DeepSeek models

- 链接：https://github.com/sgl-project/sglang/pull/5619
- 状态/时间：`merged`，created 2025-04-22, merged 2025-04-23；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+78/-25`；代码面：model wrapper；关键词：attention, cache, config, expert, kv, lora, mla, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +78/-25 (103 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, forward_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, config, expert, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5628 - Turn on DeepGemm By Default and Update Doc

- 链接：https://github.com/sgl-project/sglang/pull/5628
- 状态/时间：`merged`，created 2025-04-22, merged 2025-04-22；作者 `Fridge003`。
- 代码 diff 已读范围：`2` 个文件，`+9/-3`；代码面：quantization, docs/config；关键词：quant, attention, doc, eagle, fp8, spec。
- 代码 diff 细节：
  - `docs/references/deepseek.md` modified +8/-2 (10 lines); hunk: With data parallelism attention enabled, we have achieved up to **1.9x** decoding
  - `python/sglang/srt/layers/quantization/deep_gemm.py` modified +1/-1 (2 lines); hunk: sm_version = get_device_sm()
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py`；patch 关键词为 quant, attention, doc, eagle, fp8, spec。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/references/deepseek.md`, `python/sglang/srt/layers/quantization/deep_gemm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5707 - [BugFix] Fix combination of MTP and `--n-share-experts-fusion`with R1

- 链接：https://github.com/sgl-project/sglang/pull/5707
- 状态/时间：`merged`，created 2025-04-24, merged 2025-04-24；作者 `guoyuhong`。
- 代码 diff 已读范围：`2` 个文件，`+68/-15`；代码面：model wrapper；关键词：config, expert, fp8, kv, moe, processor, quant, attention, awq, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +50/-1 (51 lines); hunk: # ==============================================================================; from vllm._custom_ops import awq_dequantize; 符号: DeepseekModelNextN, __init__, __init__, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-14 (32 lines); hunk: def __init__(; def __init__(; 符号: __init__, determine_n_share_experts_fusion, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, expert, fp8, kv, moe, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5716 - perf: update H20 fused_moe_triton kernel config to get higher throughput during prefilling

- 链接：https://github.com/sgl-project/sglang/pull/5716
- 状态/时间：`merged`，created 2025-04-24, merged 2025-04-27；作者 `saltyfish66`。
- 代码 diff 已读范围：`1` 个文件，`+27/-27`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, fp8, moe, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +27/-27 (54 lines); hunk: "BLOCK_SIZE_K": 128,; },
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json`；patch 关键词为 config, fp8, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=272,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5740 - update triton 3.2.0 h200 fused moe triton config and add warning about triton fused_moe_kernel performance degradation due to different Triton versions.

- 链接：https://github.com/sgl-project/sglang/pull/5740
- 状态/时间：`merged`，created 2025-04-25, merged 2025-04-25；作者 `BBuf`。
- 代码 diff 已读范围：`2` 个文件，`+45/-42`；代码面：MoE/router, quantization, kernel, docs/config；关键词：config, moe, triton, benchmark, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json` modified +41/-41 (82 lines); hunk: {; "BLOCK_SIZE_M": 64,
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +4/-1 (5 lines); hunk: def get_moe_configs(; 符号: get_moe_configs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`；patch 关键词为 config, moe, triton, benchmark, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/E=264,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5748 - Fuse MLA set kv cache kernel

- 链接：https://github.com/sgl-project/sglang/pull/5748
- 状态/时间：`merged`，created 2025-04-25, merged 2025-04-27；作者 `ispobock`。
- 代码 diff 已读范围：`4` 个文件，`+100/-9`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, kv, cache, mla, flash, lora, triton。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +87/-0 (87 lines); hunk: import numpy as np; def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):; 符号: copy_two_array, set_mla_kv_buffer_kernel, set_mla_kv_buffer_triton, MLATokenToKVPool
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +6/-4 (10 lines); hunk: def forward_extend(; def forward_extend(; 符号: forward_extend, forward_extend, forward_decode, forward_decode
  - `python/sglang/srt/layers/radix_attention.py` modified +5/-2 (7 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-3 (5 lines); hunk: def forward_absorb(; 符号: forward_absorb
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py`；patch 关键词为 attention, kv, cache, mla, flash, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/layers/radix_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5793 - Auto set draft model path for MTP

- 链接：https://github.com/sgl-project/sglang/pull/5793
- 状态/时间：`merged`，created 2025-04-27, merged 2025-04-29；作者 `ispobock`。
- 代码 diff 已读范围：`6` 个文件，`+115/-287`；代码面：model wrapper, scheduler/runtime, docs/config；关键词：config, kv, cache, quant, spec, awq, cuda, expert, lora, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-257 (258 lines); hunk: def forward(; 符号: forward, load_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-17 (91 lines); hunk: def forward(; def post_load_weights(self):; 符号: forward, post_load_weights, post_load_weights, post_load_weights
  - `python/sglang/srt/server_args.py` modified +21/-11 (32 lines); hunk: import tempfile; def __post_init__(self):; 符号: __post_init__, __post_init__, __call__, auto_choose_speculative_params
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-2 (13 lines); hunk: def profile_max_num_token(self, total_gpu_memory: int):; def init_memory_pool(; 符号: profile_max_num_token, init_memory_pool
  - `python/sglang/srt/configs/model_config.py` modified +7/-0 (7 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`；patch 关键词为 config, kv, cache, quant, spec, awq。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5952 - Update ci test and doc for MTP api change

- 链接：https://github.com/sgl-project/sglang/pull/5952
- 状态/时间：`merged`，created 2025-05-01, merged 2025-05-01；作者 `ispobock`。
- 代码 diff 已读范围：`6` 个文件，`+66/-14`；代码面：attention/backend, tests/benchmarks, docs/config；关键词：spec, eagle, topk, mla, test, flash, kv, attention, cache, config。
- 代码 diff 细节：
  - `test/srt/test_mla_deepseek_v3.py` modified +57/-0 (57 lines); hunk: def test_gsm8k(self):; 符号: test_gsm8k, TestDeepseekV3MTP, setUpClass, tearDownClass
  - `python/sglang/srt/server_args.py` modified +7/-4 (11 lines); hunk: def __post_init__(self):; 符号: __post_init__
  - `docs/references/deepseek.md` modified +2/-4 (6 lines); hunk: The precompilation process typically takes around 10 minutes to complete.
  - `test/srt/test_full_deepseek_v3.py` modified +0/-2 (2 lines); hunk: def setUpClass(cls):; 符号: setUpClass
  - `test/srt/test_mla_flashinfer.py` modified +0/-2 (2 lines); hunk: def setUpClass(cls):; 符号: setUpClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md`；patch 关键词为 spec, eagle, topk, mla, test, flash。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_mla_deepseek_v3.py`, `python/sglang/srt/server_args.py`, `docs/references/deepseek.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #5977 - Overlap qk norm with two streams

- 链接：https://github.com/sgl-project/sglang/pull/5977
- 状态/时间：`merged`，created 2025-05-02, merged 2025-05-02；作者 `ispobock`。
- 代码 diff 已读范围：`1` 个文件，`+26/-6`；代码面：model wrapper；关键词：attention, cache, config, cuda, kv, lora, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-6 (32 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, forward_absorb, forward_absorb
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, config, cuda, kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6034 - Update doc for MLA attention backends

- 链接：https://github.com/sgl-project/sglang/pull/6034
- 状态/时间：`merged`，created 2025-05-05, merged 2025-05-08；作者 `Fridge003`。
- 代码 diff 已读范围：`2` 个文件，`+3/-3`；代码面：docs/config；关键词：attention, cache, doc, flash, kv, mla, spec, triton, config, cuda。
- 代码 diff 细节：
  - `docs/references/deepseek.md` modified +2/-2 (4 lines); hunk: Please refer to [the example](https://github.com/sgl-project/sglang/tree/main/be; Add arguments `--speculative-algorithm`, `--speculative-num-steps`, `--specula
  - `docs/backend/server_arguments.md` modified +1/-1 (2 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/references/deepseek.md`, `docs/backend/server_arguments.md`；patch 关键词为 attention, cache, doc, flash, kv, mla。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/references/deepseek.md`, `docs/backend/server_arguments.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6081 - feat: mtp support dp-attention

- 链接：https://github.com/sgl-project/sglang/pull/6081
- 状态/时间：`merged`，created 2025-05-07, merged 2025-06-17；作者 `u4lr451`。
- 代码 diff 已读范围：`22` 个文件，`+636/-146`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks；关键词：spec, attention, eagle, cuda, config, cache, kv, topk, mla, flash。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/eagle_worker.py` modified +125/-39 (164 lines); hunk: import torch; def draft_tp_context(tp_group: GroupCoordinator):; 符号: draft_tp_context, __init__, forward_batch_speculative_generation, check_forward_draft_extend_after_decode
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +103/-11 (114 lines); hunk: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +97/-12 (109 lines); hunk: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/speculative/eagle_utils.py` modified +74/-4 (78 lines); hunk: from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode; class EagleDraftInput:; 符号: EagleDraftInput:, prepare_for_extend, prepare_for_extend, create_idle_input
  - `test/srt/test_dp_attention.py` modified +72/-0 (72 lines); hunk: import unittest; def test_mgsm_en(self):; 符号: test_mgsm_en, TestDPAttentionDP2TP2DeepseekV3MTP, setUpClass, tearDownClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py`；patch 关键词为 spec, attention, eagle, cuda, config, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py`, `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6109 - [Feat] Support FlashMLA backend with MTP and FP8 KV cache

- 链接：https://github.com/sgl-project/sglang/pull/6109
- 状态/时间：`merged`，created 2025-05-08, merged 2025-05-15；作者 `quinnrong94`。
- 代码 diff 已读范围：`8` 个文件，`+444/-87`；代码面：attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, flash, mla, cache, cuda, eagle, kv, spec, topk, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/flashmla_backend.py` modified +340/-78 (418 lines); hunk: """; # FlashMLA only supports pagesize=64; 符号: __init__, FlashMLABackend, __init__, __init__
  - `test/srt/test_flashmla.py` modified +68/-0 (68 lines); hunk: import unittest; DEFAULT_MODEL_NAME_FOR_TEST_MLA,; 符号: test_latency, TestFlashMLAMTP, setUpClass, tearDownClass
  - `python/sglang/srt/speculative/eagle_worker.py` modified +13/-0 (13 lines); hunk: def init_attention_backend(self):; 符号: init_attention_backend
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +8/-4 (12 lines); hunk: def forward_extend(; def forward_extend(; 符号: forward_extend, forward_extend, forward_decode, __init__
  - `docs/backend/attention_backend.md` modified +7/-1 (8 lines); hunk: \| **FA3** \| ✅ \| ✅ \| ✅ \| ✅ \| ✅ \|; python3 -m sglang.launch_server --tp 8 --model deepseek-ai/D
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py`；patch 关键词为 attention, flash, mla, cache, cuda, eagle。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/flashmla_backend.py`, `test/srt/test_flashmla.py`, `python/sglang/srt/speculative/eagle_worker.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6151 - [Feat] optimize Qwen3 on H20 by hybrid Attention Backend

- 链接：https://github.com/sgl-project/sglang/pull/6151
- 状态/时间：`closed`，created 2025-05-09, closed 2025-05-18；作者 `TianQiLin666666`。
- 代码 diff 已读范围：`3` 个文件，`+39/-9`；代码面：kernel, scheduler/runtime；关键词：attention, flash, kv, spec, cache, config, cuda, lora, mla, processor。
- 代码 diff 细节：
  - `python/sglang/srt/model_executor/model_runner.py` modified +24/-4 (28 lines); hunk: def __init__(; def init_attention_backend(self):; 符号: __init__, init_attention_backend, init_double_sparsity_channel_config, apply_torch_tp
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +9/-5 (14 lines); hunk: def __init__(self, model_runner: ModelRunner):; def capture_one_batch_size(self, bs: int, forward: Callable):; 符号: __init__, capture_one_batch_size, capture_one_batch_size, replay_prepare
  - `python/sglang/srt/server_args.py` modified +6/-0 (6 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py`；patch 关键词为 attention, flash, kv, spec, cache, config。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6220 - Fuse routed scaling factor in topk_reduce kernel

- 链接：https://github.com/sgl-project/sglang/pull/6220
- 状态/时间：`merged`，created 2025-05-12, merged 2025-06-07；作者 `BBuf`。
- 代码 diff 已读范围：`10` 个文件，`+331/-9`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks；关键词：moe, quant, config, expert, router, triton, benchmark, cuda, topk, cache。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py` added +199/-0 (199 lines); hunk: +import torch; 符号: _moe_sum_reduce_kernel, moe_sum_reduce, compute_sum_scaled_baseline, compute_sum_scaled_compiled
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +124/-8 (132 lines); hunk: def inplace_fused_experts(; def inplace_fused_experts(; 符号: inplace_fused_experts, inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-1 (2 lines); hunk: def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:; 符号: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunk: def forward_cuda(; 符号: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 moe, quant, config, expert, router, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6890 - Use deepgemm instead of triton for fused_qkv_a_proj_with_mqa

- 链接：https://github.com/sgl-project/sglang/pull/6890
- 状态/时间：`merged`，created 2025-06-05, merged 2025-06-05；作者 `fzyzcjy`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：quantization；关键词：fp8, quant, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +2/-2 (4 lines); hunk: def deepgemm_w8a8_block_fp8_linear_with_fallback(; 符号: deepgemm_w8a8_block_fp8_linear_with_fallback
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 fp8, quant, triton。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #6970 - Fuse routed scaling factor in deepseek

- 链接：https://github.com/sgl-project/sglang/pull/6970
- 状态/时间：`merged`，created 2025-06-08, merged 2025-06-08；作者 `BBuf`。
- 代码 diff 已读范围：`10` 个文件，`+338/-15`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks；关键词：moe, quant, cuda, config, expert, router, triton, benchmark, topk, cache。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py` added +199/-0 (199 lines); hunk: +import torch; 符号: _moe_sum_reduce_kernel, moe_sum_reduce, compute_sum_scaled_baseline, compute_sum_scaled_compiled
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +130/-14 (144 lines); hunk: def inplace_fused_experts(; def inplace_fused_experts(; 符号: inplace_fused_experts, inplace_fused_experts, inplace_fused_experts_fake, outplace_fused_experts
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-1 (3 lines); hunk: def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:; 符号: forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +1/-0 (1 lines); hunk: def forward_cuda(; 符号: forward_cuda, forward_cpu
  - `python/sglang/srt/layers/quantization/blockwise_int8.py` modified +1/-0 (1 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 moe, quant, cuda, config, expert, router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/benchmark_sum_scale.py`, `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7146 - Support new DeepGEMM format in per token group quant

- 链接：https://github.com/sgl-project/sglang/pull/7146
- 状态/时间：`merged`，created 2025-06-13, merged 2025-06-13；作者 `fzyzcjy`。
- 代码 diff 已读范围：`5` 个文件，`+92/-44`；代码面：quantization, kernel, tests/benchmarks；关键词：fp8, quant, cuda, test。
- 代码 diff 细节：
  - `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu` modified +83/-40 (123 lines); hunk: __device__ __forceinline__ float GroupReduceMax(float val, const int tid) {; __global__ void per_token_group_quant_8bit_kernel(; 符号: void, void, void
  - `sgl-kernel/tests/test_per_token_group_quant_8bit.py` modified +4/-1 (5 lines); hunk: def sglang_per_token_group_quant_8bit(; 符号: sglang_per_token_group_quant_8bit
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +2/-1 (3 lines); hunk: void sgl_per_token_group_quant_fp8(
  - `sgl-kernel/python/sgl_kernel/gemm.py` modified +2/-1 (3 lines); hunk: def sgl_per_token_group_quant_fp8(; 符号: sgl_per_token_group_quant_fp8
  - `sgl-kernel/csrc/common_extension.cc` modified +1/-1 (2 lines); hunk: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h`；patch 关键词为 fp8, quant, cuda, test。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/gemm/per_token_group_quant_8bit.cu`, `sgl-kernel/tests/test_per_token_group_quant_8bit.py`, `sgl-kernel/include/sgl_kernel_ops.h` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7150 - Refactor DeepGEMM integration

- 链接：https://github.com/sgl-project/sglang/pull/7150
- 状态/时间：`merged`，created 2025-06-13, merged 2025-06-14；作者 `fzyzcjy`。
- 代码 diff 已读范围：`12` 个文件，`+207/-147`；代码面：model wrapper, MoE/router, quantization, kernel, scheduler/runtime, docs/config；关键词：quant, config, fp8, deepep, expert, moe, topk, cuda, triton, attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py` renamed +22/-76 (98 lines); hunk: from enum import IntEnum, auto; def get_enable_jit_deepgemm():; 符号: get_enable_jit_deepgemm, get_enable_jit_deepgemm, DeepGemmKernelHelper:, _compile_warning_1
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` added +95/-0 (95 lines); hunk: +import logging; 符号: grouped_gemm_nt_f8f8bf16_masked, grouped_gemm_nt_f8f8bf16_contig, gemm_nt_f8f8bf16, update_deep_gemm_config
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +35/-32 (67 lines); hunk: import logging; from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported; 符号: create_weights, __init__, forward, forward_deepgemm_contiguous
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` added +26/-0 (26 lines); hunk: +import logging; 符号: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +6/-10 (16 lines); hunk: import triton; sgl_per_token_quant_fp8,; 符号: is_fp8_fnuz, deep_gemm_fp8_fp8_bf16_nt, deep_gemm_fp8_fp8_bf16_nt, deep_gemm_fp8_fp8_bf16_nt_fake
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 quant, config, fp8, deepep, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7155 - Support new DeepGEMM format in per token group quant (part 2: srt)

- 链接：https://github.com/sgl-project/sglang/pull/7155
- 状态/时间：`merged`，created 2025-06-13, merged 2025-06-13；作者 `fzyzcjy`。
- 代码 diff 已读范围：`3` 个文件，`+19/-4`；代码面：quantization, kernel；关键词：config, cuda, flash, fp8, quant, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +17/-2 (19 lines); hunk: def sglang_per_token_group_quant_fp8(; def sglang_per_token_group_quant_fp8(; 符号: sglang_per_token_group_quant_fp8, sglang_per_token_group_quant_fp8
  - `python/pyproject.toml` modified +1/-1 (2 lines); hunk: runtime_common = [
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunk: def _set_envs_and_config(server_args: ServerArgs):; 符号: _set_envs_and_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`；patch 关键词为 config, cuda, flash, fp8, quant, test。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7156 - Re-quantize DeepSeek model weights to support DeepGEMM new input format

- 链接：https://github.com/sgl-project/sglang/pull/7156
- 状态/时间：`merged`，created 2025-06-13, merged 2025-06-13；作者 `fzyzcjy`。
- 代码 diff 已读范围：`3` 个文件，`+125/-0`；代码面：model wrapper, quantization；关键词：fp8, quant, config, deepep, expert, kv, moe。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +61/-0 (61 lines); hunk: import torch; def block_quant_dequant(; 符号: block_quant_dequant, requant_weight_ue8m0_inplace, _requant_weight_ue8m0, _transform_scale
  - `python/sglang/srt/models/deepseek_v2.py` modified +56/-0 (56 lines); hunk: block_quant_to_tensor_quant,; def post_load_weights(self, is_nextn=False, weight_names=None):; 符号: post_load_weights, _weight_requant_ue8m0, load_weights
  - `python/sglang/math_utils.py` added +8/-0 (8 lines); hunk: +# COPIED FROM DeepGEMM; 符号: align, ceil_div
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py`；patch 关键词为 fp8, quant, config, deepep, expert, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/math_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7172 - Support new DeepGEMM

- 链接：https://github.com/sgl-project/sglang/pull/7172
- 状态/时间：`merged`，created 2025-06-14, merged 2025-06-14；作者 `fzyzcjy`。
- 代码 diff 已读范围：`8` 个文件，`+59/-19`；代码面：model wrapper, MoE/router, quantization, kernel, docs/config；关键词：fp8, quant, moe, expert, topk, triton, config, deepep, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py` modified +18/-8 (26 lines); hunk: if ENABLE_JIT_DEEPGEMM:; 符号: grouped_gemm_nt_f8f8bf16_masked
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +20/-3 (23 lines); hunk: def prepare_block_fp8_matmul_inputs(; 符号: prepare_block_fp8_matmul_inputs
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +8/-1 (9 lines); hunk: def forward_deepgemm_masked(; 符号: forward_deepgemm_masked
  - `python/sglang/srt/layers/quantization/deep_gemm_wrapper/configurer.py` modified +7/-1 (8 lines); hunk: def _compute_enable_deep_gemm():; 符号: _compute_enable_deep_gemm
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +1/-4 (5 lines); hunk: import triton; def fused_moe_kernel(; 符号: fused_moe_kernel, ceil_div, moe_align_block_size_stage1
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 fp8, quant, moe, expert, topk, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/deep_gemm_wrapper/entrypoint.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7376 - Fix MTP with Deepseek R1 Fp4

- 链接：https://github.com/sgl-project/sglang/pull/7376
- 状态/时间：`merged`，created 2025-06-20, merged 2025-06-24；作者 `pyc96`。
- 代码 diff 已读范围：`3` 个文件，`+20/-1`；代码面：model wrapper, MoE/router, kernel；关键词：config, quant, kv, moe, awq, cache, expert, flash, fp4, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; 符号: load_weights, load_weights
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +6/-0 (6 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/deepseek_nextn.py` modified +6/-0 (6 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py`；patch 关键词为 config, quant, kv, moe, awq, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/models/deepseek_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #7762 - feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode

- 链接：https://github.com/sgl-project/sglang/pull/7762
- 状态/时间：`merged`，created 2025-07-04, merged 2025-07-07；作者 `yangsijia-celina`。
- 代码 diff 已读范围：`10` 个文件，`+1006/-9`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, quant, expert, fp8, config, topk, triton, router, cuda, flash。
- 代码 diff 细节：
  - `python/sglang/test/test_cutlass_w4a8_moe.py` added +281/-0 (281 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: pack_int4_values_to_int8, pack_interleave, test_cutlass_w4a8_moe, cutlass_moe
  - `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunk: +import logging; 符号: W4AFp8Config, for, __init__, get_name
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunk: +# SPDX-License-Identifier: Apache-2.0; 符号: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunk: ); moe_ep_deepgemm_preprocess,; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunk: def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):; def run_moe_ep_preproess(topk_ids: torch.Tensor, num_experts: int):; 符号: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_moe_ep_preproess, run_cutlass_moe_ep_preproess
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`；patch 关键词为 moe, quant, expert, fp8, config, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8118 - [feat] Support tp mode for DeepSeek-R1-W4AFP8

- 链接：https://github.com/sgl-project/sglang/pull/8118
- 状态/时间：`merged`，created 2025-07-17, merged 2025-09-02；作者 `chenxijun1029`。
- 代码 diff 已读范围：`11` 个文件，`+291/-120`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：expert, moe, fp8, config, quant, spec, topk, cuda, test, triton。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +206/-60 (266 lines); hunk: #include <cudaTypedefs.h>; void dispatch_w4a8_moe_mm_sm90(; 符号: Sched, SM90W4A8Config, JOIN_STRUCT_NAME, JOIN_STRUCT_NAME_CO
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunk: from __future__ import annotations; def get_quant_method(; 符号: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales
  - `python/sglang/test/test_cutlass_w4a8_moe.py` modified +24/-9 (33 lines); hunk: # SPDX-License-Identifier: Apache-2.0; def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Ten; 符号: pack_int4_values_to_int8, pack_interleave, pack_interleave, pack_interleave
  - `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py` modified +10/-4 (14 lines); hunk: def pack_interleave(num_experts, ref_weight, ref_scale):; def test_int4_fp8_grouped_gemm_single_expert(batch_size):; 符号: pack_interleave, test_int4_fp8_grouped_gemm_single_expert, test_int4_fp8_grouped_gemm_multi_experts
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cuh` modified +7/-6 (13 lines); hunk: using MmaType = cutlass::float_e4m3_t; // FP8 e4m3 type; static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;; 符号: int, cutlass_3x_w4a8_group_gemm, int, int
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`；patch 关键词为 expert, moe, fp8, config, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/test/test_cutlass_w4a8_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8247 - [1/N]Support DeepSeek-R1 w4a8 normal deepep

- 链接：https://github.com/sgl-project/sglang/pull/8247
- 状态/时间：`merged`，created 2025-07-22, merged 2025-10-15；作者 `ayrnb`。
- 代码 diff 已读范围：`7` 个文件，`+334/-7`；代码面：MoE/router, quantization, tests/benchmarks；关键词：deepep, moe, fp8, quant, config, expert, flash, fp4, topk, attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +196/-0 (196 lines); hunk: # SPDX-License-Identifier: Apache-2.0; ); 符号: cutlass_w4a8_moe, cutlass_w4a8_moe_deepep_normal
  - `test/srt/quant/test_w4a8_deepseek_v3.py` modified +55/-0 (55 lines); hunk: def test_gsm8k(; 符号: test_gsm8k, TestDeepseekV3W4Afp8DeepepNormal, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +47/-1 (48 lines); hunk: from __future__ import annotations; if TYPE_CHECKING:; 符号: apply, apply_deepep_normal
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +21/-2 (23 lines); hunk: CUTEDSL_MOE_NVFP4_DISPATCH,; def __init__(; 符号: __init__, __init__, moe_impl, forward_flashinfer_cutedsl
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +10/-4 (14 lines); hunk: DispatchOutput,; def dispatch_a(; 符号: dispatch_a, _dispatch_core, _dispatch_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py`；patch 关键词为 deepep, moe, fp8, quant, config, expert。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `test/srt/quant/test_w4a8_deepseek_v3.py`, `python/sglang/srt/layers/quantization/w4afp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #8464 - [2/N]Support DeepSeek-R1 w4a8 low latency deepep

- 链接：https://github.com/sgl-project/sglang/pull/8464
- 状态/时间：`merged`，created 2025-07-28, merged 2025-10-25；作者 `ayrnb`。
- 代码 diff 已读范围：`8` 个文件，`+531/-9`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：fp8, moe, quant, deepep, expert, topk, attention, config, cuda, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +194/-0 (194 lines); hunk: def zero_experts_compute_triton(; 符号: zero_experts_compute_triton, compute_problem_sizes_w4a8_kernel, compute_problem_sizes_w4a8, deepep_ll_get_cutlass_w4a8_moe_mm_data
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +138/-0 (138 lines); hunk: ); def cutlass_w4a8_moe_deepep_normal(; 符号: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` modified +72/-6 (78 lines); hunk: __global__ void int4_fp8_get_group_gemm_starts(; __global__ void int4_fp8_get_group_gemm_starts(; 符号: void, void, void
  - `test/srt/quant/test_w4a8_deepseek_v3.py` modified +69/-0 (69 lines); hunk: +import os; def test_gsm8k(; 符号: test_gsm8k, TestDeepseekV3W4Afp8DeepepAutoMtp, setUpClass, tearDownClass
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +36/-0 (36 lines); hunk: from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE; def apply(; 符号: apply, apply_deepep_ll, apply_deepep_normal
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh`；patch 关键词为 fp8, moe, quant, deepep, expert, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_get_group_starts.cuh` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10027 - [Perf] Optimize DeepSeek-R1 w4afp8 glue kernels

- 链接：https://github.com/sgl-project/sglang/pull/10027
- 状态/时间：`merged`，created 2025-09-04, merged 2025-11-24；作者 `yuhyao`。
- 代码 diff 已读范围：`3` 个文件，`+253/-77`；代码面：MoE/router, quantization, kernel；关键词：deepep, moe, quant, config, expert, fp8, topk, triton, cuda, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +227/-54 (281 lines); hunk: import triton.language as tl; def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):; 符号: _get_launch_config_1d, get_num_blocks, _get_launch_config_2d, get_num_blocks
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +21/-17 (38 lines); hunk: silu_and_mul,; def cutlass_w4a8_moe(; 符号: cutlass_w4a8_moe, cutlass_w4a8_moe, cutlass_w4a8_moe, cutlass_w4a8_moe
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +5/-6 (11 lines); hunk: def process_weights_after_loading(self, layer: Module) -> None:; def apply(; 符号: process_weights_after_loading, apply, apply_deepep_ll
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py`；patch 关键词为 deepep, moe, quant, config, expert, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/quantization/w4afp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10361 - Fix GPU fault issue when run dsv3 with dp mode and enable torch-compile

- 链接：https://github.com/sgl-project/sglang/pull/10361
- 状态/时间：`merged`，created 2025-09-12, merged 2025-09-12；作者 `kkHuang-amd`。
- 代码 diff 已读范围：`2` 个文件，`+39/-5`；代码面：attention/backend, multimodal/processor；关键词：attention, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/dp_attention.py` modified +24/-0 (24 lines); hunk: def get_local_dp_buffer_len(cls) -> int:; def get_dp_global_num_tokens() -> List[int]:; 符号: get_local_dp_buffer_len, get_dp_global_num_tokens, get_dp_hidden_size, get_dp_dtype
  - `python/sglang/srt/layers/logits_processor.py` modified +15/-5 (20 lines); hunk: get_attention_dp_rank,; def compute_dp_attention_metadata(self):; 符号: compute_dp_attention_metadata, _get_logits
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py`；patch 关键词为 attention, processor。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/layers/logits_processor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11512 - Update DeepSeek-R1-FP4 default config on blackwell

- 链接：https://github.com/sgl-project/sglang/pull/11512
- 状态/时间：`merged`，created 2025-10-12, merged 2025-10-13；作者 `Qiaolin-Yu`。
- 代码 diff 已读范围：`1` 个文件，`+26/-1`；代码面：misc；关键词：attention, config, cuda, flash, fp4, kv, mla, moe, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +26/-1 (27 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 attention, config, cuda, flash, fp4, kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11708 - Support running FP4 Deepseek on SM120.

- 链接：https://github.com/sgl-project/sglang/pull/11708
- 状态/时间：`merged`，created 2025-10-16, merged 2025-10-28；作者 `weireweire`。
- 代码 diff 已读范围：`9` 个文件，`+33/-35`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks；关键词：cuda, flash, kv, attention, cache, moe, spec, config, fp4, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +8/-7 (15 lines); hunk: from sglang.srt.layers.quantization.fp8_utils import (; ); 符号: apply, ModelOptNvFp4FusedMoEMethod, __init__, apply
  - `python/sglang/srt/models/gpt_oss.py` modified +1/-10 (11 lines); hunk: enable_fused_set_kv_buffer,
  - `python/sglang/srt/utils/common.py` modified +10/-1 (11 lines); hunk: def device_context(device: torch.device):; 符号: device_context, is_blackwell, is_blackwell_supported
  - `python/sglang/srt/server_args.py` modified +4/-3 (7 lines); hunk: get_device,; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_attention_backend_compatibility
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-5 (6 lines); hunk: get_int_env_var,; else:; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py`；patch 关键词为 cuda, flash, kv, attention, cache, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/utils/common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12000 - [1/2] deepseek deterministic: support deterministic inference for deepseek arch models on a single GPU

- 链接：https://github.com/sgl-project/sglang/pull/12000
- 状态/时间：`merged`，created 2025-10-23, merged 2025-10-24；作者 `zminglei`。
- 代码 diff 已读范围：`3` 个文件，`+64/-5`；代码面：model wrapper；关键词：attention, flash, spec, triton, cache, config, cuda, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py` modified +31/-2 (33 lines); hunk: def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype \| None =; def enable_batch_invariant_mode():; 符号: mean_batch_invariant, bmm_batch_invariant, is_batch_invariant_mode_enabled, enable_batch_invariant_mode
  - `python/sglang/srt/server_args.py` modified +24/-2 (26 lines); hunk: def _handle_deterministic_inference(self):; def _handle_deterministic_inference(self):; 符号: _handle_deterministic_inference, _handle_deterministic_inference
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunk: def handle_attention_flashinfer(attn, forward_batch):; def handle_attention_nsa(attn, forward_batch):; 符号: handle_attention_flashinfer, handle_attention_fa3, handle_attention_flashmla, handle_attention_nsa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, flash, spec, triton, cache, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/batch_invariant_ops/batch_invariant_ops.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12057 - [doc] add example of using w4fp8 for Deepseek

- 链接：https://github.com/sgl-project/sglang/pull/12057
- 状态/时间：`merged`，created 2025-10-24, merged 2025-10-27；作者 `Kevin-XiongC`。
- 代码 diff 已读范围：`1` 个文件，`+15/-0`；代码面：tests/benchmarks；关键词：benchmark, config, expert, fp8, moe, quant。
- 代码 diff 细节：
  - `benchmark/deepseek_v3/README.md` modified +15/-0 (15 lines); hunk: edit your `config.json` and remove the `quantization_config` block. For example:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/deepseek_v3/README.md`；patch 关键词为 benchmark, config, expert, fp8, moe, quant。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/deepseek_v3/README.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12778 - Update dsv3 quantization auto setting for sm100

- 链接：https://github.com/sgl-project/sglang/pull/12778
- 状态/时间：`merged`，created 2025-11-06, merged 2025-11-06；作者 `ispobock`。
- 代码 diff 已读范围：`1` 个文件，`+22/-9`；代码面：misc；关键词：config, flash, fp4, fp8, kv, moe, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +22/-9 (31 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 config, flash, fp4, fp8, kv, moe。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12921 - [perf]optimize w4afp8 kernel on deepseek-v3-0324

- 链接：https://github.com/sgl-project/sglang/pull/12921
- 状态/时间：`merged`，created 2025-11-09, merged 2025-12-18；作者 `Bruce-x-1997`。
- 代码 diff 已读范围：`3` 个文件，`+160/-264`；代码面：MoE/router, quantization, kernel；关键词：moe, topk, expert, config, cuda, fp8, quant。
- 代码 diff 细节：
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +64/-236 (300 lines); hunk: inline void invoke_gemm(; void dispatch_w4a8_moe_mm_sm90(; 符号: parameters, parameter
  - `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu` modified +96/-27 (123 lines); hunk: #include <cudaTypedefs.h>; __global__ void compute_problem_sizes_w4a8(; 符号: uint64_t, void, void, void
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +0/-1 (1 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py`；patch 关键词为 moe, topk, expert, config, cuda, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu`, `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_moe_data.cu`, `python/sglang/srt/layers/quantization/w4afp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13548 - [Fix] Fix DeepSeek V3 MTP on B200

- 链接：https://github.com/sgl-project/sglang/pull/13548
- 状态/时间：`merged`，created 2025-11-19, merged 2025-11-19；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：config, processor, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`；patch 关键词为 config, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14162 - DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- 链接：https://github.com/sgl-project/sglang/pull/14162
- 状态/时间：`merged`，created 2025-11-30, merged 2026-03-30；作者 `xieminghe1`。
- 代码 diff 已读范围：`5` 个文件，`+94/-12`；代码面：MoE/router, quantization, kernel；关键词：fp8, moe, quant, deepep, config, topk, triton, expert, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunk: def silu_and_mul_masked_post_per_tensor_quant_fwd(; 符号: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunk: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; 符号: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunk: def forward_cutlass_w4afp8(; def forward_cutlass_w4afp8_masked(; 符号: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunk: def apply_deepep_ll(; 符号: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunk: def _dispatch_core(; 符号: _dispatch_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 fp8, moe, quant, deepep, config, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14897 - Fix dsv3 dp accuracy issue when using bf16-kv

- 链接：https://github.com/sgl-project/sglang/pull/14897
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-11；作者 `Duyi-Wang`。
- 代码 diff 已读范围：`1` 个文件，`+8/-2`；代码面：attention/backend；关键词：attention, cache, fp8, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +8/-2 (10 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`；patch 关键词为 attention, cache, fp8, kv, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15304 - Fix the accuracy issue when running mxfp4 dsv3 model and enable ep

- 链接：https://github.com/sgl-project/sglang/pull/15304
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-17；作者 `kkHuang-amd`。
- 代码 diff 已读范围：`2` 个文件，`+2/-0`；代码面：MoE/router, quantization；关键词：expert, quant, fp4, moe。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/mxfp4.py` modified +1/-0 (1 lines); hunk: def apply(; 符号: apply
  - `python/sglang/srt/layers/quantization/quark/quark_moe.py` modified +1/-0 (1 lines); hunk: def apply(; 符号: apply
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py`；patch 关键词为 expert, quant, fp4, moe。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/mxfp4.py`, `python/sglang/srt/layers/quantization/quark/quark_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #15531 - Support piecewise cuda graph for dsv3 fp4

- 链接：https://github.com/sgl-project/sglang/pull/15531
- 状态/时间：`merged`，created 2025-12-20, merged 2025-12-21；作者 `ispobock`。
- 代码 diff 已读范围：`7` 个文件，`+148/-16`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, tests/benchmarks；关键词：flash, fp4, attention, cuda, quant, cache, mla, config, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +64/-10 (74 lines); hunk: StandardDispatcher,; def _quantize_hidden_states_fp4(self, hidden_states: torch.Tensor):; 符号: _quantize_hidden_states_fp4, forward, moe_forward_piecewise_cuda_graph_impl_fake, flashinfer_fp4_moe_forward_piecewise_cuda_graph_impl
  - `test/srt/test_deepseek_v3_fp4_4gpu.py` modified +67/-0 (67 lines); hunk: def test_bs_1_speed(self):; 符号: test_bs_1_speed, TestDeepseekV3FP4PiecewiseCudaGraph, setUpClass, tearDownClass
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-1 (12 lines); hunk: import concurrent.futures; def handle_attention_fa4(attn, forward_batch):; 符号: handle_attention_fa4, handle_attention_trtllm_mla, forward
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +3/-1 (4 lines); hunk: import triton; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` modified +1/-2 (3 lines); hunk: ); def apply_weights(; 符号: apply_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 flash, fp4, attention, cuda, quant, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `test/srt/test_deepseek_v3_fp4_4gpu.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16649 - [Refactor] Split out deepseek v2 weight loader function into mixin

- 链接：https://github.com/sgl-project/sglang/pull/16649
- 状态/时间：`merged`，created 2026-01-07, merged 2026-01-18；作者 `xyjixyjixyji`。
- 代码 diff 已读范围：`4` 个文件，`+721/-600`；代码面：model wrapper；关键词：cuda, fp8, moe, awq, config, fp4, kv, quant, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` added +657/-0 (657 lines); hunk: +# Copyright 2026 SGLang Team; 符号: DeepseekV2WeightLoaderMixin:, do_load_weights, post_load_weights, _maybe_quant_weights_to_fp8_ue8m0
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-594 (603 lines); hunk: """Inference-only DeepseekV2 model."""; per_tensor_quant_mla_fp8,; 符号: enable_nextn_moe_bf16_cast_to_fp8, forward, DeepseekV2ForCausalLM, DeepseekV2ForCausalLM
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +53/-1 (54 lines); hunk: +# Copyright 2026 SGLang Team; _is_cpu = is_cpu(); 符号: awq_dequantize_func, enable_nextn_moe_bf16_cast_to_fp8
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-5 (7 lines); hunk: VocabParallelEmbedding,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py`；patch 关键词为 cuda, fp8, moe, awq, config, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17133 - [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab

- 链接：https://github.com/sgl-project/sglang/pull/17133
- 状态/时间：`merged`，created 2026-01-15, merged 2026-01-16；作者 `xu-yfei`。
- 代码 diff 已读范围：`6` 个文件，`+959/-217`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, triton, config, fp8, benchmark, cache, cuda, expert, quant, router。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py` modified +337/-215 (552 lines); hunk: # Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py; sort_config,; 符号: MoeInputs:, KernelWrapper:, __init__, cuda_graph_wrapper
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` added +164/-0 (164 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128].json` added +146/-0 (146 lines); hunk: +{
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`；patch 关键词为 moe, triton, config, fp8, benchmark, cache。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17178 - Remove deepseek-r1 from THINKING_MODE_CHOICES in run_eval.py

- 链接：https://github.com/sgl-project/sglang/pull/17178
- 状态/时间：`merged`，created 2026-01-16, merged 2026-01-16；作者 `hlu1`。
- 代码 diff 已读范围：`1` 个文件，`+3/-2`；代码面：tests/benchmarks；关键词：spec, test。
- 代码 diff 细节：
  - `python/sglang/test/run_eval.py` modified +3/-2 (5 lines); hunk: def get_thinking_kwargs(args):; def run_eval(args):; 符号: get_thinking_kwargs, run_eval, run_eval
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/run_eval.py`；patch 关键词为 spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/run_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17707 - Add dsv3 router gemm benchmark on blackwell

- 链接：https://github.com/sgl-project/sglang/pull/17707
- 状态/时间：`merged`，created 2026-01-25, merged 2026-04-04；作者 `harrisonlimh`。
- 代码 diff 已读范围：`2` 个文件，`+284/-4`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks；关键词：cuda, flash, router, attention, benchmark, config, fp8, kv, mla, quant。
- 代码 diff 细节：
  - `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py` added +250/-0 (250 lines); hunk: +import argparse; 符号: create_benchmark_configs, dsv3_router_gemm_flashinfer, dsv3_router_gemm_sgl, check_accuracy
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-4 (38 lines); hunk: pass; def forward(; 符号: forward, DeepseekV32ForCausalLM, flashinfer_dsv3_router_gemm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 cuda, flash, router, attention, benchmark, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/deepseek/benchmark_deepgemm_dsv3_router_gemm_blackwell.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17744 - Fix OOM in DeepSeek weight loading by deferring dict(weights) materialization

- 链接：https://github.com/sgl-project/sglang/pull/17744
- 状态/时间：`merged`，created 2026-01-26, merged 2026-01-31；作者 `hsuchifeng`。
- 代码 diff 已读范围：`1` 个文件，`+16/-12`；代码面：model wrapper；关键词：config, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +16/-12 (28 lines); hunk: def _maybe_quant_weights_to_fp8_ue8m0(; def _maybe_quant_weights_to_fp8_ue8m0(; 符号: _maybe_quant_weights_to_fp8_ue8m0, _maybe_quant_weights_to_fp8_ue8m0
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py`；patch 关键词为 config, fp8, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18242 - [ROCm] Optimize Deepseek R1 on MI300X

- 链接：https://github.com/sgl-project/sglang/pull/18242
- 状态/时间：`merged`，created 2026-02-04, merged 2026-02-25；作者 `zhentaocc`。
- 代码 diff 已读范围：`3` 个文件，`+7/-2`；代码面：model wrapper, attention/backend, quantization；关键词：cache, cuda, fp8, attention, kv, mla, quant, router, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-2 (7 lines); hunk: get_dsv3_gemm_output_zero_allocator_size,; def forward_absorb_prepare(; 符号: forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-0 (1 lines); hunk: def forward_decode(; 符号: forward_decode
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunk: aiter_per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 cache, cuda, fp8, attention, kv, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/aiter_backend.py`, `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18451 - [AMD] Use aiter_dsv3_router_gemm kernel if number of experts <= 256.

- 链接：https://github.com/sgl-project/sglang/pull/18451
- 状态/时间：`merged`，created 2026-02-08, merged 2026-03-19；作者 `amd-mvarjoka`。
- 代码 diff 已读范围：`1` 个文件，`+5/-1`；代码面：model wrapper；关键词：router。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-1 (6 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 router。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18461 - [Intel GPU] Enable DeepSeek R1 inference on XPU

- 链接：https://github.com/sgl-project/sglang/pull/18461
- 状态/时间：`merged`，created 2026-02-09, merged 2026-03-30；作者 `polisettyvarma`。
- 代码 diff 已读范围：`6` 个文件，`+46/-28`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks；关键词：cuda, moe, quant, benchmark, config, expert, fp8, spec, awq, flash。
- 代码 diff 细节：
  - `benchmark/kernels/quantization/tuning_block_wise_kernel.py` modified +22/-20 (42 lines); hunk: _w8a8_block_fp8_matmul_unrolledx4,; def benchmark_config(; 符号: benchmark_config, run, run, tune
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +9/-4 (13 lines); hunk: ServerArgs,; def run():; 符号: benchmark_config, run, BenchmarkWorker:, __init__
  - `python/sglang/srt/layers/moe/token_dispatcher/standard.py` modified +9/-3 (12 lines); hunk: get_moe_runner_backend,; def dispatch(; 符号: dispatch
  - `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py` modified +2/-1 (3 lines); hunk: _is_fp8_fnuz,; def post_load_weights(; 符号: post_load_weights
  - `python/sglang/srt/models/deepseek_common/utils.py` modified +2/-0 (2 lines); hunk: is_hip,; _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py`；patch 关键词为 cuda, moe, quant, benchmark, config, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/quantization/tuning_block_wise_kernel.py`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`, `python/sglang/srt/layers/moe/token_dispatcher/standard.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18607 - [AMD] Fix accuracy issue when running TP4 dsv3 model with mtp

- 链接：https://github.com/sgl-project/sglang/pull/18607
- 状态/时间：`merged`，created 2026-02-11, merged 2026-02-12；作者 `1am9trash`。
- 代码 diff 已读范围：`2` 个文件，`+9/-5`；代码面：attention/backend；关键词：attention, cache, doc, fp8, kv, mla, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +7/-3 (10 lines); hunk: def __init__(; def make_mla_meta_data(; 符号: __init__, make_mla_meta_data
  - `docker/rocm.Dockerfile` modified +2/-2 (4 lines); hunk: ENV BUILD_TRITON="0"; ENV BUILD_TRITON="0"
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile`；patch 关键词为 attention, cache, doc, fp8, kv, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py`, `docker/rocm.Dockerfile` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19122 - [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py

- 链接：https://github.com/sgl-project/sglang/pull/19122
- 状态/时间：`merged`，created 2026-02-21, merged 2026-02-27；作者 `Fridge003`。
- 代码 diff 已读范围：`9` 个文件，`+906/-818`；代码面：model wrapper, attention/backend, tests/benchmarks；关键词：attention, mla, cache, kv, fp8, lora, quant, cuda, triton, config。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +22/-811 (833 lines); hunk: from __future__ import annotations; MaybeTboDeepEPDispatcher,; 符号: DeepseekV2MLP, __init__, op_output, DeepseekV2AttentionMLA
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` added +492/-0 (492 lines); hunk: +from __future__ import annotations; 符号: DeepseekMLAForwardMixin:, init_mla_forward, forward_absorb_prepare, forward_absorb_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` added +227/-0 (227 lines); hunk: +from __future__ import annotations; 符号: DeepseekMLARocmForwardMixin:, init_mla_fused_rope_rocm_forward, forward_absorb_fused_mla_rope_prepare, forward_absorb_fused_mla_rope_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` added +152/-0 (152 lines); hunk: +from __future__ import annotations; 符号: DeepseekMLACpuForwardMixin:, init_mla_fused_rope_cpu_forward, forward_absorb_fused_mla_rope_cpu_prepare, forward_absorb_fused_mla_rope_cpu_core
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/__init__.py` modified +6/-0 (6 lines); hunk: from .forward_methods import AttnForwardMethod
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py`；patch 关键词为 attention, mla, cache, kv, fp8, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19425 - [AMD] Fix weight load shape mismatch for amd dsr1 0528 mxfp4

- 链接：https://github.com/sgl-project/sglang/pull/19425
- 状态/时间：`merged`，created 2026-02-26, merged 2026-02-27；作者 `billishyahao`。
- 代码 diff 已读范围：`2` 个文件，`+18/-2`；代码面：model wrapper, quantization；关键词：config, kv, cache, cuda, fp4, fp8, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +11/-0 (11 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; class DeepseekModelNextN(nn.Module):; 符号: DeepseekModelNextN, __init__, forward, DeepseekV3ForCausalLMNextN
  - `python/sglang/srt/layers/quantization/quark/quark.py` modified +7/-2 (9 lines); hunk: def __init__(; def get_min_capability(cls) -> int:; 符号: __init__, get_min_capability, get_name, apply_weight_name_mapper
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py`；patch 关键词为 config, kv, cache, cuda, fp4, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py`, `python/sglang/srt/layers/quantization/quark/quark.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19834 - [AMD] CI - Add MI35x nightly/PR tests for kv-cache-fp8 and allreduce-fusion (DeepSeek)

- 链接：https://github.com/sgl-project/sglang/pull/19834
- 状态/时间：`merged`，created 2026-03-04, merged 2026-03-05；作者 `yctseng0211`。
- 代码 diff 已读范围：`13` 个文件，`+1614/-177`；代码面：quantization, tests/benchmarks；关键词：test, cache, fp4, config, fp8, kv, benchmark, quant, doc, attention。
- 代码 diff 细节：
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py` added +281/-0 (281 lines); hunk: +"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with KV Cache FP8 (8-GPU); 符号: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py` added +280/-0 (280 lines); hunk: +"""MI35x DeepSeek-R1-MXFP4 GSM8K Completion Evaluation Test with AIter AllReduce Fusion (8-GPU); 符号: get_model_path, ModelConfig:, __post_init__, get_display_name
  - `.github/workflows/nightly-test-amd-rocm720.yml` modified +153/-68 (221 lines); hunk: on:; jobs:
  - `.github/workflows/nightly-test-amd.yml` modified +153/-68 (221 lines); hunk: on:; jobs:
  - `test/registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_kv_fp8_perf_mi35x.py` added +178/-0 (178 lines); hunk: +"""MI35x Nightly performance benchmark for DeepSeek-R1-MXFP4 model with KV Cache FP8.; 符号: generate_simple_markdown_report, get_model_path, TestDeepseekR1MXFP4KvFp8PerfMI35x, setUpClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml`；patch 关键词为 test, cache, fp4, config, fp8, kv。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_kv_fp8_eval_mi35x.py`, `test/registered/amd/accuracy/mi35x/test_deepseek_r1_mxfp4_ar_fusion_eval_mi35x.py`, `.github/workflows/nightly-test-amd-rocm720.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19843 - [AMD] Use bfloat16 for correction_bias in AITER FP8 path to avoid runtime dtype conversion for dsv3

- 链接：https://github.com/sgl-project/sglang/pull/19843
- 状态/时间：`merged`，created 2026-03-04, merged 2026-03-06；作者 `inkcherry`。
- 代码 diff 已读范围：`1` 个文件，`+12/-7`；代码面：model wrapper；关键词：config, expert, flash, fp4, fp8, moe, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +12/-7 (19 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, expert, flash, fp4, fp8, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20841 - Fix gpu-fault issue when run deepseek-r1 and enable dp

- 链接：https://github.com/sgl-project/sglang/pull/20841
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-19；作者 `kkHuang-amd`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention, cuda, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/aiter_backend.py` modified +1/-1 (2 lines); hunk: def init_forward_metadata_replay_cuda_graph(; 符号: init_forward_metadata_replay_cuda_graph, get_cuda_graph_seq_len_fill_value, update_verify_buffers_to_fill_after_draft
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/aiter_backend.py`；patch 关键词为 attention, cuda, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/aiter_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21280 - [RL] Support mxfp8 DeepSeek V3

- 链接：https://github.com/sgl-project/sglang/pull/21280
- 状态/时间：`merged`，created 2026-03-24, merged 2026-04-04；作者 `zianglih`。
- 代码 diff 已读范围：`3` 个文件，`+105/-45`；代码面：attention/backend, MoE/router, quantization, kernel, scheduler/runtime；关键词：fp8, moe, quant, config, cuda, flash, fp4, topk, cache, expert。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py` modified +86/-38 (124 lines); hunk: else:; def align_fp8_moe_weights_for_flashinfer_trtllm(; 符号: align_fp8_moe_weights_for_flashinfer_trtllm, align_fp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm, align_mxfp8_moe_weights_for_flashinfer_trtllm
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-7 (19 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/layers/quantization/fp8.py` modified +7/-0 (7 lines); hunk: from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput; def get_quant_method(; 符号: get_quant_method, get_scaled_act_names, apply_weight_name_mapper, Fp8LinearMethod
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`；patch 关键词为 fp8, moe, quant, config, cuda, flash。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21529 - Add MXFP4 (including Quark W4A4) quantization support for DeepSeek-architecture on ROCm

- 链接：https://github.com/sgl-project/sglang/pull/21529
- 状态/时间：`open`，created 2026-03-27；作者 `JohnQinAMD`。
- 代码 diff 已读范围：`10` 个文件，`+308/-126`；代码面：model wrapper, MoE/router, quantization, kernel, tests/benchmarks；关键词：quant, fp4, config, expert, moe, attention, cuda, triton, cache, kv。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py` modified +85/-44 (129 lines); hunk: if _use_aiter:; def __init__(self, weight_config: dict[str, Any], input_config: dict[str, Any]):; 符号: __init__, create_weights, create_weights, process_weights_after_loading
  - `test/registered/amd/test_glm5_mxfp4.py` added +114/-0 (114 lines); hunk: +"""GLM-5 MXFP4 tests (4-GPU, MI35x); 符号: _GLM5MXFP4Base, for, setUpClass, tearDownClass
  - `test/registered/amd/test_kimi_k25_mxfp4.py` modified +40/-62 (102 lines); hunk: -"""Kimi-K2.5-MXFP4 aiter MLA backend test (4-GPU, FP8 KV cache); register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x"); 符号: TestKimiK25QuarkMXFP4, TestKimiK25MXFP4, setUpClass, tearDownClass
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +28/-14 (42 lines); hunk: def __init__(; def _load_w13(; 符号: __init__, _load_w13, _load_w2, weight_loader
  - `python/sglang/srt/models/deepseek_v2.py` modified +20/-2 (22 lines); hunk: def forward(; def determine_num_fused_shared_experts(; 符号: forward, DeepseekV2ForCausalLM, __init__, determine_num_fused_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py`；patch 关键词为 quant, fp4, config, expert, moe, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4_moe.py`, `test/registered/amd/test_glm5_mxfp4.py`, `test/registered/amd/test_kimi_k25_mxfp4.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21531 - [JIT Kernel] Migrate dsv3_router_gemm from AOT sgl-kernel to JIT kernel

- 链接：https://github.com/sgl-project/sglang/pull/21531
- 状态/时间：`open`，created 2026-03-27；作者 `meinie0826`。
- 代码 diff 已读范围：`11` 个文件，`+450/-39`；代码面：model wrapper, MoE/router, kernel, tests/benchmarks；关键词：router, cuda, expert, test, attention, awq, benchmark, cache, config, fp8。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh` added +184/-0 (184 lines); hunk: +/*; 符号: int, int, int, __launch_bounds__
  - `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py` added +104/-0 (104 lines); hunk: +"""Benchmark for DeepSeek V3 router GEMM (JIT kernel vs torch.nn.functional.linear).; 符号: benchmark_bf16_output, benchmark_float32_output
  - `python/sglang/jit_kernel/dsv3_router_gemm.py` added +103/-0 (103 lines); hunk: +"""; 符号: _jit_dsv3_router_gemm_module, can_use_dsv3_router_gemm, dsv3_router_gemm
  - `python/sglang/jit_kernel/tests/test_dsv3_router_gemm.py` added +36/-0 (36 lines); hunk: +"""Tests for JIT dsv3_router_gemm kernel."""; 符号: _ref, test_dsv3_router_gemm
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-6 (19 lines); hunk: pass; def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py`；patch 关键词为 router, cuda, expert, test, attention, awq。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/gemm/dsv3_router_gemm.cuh`, `python/sglang/jit_kernel/benchmark/bench_dsv3_router_gemm.py`, `python/sglang/jit_kernel/dsv3_router_gemm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21599 - [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1

- 链接：https://github.com/sgl-project/sglang/pull/21599
- 状态/时间：`merged`，created 2026-03-28, merged 2026-04-20；作者 `alphabetc1`。
- 代码 diff 已读范围：`13` 个文件，`+1296/-33`；代码面：kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：spec, config, cuda, eagle, attention, topk, cache, kv, quant, test。
- 代码 diff 细节：
  - `benchmark/bench_adaptive_speculative.py` added +263/-0 (263 lines); hunk: +"""Benchmark adaptive speculative decoding against static baselines.; 符号: build_phase_plan, send_request, run_phase, summarize_phases
  - `test/registered/unit/spec/test_adaptive_spec_params.py` added +195/-0 (195 lines); hunk: +import unittest; 符号: TestAdaptiveSpeculativeParams, test_initial_steps_snap_to_nearest_candidate_preferring_larger_step, test_update_respects_warmup_and_interval, test_empty_batches_do_not_consume_warmup_or_shift_steps
  - `test/registered/spec/eagle/test_adaptive_speculative.py` added +170/-0 (170 lines); hunk: +import json; 符号: TestAdaptiveSpeculativeServer, setUpClass, tearDownClass, _get_internal_state
  - `python/sglang/srt/speculative/eagle_worker.py` modified +162/-4 (166 lines); hunk: import logging; alloc_token_slots,; 符号: __init__, __init__, init_cuda_graphs, apply_runtime_state
  - `docs/advanced_features/adaptive_speculative_decoding.md` added +156/-0 (156 lines); hunk: +# Adaptive Speculative Decoding
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py`；patch 关键词为 spec, config, cuda, eagle, attention, topk。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21719 - Revert "DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication"

- 链接：https://github.com/sgl-project/sglang/pull/21719
- 状态/时间：`merged`，created 2026-03-31, merged 2026-03-31；作者 `BBuf`。
- 代码 diff 已读范围：`5` 个文件，`+12/-94`；代码面：MoE/router, quantization, kernel；关键词：fp8, moe, quant, deepep, config, topk, triton, expert, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +0/-73 (73 lines); hunk: def silu_and_mul_masked_post_per_tensor_quant_fwd(; 符号: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +7/-15 (22 lines); hunk: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; 符号: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +3/-3 (6 lines); hunk: def forward_cutlass_w4afp8(; def forward_cutlass_w4afp8_masked(; 符号: forward_cutlass_w4afp8, forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +1/-2 (3 lines); hunk: def apply_deepep_ll(; 符号: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunk: def _dispatch_core(; 符号: _dispatch_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 fp8, moe, quant, deepep, config, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22128 - Allow piecewise CUDA graph with speculative decoding

- 链接：https://github.com/sgl-project/sglang/pull/22128
- 状态/时间：`merged`，created 2026-04-05, merged 2026-04-17；作者 `narutolhy`。
- 代码 diff 已读范围：`4` 个文件，`+272/-18`；代码面：kernel, scheduler/runtime, tests/benchmarks；关键词：cuda, spec, quant, attention, cache, config, eagle, expert, fp8, lora。
- 代码 diff 细节：
  - `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py` added +243/-0 (243 lines); hunk: +"""Test piecewise CUDA graph coexisting with speculative decoding.; 符号: TestPCGWithMTP, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/server_args.py` modified +15/-18 (33 lines); hunk: def _handle_piecewise_cuda_graph(self):; 符号: _handle_piecewise_cuda_graph
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +10/-0 (10 lines); hunk: def can_run(self, forward_batch: ForwardBatch):; 符号: can_run
  - `python/sglang/srt/model_executor/model_runner.py` modified +4/-0 (4 lines); hunk: def init_piecewise_cuda_graphs(self):; 符号: init_piecewise_cuda_graphs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`；patch 关键词为 cuda, spec, quant, attention, cache, config。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22268 - [Bugfix] Fix prepare_qkv_latent bypassing LoRA adapters in DeepSeek V2/V3

- 链接：https://github.com/sgl-project/sglang/pull/22268
- 状态/时间：`open`，created 2026-04-07；作者 `SuperMarioYL`。
- 代码 diff 已读范围：`1` 个文件，`+5/-0`；代码面：model wrapper；关键词：kv, lora。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunk: def prepare_qkv_latent(; 符号: prepare_qkv_latent
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 kv, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22316 - [Reland] DeepSeek-R1-0528-w4a8: DeepEP Low Latency Dispatch Adopts FP8 Communication

- 链接：https://github.com/sgl-project/sglang/pull/22316
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-10；作者 `xieminghe1`。
- 代码 diff 已读范围：`5` 个文件，`+91/-12`；代码面：MoE/router, quantization, kernel；关键词：fp8, moe, quant, deepep, config, topk, triton, expert, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +73/-0 (73 lines); hunk: def silu_and_mul_masked_post_per_tensor_quant_fwd(; 符号: silu_and_mul_masked_post_per_tensor_quant_fwd, _fp8_per_token_quant_to_per_tensor_quant_kernel, fp8_per_token_to_per_tensor_quant_triton
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +15/-7 (22 lines); hunk: deepep_permute_triton_kernel,; def cutlass_w4a8_moe_deepep_normal(; 符号: cutlass_w4a8_moe_deepep_normal, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll, cutlass_w4a8_moe_deepep_ll
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3 (3 lines); hunk: def forward_cutlass_w4afp8_masked(; 符号: forward_cutlass_w4afp8_masked
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +2/-1 (3 lines); hunk: def apply_deepep_ll(; 符号: apply_deepep_ll
  - `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` modified +1/-1 (2 lines); hunk: def _dispatch_core(; 符号: _dispatch_core
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；patch 关键词为 fp8, moe, quant, deepep, config, topk。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/ep_moe/kernels.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22323 - [Lora] Lora quat info re-factor and support deepseekv3 mla lora

- 链接：https://github.com/sgl-project/sglang/pull/22323
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-09；作者 `yushengsu-thu`。
- 代码 diff 已读范围：`16` 个文件，`+458/-80`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：moe, config, lora, triton, expert, kv, quant, mla, attention, cache。
- 代码 diff 细节：
  - `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py` added +156/-0 (156 lines); hunk: +# Copyright 2023-2025 SGLang Team; 符号: kl_v2, get_prompt_logprobs, TestLoRADeepSeekV3BaseLogprobDiff, test_lora_deepseek_v3_base_logprob_accuracy
  - `python/sglang/srt/lora/layers.py` modified +91/-7 (98 lines); hunk: ColumnParallelLinear,; def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):; 符号: slice_lora_b_weights, ReplicatedLinearWithLoRA, __init__, set_lora_info
  - `python/sglang/srt/lora/utils.py` modified +41/-3 (44 lines); hunk: def get_hidden_dim(; def get_normalized_target_modules(; 符号: get_hidden_dim, get_normalized_target_modules, get_stacked_multiply, get_target_module_name
  - `python/sglang/srt/layers/quantization/fp8.py` modified +21/-20 (41 lines); hunk: def create_moe_runner(; def apply(; 符号: create_moe_runner, get_triton_quant_info, apply, apply
  - `python/sglang/srt/lora/lora.py` modified +29/-1 (30 lines); hunk: def _process_weight(self, name: str, loaded_weight: torch.Tensor):; def normalize_gate_up_proj(; 符号: _process_weight, _normalize_weights, normalize_qkv_proj, normalize_gate_up_proj
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py`；patch 关键词为 moe, config, lora, triton, expert, kv。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/lora/test_lora_deepseek_v3_base_logprob_diff.py`, `python/sglang/srt/lora/layers.py`, `python/sglang/srt/lora/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22933 - [CPU] expand the interface of shared_expert without scaling factor

- 链接：https://github.com/sgl-project/sglang/pull/22933
- 状态/时间：`merged`，created 2026-04-16, merged 2026-04-21；作者 `mingfeima`。
- 代码 diff 已读范围：`9` 个文件，`+313/-623`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：moe, topk, expert, kv, fp8, test, quant, awq, cache, scheduler。
- 代码 diff 细节：
  - `sgl-kernel/csrc/cpu/moe_int4.cpp` modified +10/-176 (186 lines); hunk: #include "common.h"; 符号: int, int, int, int
  - `sgl-kernel/csrc/cpu/moe.h` added +173/-0 (173 lines); hunk: +#pragma once; 符号: int, int, int, int
  - `sgl-kernel/csrc/cpu/moe.cpp` modified +25/-119 (144 lines); hunk: +#include "moe.h"; namespace {; 符号: int, int, int, int64_t
  - `sgl-kernel/csrc/cpu/moe_fp8.cpp` modified +6/-136 (142 lines); hunk: #include "common.h"; void shared_expert_fp8_kernel_impl(; 符号: int, int, int
  - `sgl-kernel/csrc/cpu/moe_int8.cpp` modified +6/-108 (114 lines); hunk: #include "common.h"; void shared_expert_int8_kernel_impl(; 符号: int, int, int
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp`；patch 关键词为 moe, topk, expert, kv, fp8, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/cpu/moe_int4.cpp`, `sgl-kernel/csrc/cpu/moe.h`, `sgl-kernel/csrc/cpu/moe.cpp` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22950 - [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373)

- 链接：https://github.com/sgl-project/sglang/pull/22950
- 状态/时间：`closed`，created 2026-04-16, closed 2026-04-21；作者 `Wen-xuan-Xu`。
- 代码 diff 已读范围：`11` 个文件，`+597/-64`；代码面：scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, kv, cuda, eagle, test, attention, spec。
- 代码 diff 细节：
  - `test/registered/unit/mem_cache/test_radix_cache_thinking.py` added +238/-0 (238 lines); hunk: +import unittest; 符号: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py` added +220/-0 (220 lines); hunk: +import unittest; 符号: _MockReqToTokenPool:, __init__, write, _MockAllocator:
  - `python/sglang/srt/mem_cache/mamba_radix_cache.py` modified +62/-50 (112 lines); hunk: from numpy import float64; MatchPrefixParams,; 符号: cache_finished_req, _skip_cache_unfinished_req, _skip_cache_unfinished_req
  - `python/sglang/srt/mem_cache/radix_cache_cpp.py` modified +27/-14 (41 lines); hunk: MatchPrefixParams,; def cache_finished_req(self, req: Req, is_insert: bool = True):; 符号: cache_finished_req, cache_unfinished_req, cache_unfinished_req, pretty_print
  - `python/sglang/srt/mem_cache/common.py` modified +22/-0 (22 lines); hunk: def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; 符号: alloc_for_decode, maybe_strip_thinking_tokens, release_kv_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py`；patch 关键词为 cache, kv, cuda, eagle, test, attention。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23195 - [Bugfix] Guard .weight access in DeepseekV2AttentionMLA for AWQ / compressed-tensors

- 链接：https://github.com/sgl-project/sglang/pull/23195
- 状态/时间：`open`，created 2026-04-20；作者 `JasonLeviGoodison`。
- 代码 diff 已读范围：`4` 个文件，`+138/-14`；代码面：model wrapper, attention/backend, tests/benchmarks；关键词：attention, kv, mla, cuda, fp8, awq, config, marlin, moe, quant。
- 代码 diff 细节：
  - `test/registered/unit/models/test_deepseek_v2_attention_mla.py` added +111/-0 (111 lines); hunk: +import unittest; 符号: TestDeepseekV2AttentionMLA, _make_attn, test_get_fused_qkv_a_proj_weight_returns_none_when_missing, test_can_use_min_latency_fused_a_gemm_preserves_bf16_path
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-9 (27 lines); hunk: class DeepseekV2AttentionMLA(; def __init__(; 符号: DeepseekV2AttentionMLA, _get_fused_qkv_a_proj_weight, _can_use_min_latency_fused_a_gemm, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` modified +5/-4 (9 lines); hunk: def init_mla_fused_rope_cpu_forward(self: DeepseekV2AttentionMLA):; 符号: init_mla_fused_rope_cpu_forward
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` modified +4/-1 (5 lines); hunk: def _dispatch_mla_subtype(attn, forward_batch):; 符号: _dispatch_mla_subtype
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py`；patch 关键词为 attention, kv, mla, cuda, fp8, awq。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/models/test_deepseek_v2_attention_mla.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_cpu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23257 - Fix double-reduce in DeepseekV2MoE with flashinfer_cutedsl + EP + DP-attention

- 链接：https://github.com/sgl-project/sglang/pull/23257
- 状态/时间：`open`，created 2026-04-20；作者 `yhyang201`。
- 代码 diff 已读范围：`2` 个文件，`+5/-0`；代码面：model wrapper, attention/backend, MoE/router, scheduler/runtime；关键词：flash, moe, attention, config, cuda, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-0 (3 lines); hunk: from sglang.srt.layers.moe import (; def forward_normal_dual_stream(; 符号: forward_normal_dual_stream, _post_combine_hook
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` modified +2/-0 (2 lines); hunk: def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:; 符号: ensure_cutedsl_wrapper
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py`；patch 关键词为 flash, moe, attention, config, cuda, fp4。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutedsl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23315 - Opt-in strip of thinking tokens from radix cache

- 链接：https://github.com/sgl-project/sglang/pull/23315
- 状态/时间：`merged`，created 2026-04-21, merged 2026-04-21；作者 `hnyls2002`。
- 代码 diff 已读范围：`4` 个文件，`+72/-4`；代码面：scheduler/runtime, tests/benchmarks；关键词：cache, kv, spec, cuda, scheduler, test。
- 代码 diff 细节：
  - `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py` modified +52/-1 (53 lines); hunk: from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType; def test_cache_finished_req_insert(self):; 符号: test_cache_finished_req_insert, test_cache_finished_req_strips_thinking, test_cache_finished_req_no_insert
  - `python/sglang/srt/managers/schedule_batch.py` modified +9/-2 (11 lines); hunk: def output_ids_through_stop(self) -> List[int]:; def pop_overallocated_kv_cache(self) -> Tuple[int, int]:; 符号: output_ids_through_stop, _cache_commit_len, pop_committed_kv_cache, pop_overallocated_kv_cache
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
  - `python/sglang/srt/mem_cache/common.py` modified +3/-1 (4 lines); hunk: def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = Tr; 符号: release_kv_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, kv, spec, cuda, scheduler, test。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23336 - [SPEC V2][2/N] feat: adaptive spec support spec v2

- 链接：https://github.com/sgl-project/sglang/pull/23336
- 状态/时间：`open`，created 2026-04-21；作者 `alphabetc1`。
- 代码 diff 已读范围：`6` 个文件，`+193/-10`；代码面：multimodal/processor, scheduler/runtime；关键词：spec, eagle, cuda, scheduler, attention, processor, config, kv, moe, topk。
- 代码 diff 细节：
  - `python/sglang/srt/speculative/eagle_worker_v2.py` modified +173/-0 (173 lines); hunk: from sglang.srt.managers.schedule_batch import ModelWorkerBatch; def __init__(; 符号: __init__, __init__, target_worker, forward_batch_generation
  - `python/sglang/srt/speculative/eagle_info_v2.py` modified +8/-4 (12 lines); hunk: def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):; 符号: prepare_for_decode, prepare_for_decode, prepare_for_v2_draft
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +10/-1 (11 lines); hunk: def _resolve_spec_overlap_token_ids(; 符号: _resolve_spec_overlap_token_ids
  - `python/sglang/srt/speculative/adaptive_spec_params.py` modified +0/-5 (5 lines); hunk: def adaptive_unsupported_reason(server_args: ServerArgs) -> str \| None:; 符号: adaptive_unsupported_reason
  - `python/sglang/srt/managers/utils.py` modified +1/-0 (1 lines); hunk: class GenerationBatchResult:; 符号: GenerationBatchResult:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`；patch 关键词为 spec, eagle, cuda, scheduler, attention, processor。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：99；open PR 数：6。
- 仍需跟进的 open PR：[#21529](https://github.com/sgl-project/sglang/pull/21529), [#21531](https://github.com/sgl-project/sglang/pull/21531), [#22268](https://github.com/sgl-project/sglang/pull/22268), [#23195](https://github.com/sgl-project/sglang/pull/23195), [#23257](https://github.com/sgl-project/sglang/pull/23257), [#23336](https://github.com/sgl-project/sglang/pull/23336)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
