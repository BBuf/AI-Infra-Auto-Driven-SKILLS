# SGLang DeepSeek V3.2 支持与优化时间线

本文基于 SGLang `origin/main` 最新快照 `929e00eea`、sgl-cookbook `origin/main` 快照 `8ec4d03`，以及 DeepSeek V3.2 / DSA / NSA 相关 merged 和 open PR 的 patch 阅读结果整理。范围覆盖 DeepSeek-V3.2-Exp、DeepSeek-V3.2、DeepSeek-V3.2-Speciale、DeepSeek-V3.2-NVFP4、DeepSeek-V3.2-MXFP4，以及这些模型在 SGLang 中的 DSA/NSA、Indexer、KV cache、Context Parallel、MTP、IndexCache、DSML parser、AMD/NPU/Blackwell 后端和测试文档演进。

结论：截至 `929e00eea`，DeepSeek V3.2 已经不是普通 DeepSeek V3 新 checkpoint，而是通过 `is_deepseek_nsa(config)` 进入 DSA/NSA 稀疏注意力体系。当前主线入口是 `DeepseekV32ForCausalLM`，核心运行面是 `nsa_indexer.py`、`nsa_backend.py`、`deepseek_v2.py`、`deepseek_nextn.py` 和 `server_args.py`。V3.2 已经具备基础 DSA、NSA backend auto-selection、BF16/FP8 KV cache、MTP、CP、PP、DP attention、TRTLLM/FlashMLA/FA3/TileLang/AITER backend、NVFP4、AMD MXFP4/FP8 KV、NPU、HiSparse/HiCache、IndexCache 和 DSML tool parser 支持。新增运行时内容包括 CP all-reduce fusion、`moe_dp_size=1` 与 `attention_cp_size` 组合、adaptive EAGLE、PCG + speculative decoding、shared `deepseek_nextn.py` 变化和 thinking token radix-cache strip。后续需要跟进的是 NSA PCG、spec v2 adaptive spec、CPU/GPU sparse KV 调度、TP-SP、V3.2 DCP、CP AMD round-robin、IndexCache/top-k backend、short-seq dense fallback、partial JSON parser 和 HiCache/3FS 等 open 方向。

## 1. 时间线总览

| 创建日期 | PR | 状态 | 主线 | 作用 |
| --- | ---: | --- | --- | --- |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | PD | 给 Qwen3-Next 和 DeepSeek V3.2 Exp 等 hybrid model 增加 PD 支持。 |
| 2025-09-29 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | bring-up | 支持 DeepSeek V3.2 Exp，新增 NSA backend、Indexer、sparse attention plumbing、KV quant/dequant、memory pool、runner 和测试。 |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | sparse KV | 支持 GQA/DSA sparse attention 的 CPU/GPU KV cache 调度。 |
| 2025-10-12 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | bugfix | 修复 Qwen3/DSV3/DSV3.2 模型支持。 |
| 2025-10-15 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | MTP | 给 DSV3.2 增加 MTP。 |
| 2025-10-20 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | docs | 增加 DeepSeek V3.2 文档。 |
| 2025-10-21 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | NSA tests | 增加 V3.2 NSA backend 测试。 |
| 2025-10-24 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | Indexer | 给 NSA Indexer 启用 mixed type LayerNorm kernel。 |
| 2025-10-24 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | CP | 初始支持 DeepSeek V3.2 DSA Context Parallel。 |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | template | 修复 DeepSeek template 中 tool arguments 的 dict/string 类型处理。 |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | docs | 更新 `deepseek_v32.md`。 |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | docs | 补充 V3.2 MHA short-seq prefill 文档。 |
| 2025-11-20 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | TP/DP attention | 启用 pure TP 和 partial DP attention。 |
| 2025-11-23 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | Indexer perf | 用 fused Triton kernels 优化 NSA Indexer K/S buffer 访问。 |
| 2025-11-26 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | CP perf | 优化 CP，支持 fused MoE、multi-batch 和 FP8 KV cache。 |
| 2025-12-06 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | NPU CP | 给 NPU 增加 V3.2 CP 支持。 |
| 2025-12-07 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | NPU perf | 增加 V3.2 NPU 优化。 |
| 2025-12-14 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | MTP tests | 增加 pure TP + MTP 测试。 |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | spec overlap | 支持 overlap speculative decoding + NSA。 |
| 2025-12-18 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | NPU | 支持 NPU MLA prolog。 |
| 2025-12-27 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | env cleanup | 清理 V3.2 环境变量。 |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | CP bugfix | 修复 V3.2 CP。 |
| 2025-12-30 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | CP guard | 在 PD decode mode 下对 V3.2 CP 加 assert。 |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | V32/CP updates | 多项 DeepSeek V32 和 CP 更新。 |
| 2026-01-02 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | refactor | 重构 DeepSeek attention backend handlers 和 forward method 定义。 |
| 2026-01-04 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | PP/CP | context pipeline 启用时支持并优化 PP。 |
| 2026-01-07 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | Indexer overlap | dual-stream decode 中 overlap Indexer `weights_proj`。 |
| 2026-01-11 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | AWQ loading | 修复 DeepSeek-V3.2-AWQ 加载。 |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | docs | 增加 V3.2 CP+PP 文档。 |
| 2026-01-12 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | MTP perf | 优化 MTP decode CUDA batch sizes 和 NSA 实现。 |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | NPU bugfix | 修复 V3.2 weight cast bug。 |
| 2026-01-13 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | NPU bugfix | 修复 V3.2 和 DSVL2 NPU 问题。 |
| 2026-01-14 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | Indexer/FA3 | 修复 slice indexer 和无法 CUDA graph 时的 FA3 padding。 |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | MoE tuning | 为 V3.1/V3.2 增加 H20/H20-3E fused MoE configs。 |
| 2026-01-23 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | NVFP4 | 更新 V3.2 NVFP4 checkpoint 测试和文档。 |
| 2026-01-23 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | TRTLLM NSA | 修复 target_verify/draft_extend 里的 TRT-LLM NSA。 |
| 2026-01-25 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | Indexer overlap | overlap Indexer q/k projection 和 activation quant。 |
| 2026-01-26 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | AMD docs | 更新 V3.2 AMD GPU 文档并统一 ROCm TileLang build。 |
| 2026-02-05 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | CP scale | `get_index_k_scale_buffer` 支持 CP。 |
| 2026-02-07 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | NVFP4/TRTLLM | DeepSeek V3.2 NVFP4 支持 NSA TRTLLM sparse MLA FP8。 |
| 2026-02-10 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | bugfix | 修复 V3.2 bug。 |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | CP default | 将 CP token split 默认改成 `round-robin-split`。 |
| 2026-02-16 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | MoE tune | 将 DeepSeek3.2 和 GLM-MoE-DSA 加入 MoE tune。 |
| 2026-02-17 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | FP8 KV | 修复 both-TRTLLM MHA one-shot 下 NSA FP8 KV cache path。 |
| 2026-02-18 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | AMD MTP | 修复 MI35x V3.2 MTP nightly。 |
| 2026-02-19 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | spec bugfix | 修复 target_verify 中 NSA backend page-table overflow。 |
| 2026-02-20 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | quality | 避免 `weights_proj` 中 FP32 精度损失。 |
| 2026-02-20 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | MTP/CP | 修复 MTP 和 CP 兼容性。 |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | MLA refactor | 迁移 DeepSeek MLA forward method。 |
| 2026-02-22 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | JIT kernel | 增加 NSA fused store indexer K cache JIT kernel。 |
| 2026-02-25 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | 128K bugfix | 修复 128K seqlen 下 `get_k_and_s_triton` bug。 |
| 2026-02-25 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | MTP/CP | 修复 EAGLE NextN 中 NSA CP positions mismatch。 |
| 2026-02-26 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | qlora/ag | 给 V3.2 增加 `mla_ag_after_qlora`。 |
| 2026-02-28 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | MTP metadata | 优化 MTP 下 NSA backend metadata。 |
| 2026-03-05 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | AMD TileLang | 给 V3.2 MI355/MI300 增加 TileLang sparse forward。 |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | NVFP4 default | 调整 V3.2 NVFP4 TP4 默认设置。 |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | docs | 在 support matrix 中加入 DSA/NSA attention backend。 |
| 2026-03-12 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | CP perf | overlap NSA-CP key all-gather 和 query computation。 |
| 2026-03-13 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | EAGLE3/DP | 修复 Attn-DP 模式下 DeepSeek Eagle3。 |
| 2026-03-15 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FP8 KV offset | flashmla_sparse + FP8 KV cache 时计算 `topk_indices_offset`。 |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | AMD accuracy | 修复 MI355 上 V3.2 精度。 |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | FP4 test | 修复 DeepSeek V32 FP4 test。 |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | revert | 回滚 `#20984`。 |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | CP tests | 修复 CP in-seq-split 并更新测试。 |
| 2026-03-24 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | CP/all-reduce | 支持 context parallel 下的 all-reduce fusion。 |
| 2026-03-24 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | HiCache | mooncake backend 支持 DSA 和 mamba hybrid model。 |
| 2026-03-24 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | B200+DP perf | 绕过 B200 + DP 下 DSA 性能下降。 |
| 2026-03-25 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | IndexCache | 给 DeepSeek V3.2 启用 IndexCache。 |
| 2026-03-26 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | NPU docs | 更新 V3.2 NPU 部署文档。 |
| 2026-03-27 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | AMD FP8 KV | 给 NSA TileLang 启用 FP8 KV cache 和 FP8 attention kernel。 |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | CI | 将 V3.2 CP test 移到 DeepEP suite。 |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | MTP/spec | 给 EAGLE top-k=1 增加自适应 `speculative_num_steps`。 |
| 2026-03-31 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | TRTLLM prefill | DSA prefill batch 支持 TRTLLM sparse MLA kernel。 |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | Blackwell default | Blackwell DSA 默认使用 TRTLLM kernels。 |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | CP topology | 支持 `moe_dp_size = 1` 搭配不同 `attention_cp_size`。 |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | HiSparse guard | HiSparse 暂时只允许 DSA model。 |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | PCG/spec | 允许 piecewise CUDA graph 和 speculative decoding 同时使用。 |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | docs | 改进 DeepSeek V3.2 / GLM-5 文档。 |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | Indexer perf | 减少 NSA Indexer 中不必要的 kernels 和 copies。 |
| 2026-04-07 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | AMD perf | AMD/HIP NSA 中 BF16 从 RMSNorm 直通，避免 FP8 dequant。 |
| 2026-04-08 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | Hopper FP8 KV | 增加 Hopper FP8 FlashMLA KV padding。 |
| 2026-04-08 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | AR fusion | 给 DSA model 启用 all-reduce fusion。 |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | AMD LayerNorm | 使用 AITER CK LayerNorm2D 降低 NSA Indexer kernel 数。 |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | HiSparse CI | 增加 HiSparse-DSA nightly CI。 |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | DSA bugfix | 修复多个 DSA model bug。 |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | AMD Indexer perf | 融合 weights_proj 和 K-cache store，减少 NSA Indexer kernels。 |
| 2026-04-16 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | CP refactor | 将 NSA utils 去重到 CP utils。 |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | reasoning cache | 探索 parser-gated 两阶段 reasoning radix-cache stripping，已关闭。 |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | shared NextN | 为 GLM-5 MXFP4 启用 MTP，改动共享 `deepseek_nextn.py`。 |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | reasoning cache | 增加可选的 thinking token radix-cache strip。 |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | spec v2 | 把 adaptive speculative decoding 扩展到 spec v2。 |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | PCG | 支持 NSA 的 piecewise CUDA graph。 |

## 1.1 主时间线之外的 V3.2 相关 PR

主时间线之外还包括一组和 V3.2/DSA/NSA/tool/parser/平台后端直接相关的 PR：

- 早期 bring-up polish：`#11063`、`#11194`、`#11308`、`#11309`、`#11450`、`#11557`、`#11565`、`#11682`、`#11815`、`#11835`。它们分别覆盖 V3.2 tool template、fast-topk、基础测试、KV cache 估算、NSA act quant kernel、默认 config、`_get_logits_head_gate` torch.compile、Indexer cleanup、ragged fast-topk transform 和 MTP CI。
- Short-seq MHA / Indexer 修复：`#11892`、`#12094`、`#12582`、`#12583`、`#12645`、`#12788`、`#12816`、`#12964`、`#13022`、`#13459`、`#13544`。这些 PR 补齐 adaptive MHA short-seq prefill、Indexer `wk+weight_proj`、top-k row starts、Indexer accuracy、KV-buffer shape、B200 short-seq MHA、extend-without-spec skip logits、MHA FP8、NSA `torch.cat` compile、Indexer FP32 权重投影和 NSA dispatch 集中化。
- DSML/tool/parser 路径：`#14304`、`#14307`、`#14353`、`#14573`、`#14750`、`#15064`、`#15278`、`#16091`、`#17951`、`#18126`、`#18174`。它们覆盖 OpenAI developer role、DS32 role 支持、encoder 错误处理、无参数函数 streaming bug、function-call 参数 streamlining、默认 drop_thinking、streaming tool-call 输出、JSON 参数 streaming、tool-call nightly、`encode_messages` 修复和 malformed JSON 容错。
- NSA backend / metadata / sparse-cache 工作：`#14781`、`#14901`、`#15040`、`#15086`、`#15242`、`#15429`、`#16520`、`#16758`、`#16841`、`#17205`、`#17554`、`#18319`。这些 PR 补充 multi-step speculative metadata、prefill TBO、paged MQA logits metadata 初始化、PP + radix-cache assertion、FlashMLA sparse FP8、V1 MTP 修复、BaseIndexerMetadata 方法、TRTLLM NSA BF16 KV、AMD CUDA graph/FP8 RMSNorm、Indexer `weight_proj` MMA 优化、NSA multi-spec fused metadata kernels 和 AMD TileLang 默认 NSA dispatch。
- HiSparse/HiCache 与平台修复：`#14741`、`#17409`、`#17518`、`#17523`、`#17633`、`#18297`、`#18526`、`#20343`、`#21932`、`#22238`。这些 PR 连接 sparse interface、fused-MoE config lookup、AMD dtype mismatch、MI325 CI、transformers v5 兼容、AITER NSA CUDA graph、HiSparse、decode backup scheduling 和 HiSparse 文档。
- 额外 open PR：`#14332`、`#14524`、`#15322`、`#18094`、`#18542`、`#19987`、`#20534`、`#21623`、`#22792`、`#23268`。分别指向无 DSML tag 的 V32 tool-call parsing、NSA backend test suite、`o_proj` TP、V3.2 PCG、EAGLE3+NSA CP aux hidden state、TileLang NSA FP8 KV、CP prefill gather FP8 K/K-scale、`encoding_dsv32.py` 单测、AITER `indexer_k_quant_and_cache`、以及 NPU NSA CP + prefix cache 精度。
- closed / superseded 历史：`#11109`、`#11596`、`#11761`、`#12017`、`#12052`、`#13531`、`#13546`、`#14619`、`#14904`、`#15051`、`#15217`、`#15310`、`#15807`、`#16079`、`#16881`、`#17024`、`#17199`、`#17310`、`#17647`。这些不能当当前主线支持，但排查历史和 open PR 继承关系时应作为背景。
- 运行时增量：`#21249` 和 `#22003` 补齐 CP/all-reduce 与拓扑约束；`#21599`、`#22128`、`#23336` 补齐 speculative decoding 自适应和 PCG 组合；`#23219` 是 GLM-5 MXFP4 方向，但改动共享 `deepseek_nextn.py`，因此属于 DSA/NextN 邻近历史；`#22950`、`#23315` 则区分 closed/current 的 thinking radix-cache strip。

## 2. V3.2 的本质：DSA/NSA 运行面

V3.2 和 V3/R1 的最大区别是 attention。模型配置满足 `is_deepseek_nsa(config)` 时，SGLang 将其视为 DSA/NSA 模型。docs 里常称 DSA，代码里主要叫 NSA。

`#11061` 是整个 V3.2 支持的基石。该 PR 增加了 `DeepseekV32ForCausalLM`，并引入完整的稀疏注意力路径：

- model config 中识别 DSA/NSA，并读取 `index_topk`、indexer head 数、index head dim 等字段。
- `server_args.py` 为 DSA 设置 `attention_backend = "nsa"`。
- `nsa_backend.py` 新增 `NativeSparseAttnBackend`。
- `nsa_indexer.py` 新增 Indexer，负责生成稀疏 attention 的 top-k 索引。
- `transform_index.py`、Triton/TileLang kernels 负责把 top-k 转成 backend 需要的 paged/ragged 结构。
- `quant_k_cache.py` / `dequant_k_cache.py` 处理 FP8 K cache。
- memory pool、model runner、CUDA graph、forward batch metadata 都为 NSA 增加了必要字段。

因此，V3.2 的性能或正确性问题通常不能只看 `deepseek_v2.py`。运行时调用链是：model layer 调 Indexer 生成/复用 top-k，NSA backend 依据 top-k 和 cache seqlens 构造 metadata，最后再分发给 TRTLLM、FlashMLA、FA3、TileLang 或 AITER。

## 3. Server args：默认值是 V3.2 支持的一部分

当前 `server_args.py` 对 DSA model 有专门逻辑：

- 如果没有手动设置 attention backend，会把 `attention_backend` 设置为 `nsa`。
- 如果用户没设置 `SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`，会把它设置成模型的 `index_topk`。
- DSA KV cache dtype 在 SM100 默认 `fp8_e4m3`，其他设备默认 `bfloat16`。
- DSA 主线只允许 `bfloat16` 或 `fp8_e4m3` KV cache dtype。
- ROCm 上 NSA prefill/decode 默认 TileLang。
- FP8 KV + SM100 默认 TRTLLM。
- FP8 KV + Hopper 可以走 `flashmla_kv`。
- BF16 KV + SM100 可以用 `flashmla_sparse` + `trtllm`，BF16 KV + Hopper 可以用 `flashmla_sparse` + `fa3`。

这意味着比较性能时必须同时记录 `--kv-cache-dtype`、`--nsa-prefill-backend`、`--nsa-decode-backend` 和硬件。用户指定了 NSA backend 但 KV dtype 仍是 `auto` 时，结果可能被默认值影响。

## 4. NSA Indexer：V3.2 优化最密集的区域

`nsa_indexer.py` 是 DSA 的核心热点。它做几件事：

- 对 hidden states 做 q/k 相关 projection。
- 通过 `weights_proj` 计算索引权重。
- 做 LayerNorm、RoPE、activation quant。
- 生成 top-k sparse indices。
- 处理 CP 下的 key all-gather、rerange、round-robin/in-seq split。
- 写入或量化 K cache。

围绕 Indexer 的优化较为密集：

- `#12044` 启用 mixed type LayerNorm kernel，避免类型转换和多余 kernel。
- `#13812` 用 fused Triton kernels 优化 K/S buffer 访问。
- `#16637` 在 dual-stream decode 中 overlap `weights_proj`。
- `#17688` overlap q/k projection 和 activation quant。
- `#19041` 避免 `weights_proj` 中 FP32 精度损失，属于质量修复，也会影响性能排查。
- `#19148` 增加 JIT NSA fused store indexer K cache。
- `#19319` 修复 128K seqlen 下 `get_k_and_s_triton` bug。
- `#22232` 减少 Indexer 中多余 kernels 和 copies。
- `#22424` AMD 上使用 AITER CK LayerNorm2D。
- `#22850` AMD 上进一步融合 `weights_proj` 和 K-cache store。

如果遇到 V3.2 首 token 慢、decode kernel 数异常、128K 长上下文错误、FP8 KV scale 错、AMD 上 Indexer kernel 太多，第一优先级通常都是读 `nsa_indexer.py` 和对应 backend kernels。

## 5. NSA backend：metadata、top-k transform 和 sparse MLA

`nsa_backend.py` 的 `NativeSparseAttnBackend` 负责把 Indexer 结果转成 attention backend 能执行的 metadata。关键字段包括：

- `nsa_cache_seqlens_int32`：裁剪到 top-k 的 cache seqlens。
- `nsa_cu_seqlens_q` / `nsa_cu_seqlens_k`：prefill/decode 所需 cumulative seqlens。
- `nsa_seqlens_expanded`：扩展后的真实 seqlens。
- `nsa_extend_seq_lens_list`：extend 阶段 CPU 侧长度。
- FlashMLA metadata、paged MQA schedule、page table、top-k offsets。

`#11936` 让 NSA backend 进入测试；`#18389` 给 NVFP4 方向加入 TRTLLM sparse MLA FP8；`#18931` 修 both-TRTLLM MHA one-shot 下 FP8 KV path；`#21783` 支持 DSA prefill batch 的 TRTLLM sparse MLA；`#21914` 把 Blackwell 默认推向 TRTLLM kernels；`#22372` 处理 Hopper FP8 FlashMLA KV padding。

如果 top-k indices 看起来正确但 attention 输出错，问题经常不在 Indexer，而在 top-k transform、cache seqlens、page-table offset 或 FP8 KV padding。

## 6. Context Parallel、PP 与 DP Attention

`#12065` 是 V3.2 CP 的起点。它同时改了 server args、pynccl、parallel state、NSA utils/backend、communicator、DP attention、schedule policy、CUDA graph、forward batch、`deepseek_v2.py`、`deepseek_nextn.py`、docs 和 tests。这说明 CP 不是简单加一个通信 op，而是贯穿调度、attention metadata 和模型 forward。

后续 CP 演进包括：

- `#13959` 支持 fused MoE、multi-batch、FP8 KV cache。
- `#16119` 修 CP bug。
- `#16156` 在 PD decode mode 下 assert V3.2 CP。
- `#16380` 在 context pipeline 启用时支持并优化 PP。
- `#18613` 将 CP token split 默认改成 `round-robin-split`。
- `#20438` overlap NSA-CP key all-gather 和 query computation。
- `#21192` 修 CP `in-seq-split` 并更新测试。
- `#21249` 支持 CP 下的 all-reduce fusion，改动 `communicator.py`、`flashinfer_comm_fusion.py`、`model_runner.py` 和 server args。
- `#22003` 让 `moe_dp_size = 1` 能搭配不同 `attention_cp_size`，需要同时读 `parallel_state.py`、`dp_attention.py` 和 CP utils。
- `#22914` 将 NSA utils 去重到 CP utils。

当前要记住的约束：

- `round-robin-split` 是默认 CP token split。
- `in-seq-split` 要求 DeepEP，并要求 `ep_size == tp_size`。
- CP in PD decode mode 受限制。
- CP 只在 V3.2/DSA 这种 `is_deepseek_nsa(config)` 场景下成立。
- all-reduce fusion 与 CP 不应视为互斥；排查时要看当前 communicator 和 fusion backend 是否启用。

open `#20360` 和 `#20531` 说明 AMD CP round-robin 和 ragged gather 仍有边界问题；open `#17185` 和 `#19609` 则分别指向 CP NSA 下的 TP `o_proj` 和 TP indexer weight。

## 7. MTP 与 speculative decoding：NSA metadata 也在关键路径

V3.2 MTP 从 `#11652` 开始，稳定化依赖一组后续修复：

- `#15088` 增加 pure TP + MTP test。
- `#15307` 支持 overlap spec + NSA。
- `#16961` 优化 MTP decode CUDA batch sizes 和 NSA implementation。
- `#17662` 修复 TRTLLM NSA 在 `target_verify` / `draft_extend` 中的问题。
- `#19016` 修复 speculative target_verify 的 page_table overflow。
- `#19062` 修复 MTP + CP 兼容性。
- `#19367` 修复 EAGLE NextN 中 NSA CP positions mismatch。
- `#19536` 优化 MTP 下 NSA backend metadata。
- `#20492` 修复 Attn-DP 模式下 DeepSeek Eagle3。
- `#21599` 让 EAGLE top-k=1 的 draft step 数自适应。
- `#22128` 允许 PCG 和 speculative decoding 共存。
- `#23219` 虽然是 GLM-5 MXFP4 MTP，但改动共享的 `deepseek_nextn.py`，要作为 DSA/NextN 邻近历史阅读。
- open `#23336` 把 adaptive spec 推到 spec v2 workers。

因此，V3.2 的 MTP bug 不能只查 `deepseek_nextn.py`。它可能来自 NSA metadata 预计算、target verify、draft extend、page table、CP positions、DP attention 或 backend auto-selection。open `#20809` 还在跟踪把 `DeepseekV32ForCausalLM` 加入 MTP draft model mapping 的方向。

## 8. 量化与平台后端：NVFP4、AMD、NPU、HiSparse

V3.2 的平台线很丰富：

NVFP4 / Blackwell：

- `#17657` 更新 V3.2 NVFP4 checkpoint 的测试和文档。
- `#18389` 增加 NSA TRTLLM sparse MLA FP8 支持。
- `#20086` 调整 TP4 下 V3.2 NVFP4 默认配置。
- `#21914` 将 Blackwell 默认设置为 TRTLLM kernels。

AMD / ROCm：

- `#17783` 更新 AMD GPU docs 并统一 ROCm TileLang build。
- `#19945` 增加 MI355/MI300 TileLang sparse forward。
- `#20840` 修 MI355 精度。
- `#21511` 启用 NSA TileLang 的 FP8 KV cache 和 FP8 attention kernel。
- `#22258` 让 BF16 从 RMSNorm 直通，避免 FP8 dequant。
- `#22424` 用 CK LayerNorm2D 减少 Indexer kernel launches。
- `#22850` 融合 weights_proj 和 K-cache store。

NPU：

- `#14541` 加 V3.2 CP。
- `#14572` 加 NPU 优化。
- `#15381` 支持 NPU MLA prolog。
- `#16990` 修 weight cast bug。
- `#17007` 修 V3.2 / DSVL2 NPU bug。
- `#21468` 更新 NPU 部署文档。

HiSparse / HiCache：

- `#21259` 让 mooncake backend 支持 DSA 和 mamba hybrid model。
- `#22065` 把 HiSparse 检查限制在 DSA model。
- `#22425` 增加 HiSparse-DSA nightly CI。
- open `#23241` 继续推进 3FS backend 支持 DSA/mamba。

## 9. IndexCache：跳过部分层 top-k

`#21405` 给 V3.2 启用 IndexCache。当前 `deepseek_v2.py` 中每层会设置：

- `skip_topk`：当前层是否跳过 top-k 计算并复用上一层 top-k。
- `next_skip_topk`：下一层是否会复用当前层 top-k。
- `index_topk_freq`：按频率决定哪些层计算 top-k。
- `index_topk_pattern`：用显式字符串控制每层是计算还是跳过。

如果没有 pattern，逻辑大致是按 `index_topk_freq` 周期计算；如果给了 pattern，`S` 表示 skip，其他字符表示计算。`test_deepseek_v32_indexcache.py` 覆盖了 `index_topk_freq=4` 和一个长的 `index_topk_pattern`，并用 GSM8K 阈值 `0.935` 保证复用 top-k 不破坏精度。

IndexCache 不是单纯性能 knob。它改变了哪些层生成 top-k，哪些层复用 top-k，因此必须做模型精度验证。

## 10. DSML tool parser 与 reasoning 交互

标准 DeepSeek V3.2 使用 `DeepSeekV32Detector`，tool-call 格式是 DSML：

```text
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```

detector 支持两种参数形态：

- XML parameter tags。
- invoke 内直接 JSON。

streaming parser 会先发 tool name，再通过 previous arguments 和 common prefix 计算参数 diff。`_parse_parameters_from_xml(..., allow_partial=True)` 用 partial JSON parser 处理未闭合 JSON 或未闭合 parameter tag。

cookbook 当前写法需要区分：DeepSeek-V3.2-Exp 的 tool path 可能使用 `deepseekv31` 加 `tool_chat_template_deepseekv32.jinja`；标准 DeepSeek-V3.2 使用 `--tool-call-parser deepseekv32` 并移除自定义 chat template。DeepSeek-V3.2-Speciale 不支持 tool calling。

open `#21179` 指出 reasoning parser 可能吞掉 V3.2 tool-call markers；open `#21546` 则修 partial parsing 遇到 malformed JSON 的异常处理。

## 10.1 Thinking radix cache：parser 之外的缓存层语义

V3.2 的 reasoning/tool 输出除了 parser 外，还要看 prefix cache 是否复用 thinking tokens。`#22950` 是 closed 的 parser-gated reasoning cache strip 早期方案；当前主线是 `#23315`，它在 `server_args.py` 加 opt-in flag，并在 `schedule_batch.py`、`mem_cache/common.py` 中支持从 radix-cache entry 里剥离 thinking tokens。

这条线和 DSML parser 是两层问题：如果 V3.2 tool-call marker 被 reasoning parser 吞掉，要看 `deepseekv32_detector.py` / `reasoning_parser.py`；如果多轮请求复用了不该复用的 `<think>` / `</think>` 前缀，要看 `#23315` 的 radix-cache strip 行为。

## 11. 当前验证面与未合入方向

当前验证面：

- `test/registered/8-gpu-models/test_deepseek_v32.py`：DP8、DP8+MTP、TP8、TP8+MTP，带 `deepseekv32` tool parser 和 `deepseek-v3` reasoning parser。
- 同文件的 `test_deepseek_v32_nsa_backends`：H200 上测试 `flashmla_sparse+flashmla_kv`、`fa3+fa3` 和 FP8 KV cache。
- 同文件的 B200 GPQA 测试：reasoning mode，GPQA baseline `0.83`。
- `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`：`index_topk_pattern` 和 `index_topk_freq=4`。
- `test/manual/test_deepseek_chat_templates.py`：DeepSeek V3/V3.1/V3.2 template 参数类型。

需要跟进的 open PR：

- `#11191`：DSA sparse attention 与 CPU/GPU KV cache scheduling。
- `#12820`：TP-SP 支持 DeepSeek V2/V3/V3.2。
- `#16148`：V3.2 W4AFP8 MTP 使用 FP8 draft model。
- `#17185`：context parallel NSA 中 TP `o_proj` linear。
- `#17761`：V3.1/V3.2 tool output 后缺 Assistant token。
- `#18167`：V3.2 DCP。
- `#18275`：NPU allgather after qlora。
- `#18733`：V3.2 PD disaggregation test。
- `#19211`：抽出 `DeepseekV32Mixin`，降低 `deepseek_v2.py` 中 V3.2/NSA 逻辑复杂度。
- `#19299`：DeepSeek weight loader 中 O(1) expert weight matching。
- `#19609`：NSA attention 中 TP indexer weight。
- `#19975`：AMD 上 V3.2 context parallel。
- `#20360`：AMD CP round-robin-split 输出异常。
- `#20531`：CP round-robin 下 NSA indexer ragged gather batch-view mismatch。
- `#20809`：MTP draft model mapping 加 `DeepseekV32ForCausalLM`。
- `#20880`：NSA model 初始化时拒绝 HiCache L3。
- `#21179`：reasoning parsing 保留 V3.2 tool-call markers。
- `#21194`：AMD AITER gfx95 路径中 DeepSeek `PPMissingLayer` 修复。
- `#21506`：V3.2 NPU torch compile。
- `#21529`：ROCm DeepSeek 架构 MXFP4 / Quark W4A4。
- `#21530`：ROCm DeepSeek-variant fused MLA decode RoPE。
- `#21546`：partial function-call parsing 捕获 MalformedJSON。
- `#21889`：AMD TileLang NSA FP4 KV cache quantization。
- `#22268`：DeepSeek MLA `prepare_qkv_latent` 绕过 LoRA adapter。
- `#22473`：short sequences 的 dense MLA decode fallback。
- `#22774`：MUSA backend 支持 DeepSeek V2/V3/R1-class layers，已于 `2026-04-24T01:59:51Z` 合入；它修改 shared DeepSeek MHA/MLA/loader 文件，V3.2 相邻路径排查仍需关注这些公共文件。
- `#22851`：新增 `--nsa-topk-backend`，接入 FlashInfer 和 PyTorch top-k。
- `#22865`：扩展 sparsity framework 支持非 NSA sparse algorithms。
- `#22938`：MLA refactor 后恢复 MI300X DeepSeek MLA 路径。
- `#23195`：AWQ/compressed-tensors 下 DeepSeek MLA `.weight` 访问保护。
- `#23241`：3FS backend 支持 DSA/mamba。
- `#23257`：`DeepseekV2MoE` 在 CuteDSL EP + DP attention 下的 double-reduce。
- `#23336`：spec v2 adaptive speculative decoding。
- `#23351`：NSA piecewise CUDA graph。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `DeepSeek V3.2`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-09-25 | [#10912](https://github.com/sgl-project/sglang/pull/10912) | merged | [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp) | attention/backend, scheduler/runtime, tests/benchmarks | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py` |
| 2025-09-29 | [#11061](https://github.com/sgl-project/sglang/pull/11061) | merged | Support DeepSeek V3.2 Exp | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-09-29 | [#11063](https://github.com/sgl-project/sglang/pull/11063) | merged | Add DeepSeek-V3.2 Tool Call Template | docs/config | `examples/chat_template/tool_chat_template_deepseekv32.jinja` |
| 2025-09-30 | [#11109](https://github.com/sgl-project/sglang/pull/11109) | closed | [Draft] Support MTP for DeepSeek-V3.2 | attention/backend, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-03 | [#11191](https://github.com/sgl-project/sglang/pull/11191) | open | [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA. | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` |
| 2025-10-03 | [#11194](https://github.com/sgl-project/sglang/pull/11194) | merged | [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2 | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-07 | [#11308](https://github.com/sgl-project/sglang/pull/11308) | merged | [CI] Add Basic Test for DeepSeek V3.2 | tests/benchmarks | `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh` |
| 2025-10-07 | [#11309](https://github.com/sgl-project/sglang/pull/11309) | merged | [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` |
| 2025-10-11 | [#11450](https://github.com/sgl-project/sglang/pull/11450) | merged | [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton | attention/backend, quantization, kernel, tests/benchmarks | `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-12 | [#11510](https://github.com/sgl-project/sglang/pull/11510) | merged | [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py` |
| 2025-10-13 | [#11557](https://github.com/sgl-project/sglang/pull/11557) | merged | Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3)) | misc | `python/sglang/srt/server_args.py` |
| 2025-10-13 | [#11565](https://github.com/sgl-project/sglang/pull/11565) | merged | [DSv32] Use torch.compile for _get_logits_head_gate | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-14 | [#11596](https://github.com/sgl-project/sglang/pull/11596) | closed | [Spec Decoding] Support MTP for dsv3.2 | attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml` |
| 2025-10-15 | [#11652](https://github.com/sgl-project/sglang/pull/11652) | merged | [Spec Decoding] Support MTP for dsv3.2 | attention/backend, kernel, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py` |
| 2025-10-15 | [#11682](https://github.com/sgl-project/sglang/pull/11682) | merged | Cleaning indexer for DeepSeek V3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-10-17 | [#11761](https://github.com/sgl-project/sglang/pull/11761) | closed | (beta)support context parallel with deepseekv3.2-DSA | misc |  |
| 2025-10-19 | [#11815](https://github.com/sgl-project/sglang/pull/11815) | merged | [DeepseekV32] Add fast_topk_transform_ragged_fused kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-10-19 | [#11835](https://github.com/sgl-project/sglang/pull/11835) | merged | [CI] Add CI test for DeepSeek V3.2 MTP | tests/benchmarks | `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py` |
| 2025-10-20 | [#11877](https://github.com/sgl-project/sglang/pull/11877) | merged | [Doc] Add documentation for DeepSeek V3.2 | docs/config | `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb` |
| 2025-10-21 | [#11892](https://github.com/sgl-project/sglang/pull/11892) | merged | DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-21 | [#11936](https://github.com/sgl-project/sglang/pull/11936) | merged | [Test] Add dsv3.2 nsa backend testing | tests/benchmarks | `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py` |
| 2025-10-23 | [#12017](https://github.com/sgl-project/sglang/pull/12017) | closed | (beta)support context parallel with deepseekv3.2-DSA | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py` |
| 2025-10-24 | [#12044](https://github.com/sgl-project/sglang/pull/12044) | merged | Enable mixed type LayerNorm kernel for NSA indexer | attention/backend, tests/benchmarks | `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-24 | [#12052](https://github.com/sgl-project/sglang/pull/12052) | closed | Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models | attention/backend | `python/sglang/srt/layers/dp_attention.py` |
| 2025-10-24 | [#12065](https://github.com/sgl-project/sglang/pull/12065) | merged | (1/n)support context parallel with deepseekv3.2-DSA | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-10-24 | [#12094](https://github.com/sgl-project/sglang/pull/12094) | merged | Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4 | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-10-25 | [#12123](https://github.com/sgl-project/sglang/pull/12123) | merged | Fix DeepSeek chat templates to handle tool call arguments type checking (#11700) | tests/benchmarks, docs/config | `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` |
| 2025-10-28 | [#12296](https://github.com/sgl-project/sglang/pull/12296) | merged | Update deepseek_v32.md | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2025-11-04 | [#12582](https://github.com/sgl-project/sglang/pull/12582) | merged | [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel | MoE/router, kernel, tests/benchmarks | `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` |
| 2025-11-04 | [#12583](https://github.com/sgl-project/sglang/pull/12583) | merged | [Deepseek V3.2] Fix accuracy bug in the Indexer | attention/backend, tests/benchmarks | `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py` |
| 2025-11-04 | [#12645](https://github.com/sgl-project/sglang/pull/12645) | merged | [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2 | scheduler/runtime | `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-06 | [#12788](https://github.com/sgl-project/sglang/pull/12788) | merged | [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100) | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-07 | [#12816](https://github.com/sgl-project/sglang/pull/12816) | merged | [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2025-11-08 | [#12868](https://github.com/sgl-project/sglang/pull/12868) | merged | [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2025-11-10 | [#12964](https://github.com/sgl-project/sglang/pull/12964) | merged | [DeepseekV3.2] Deepseek fp8 support for MHA path | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-10 | [#13022](https://github.com/sgl-project/sglang/pull/13022) | merged | [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-11-17 | [#13459](https://github.com/sgl-project/sglang/pull/13459) | merged | [Deepseek V3.2] Change indexer weights_proj to fp32 | model wrapper, attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-11-18 | [#13531](https://github.com/sgl-project/sglang/pull/13531) | closed | DeepSeek V3.2 indexer RoPE fix | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-11-18 | [#13544](https://github.com/sgl-project/sglang/pull/13544) | merged | [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-18 | [#13546](https://github.com/sgl-project/sglang/pull/13546) | closed | [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention | model wrapper, attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-11-20 | [#13646](https://github.com/sgl-project/sglang/pull/13646) | merged | [DeepSeekV3.2] Enable pure TP & Partial DP Attention | attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py` |
| 2025-11-23 | [#13812](https://github.com/sgl-project/sglang/pull/13812) | merged | [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels | attention/backend, scheduler/runtime, tests/benchmarks | `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py` |
| 2025-11-26 | [#13959](https://github.com/sgl-project/sglang/pull/13959) | merged | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | model wrapper, attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-02 | [#14304](https://github.com/sgl-project/sglang/pull/14304) | merged | [FIX][DS32]openai protocol: support openai message role: developer | misc | `python/sglang/srt/entrypoints/openai/protocol.py` |
| 2025-12-02 | [#14307](https://github.com/sgl-project/sglang/pull/14307) | merged | [SMG][DS32][fix] support dsv32, add role developer | MoE/router | `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs` |
| 2025-12-03 | [#14332](https://github.com/sgl-project/sglang/pull/14332) | open | feat: V32 tool call parsing for no-dsml tag | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-03 | [#14353](https://github.com/sgl-project/sglang/pull/14353) | merged | feat(dsv32): better error handling for DeepSeek-v3.2 encoder | misc | `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py` |
| 2025-12-06 | [#14524](https://github.com/sgl-project/sglang/pull/14524) | open | [Test] Add test suite for NSA backend | attention/backend, tests/benchmarks | `python/sglang/test/attention/test_nsa_backend.py` |
| 2025-12-06 | [#14541](https://github.com/sgl-project/sglang/pull/14541) | merged | [NPU]dsv3.2 cp for npu | attention/backend, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-07 | [#14572](https://github.com/sgl-project/sglang/pull/14572) | merged | [NPU] optimization for dsv3.2 | model wrapper, attention/backend, MoE/router, quantization | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-12-07 | [#14573](https://github.com/sgl-project/sglang/pull/14573) | merged | [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-08 | [#14619](https://github.com/sgl-project/sglang/pull/14619) | closed | [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32. | model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` |
| 2025-12-09 | [#14741](https://github.com/sgl-project/sglang/pull/14741) | merged | [1/N][Sparse With Hicache]: Add Sparse Interface | scheduler/runtime | `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` |
| 2025-12-09 | [#14750](https://github.com/sgl-project/sglang/pull/14750) | merged | [Tool Call][DSV32] Streamline function call parameters | tests/benchmarks | `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-10 | [#14781](https://github.com/sgl-project/sglang/pull/14781) | merged | [Performance] optimize NSA backend metadata computation for multi-step speculative decoding | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-11 | [#14901](https://github.com/sgl-project/sglang/pull/14901) | merged | fix ds3.2 nsa backend prefill TBO | model wrapper, attention/backend, tests/benchmarks | `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` |
| 2025-12-11 | [#14904](https://github.com/sgl-project/sglang/pull/14904) | closed | [DeepSeek V3.2] Proper drop_thinking logic | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-13 | [#15040](https://github.com/sgl-project/sglang/pull/15040) | merged | [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-13 | [#15051](https://github.com/sgl-project/sglang/pull/15051) | closed | feat(ds32): support <function_call> tag for deepseek 3.2 tool call | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-13 | [#15064](https://github.com/sgl-project/sglang/pull/15064) | merged | fix: dpskv32 chat history processing, default drop_thinking to true | misc | `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2025-12-13 | [#15086](https://github.com/sgl-project/sglang/pull/15086) | merged | [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache | attention/backend, quantization, scheduler/runtime | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` |
| 2025-12-14 | [#15088](https://github.com/sgl-project/sglang/pull/15088) | merged | [DeepSeekV3.2] Add pure TP+MTP test | tests/benchmarks, docs/config | `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md` |
| 2025-12-16 | [#15217](https://github.com/sgl-project/sglang/pull/15217) | closed | fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2025-12-16 | [#15242](https://github.com/sgl-project/sglang/pull/15242) | merged | [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations | attention/backend, kernel | `sgl-kernel/cmake/flashmla.cmake` |
| 2025-12-16 | [#15278](https://github.com/sgl-project/sglang/pull/15278) | merged | feat: DeepSeek-V3.2 Streaming tool call output | tests/benchmarks | `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-17 | [#15307](https://github.com/sgl-project/sglang/pull/15307) | merged | [Deepseek V3.2] Support Overlap Spec + NSA | attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2025-12-17 | [#15310](https://github.com/sgl-project/sglang/pull/15310) | closed | [Deepseek V3.2] Enable TRTLLM Allreduce Fusion | misc | `python/sglang/srt/server_args.py` |
| 2025-12-17 | [#15322](https://github.com/sgl-project/sglang/pull/15322) | open | dsv32 support o_proj tp | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py` |
| 2025-12-18 | [#15381](https://github.com/sgl-project/sglang/pull/15381) | merged | [NPU]DeepSeek-V3.2 support npu mlaprolog | attention/backend, quantization | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` |
| 2025-12-19 | [#15429](https://github.com/sgl-project/sglang/pull/15429) | merged | [Deepseek V3.2] Fix Deepseek MTP in V1 mode | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-25 | [#15807](https://github.com/sgl-project/sglang/pull/15807) | closed | [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side. | attention/backend, scheduler/runtime | `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py` |
| 2025-12-27 | [#15938](https://github.com/sgl-project/sglang/pull/15938) | merged | Clean Some Environment Variables for DeepSeek V32 | attention/backend, quantization, scheduler/runtime, docs/config | `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` |
| 2025-12-29 | [#16079](https://github.com/sgl-project/sglang/pull/16079) | closed | [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2025-12-29 | [#16091](https://github.com/sgl-project/sglang/pull/16091) | merged | [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format. | tests/benchmarks | `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` |
| 2025-12-30 | [#16119](https://github.com/sgl-project/sglang/pull/16119) | merged | [cp] bug fix for dsv3.2 cp | attention/backend | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2025-12-30 | [#16156](https://github.com/sgl-project/sglang/pull/16156) | merged | [cp] assert dsv3.2 cp in pd decode mode | misc | `python/sglang/srt/server_args.py` |
| 2026-01-02 | [#16305](https://github.com/sgl-project/sglang/pull/16305) | merged | Multiple updates of DeepSeek V32 and context parallel | tests/benchmarks, docs/config | `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md` |
| 2026-01-02 | [#16306](https://github.com/sgl-project/sglang/pull/16306) | merged | [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` |
| 2026-01-04 | [#16380](https://github.com/sgl-project/sglang/pull/16380) | merged | [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled | attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py` |
| 2026-01-05 | [#16520](https://github.com/sgl-project/sglang/pull/16520) | merged | fix: unimplemented methods in BaseIndexerMetadata | kernel, tests/benchmarks | `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-07 | [#16637](https://github.com/sgl-project/sglang/pull/16637) | merged | [DSv32] Overlap indexer weights_proj during dual_stream decode | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-08 | [#16758](https://github.com/sgl-project/sglang/pull/16758) | merged | [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-10 | [#16841](https://github.com/sgl-project/sglang/pull/16841) | merged | [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant | model wrapper, attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` |
| 2026-01-11 | [#16881](https://github.com/sgl-project/sglang/pull/16881) | closed | [DSv32] Add returning DSA topk indices | model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks | `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-01-11 | [#16907](https://github.com/sgl-project/sglang/pull/16907) | merged | Fix model loading for DeepSeek-V3.2-AWQ | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-01-12 | [#16916](https://github.com/sgl-project/sglang/pull/16916) | merged | add doc for dsv32 cp+pp | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-01-12 | [#16961](https://github.com/sgl-project/sglang/pull/16961) | merged | [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation | attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2026-01-13 | [#16990](https://github.com/sgl-project/sglang/pull/16990) | merged | [Ascend] fix dsv3.2 weight cast bug | quantization | `python/sglang/srt/layers/quantization/unquant.py` |
| 2026-01-13 | [#17007](https://github.com/sgl-project/sglang/pull/17007) | merged | [NPU]bugfix: fix for dsv3.2 and dsvl2 | model wrapper, attention/backend, tests/benchmarks, docs/config | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py` |
| 2026-01-13 | [#17024](https://github.com/sgl-project/sglang/pull/17024) | closed | [PD] Fix DeepSeek V3.2 indexer cache transfer | misc | `python/sglang/srt/disaggregation/prefill.py` |
| 2026-01-14 | [#17076](https://github.com/sgl-project/sglang/pull/17076) | merged | [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-01-15 | [#17133](https://github.com/sgl-project/sglang/pull/17133) | merged | [DeepSeek V3.1/V3.2] Optimize fused moe configs for H20 & H20-3E based on swapab | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128, 128]_down.json`, `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128, 128]_down.json` |
| 2026-01-16 | [#17199](https://github.com/sgl-project/sglang/pull/17199) | closed | [Feature] add feature mla_ag_after_qlora for dsv3.2 | model wrapper, attention/backend | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-16 | [#17205](https://github.com/sgl-project/sglang/pull/17205) | merged | [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-18 | [#17310](https://github.com/sgl-project/sglang/pull/17310) | closed | [TileLang] Align TileLang NSA kernel with current TileLang and stabilize output | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-01-20 | [#17409](https://github.com/sgl-project/sglang/pull/17409) | merged | [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-01-21 | [#17518](https://github.com/sgl-project/sglang/pull/17518) | merged | [HotFix]Fix dtype mismatch in nsa indexer on AMD device | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-21 | [#17523](https://github.com/sgl-project/sglang/pull/17523) | merged | [AMD] Add Kimi-K2, DeepSeek-V3.2 tests to nightly CI | quantization, tests/benchmarks | `test/registered/amd/accuracy/test_deepseek_v32_eval_amd.py`, `.github/workflows/nightly-test-amd.yml`, `test/registered/amd/perf/test_deepseek_v32_mtp_perf_amd.py` |
| 2026-01-22 | [#17554](https://github.com/sgl-project/sglang/pull/17554) | merged | Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels | attention/backend, kernel, tests/benchmarks | `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` |
| 2026-01-23 | [#17633](https://github.com/sgl-project/sglang/pull/17633) | merged | [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites | attention/backend, MoE/router, kernel, tests/benchmarks | `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py` |
| 2026-01-23 | [#17647](https://github.com/sgl-project/sglang/pull/17647) | closed | [Perf] opt nsa backend init forward metada | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-01-23 | [#17657](https://github.com/sgl-project/sglang/pull/17657) | merged | [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint | quantization, tests/benchmarks, docs/config | `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py` |
| 2026-01-23 | [#17662](https://github.com/sgl-project/sglang/pull/17662) | merged | [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-01-25 | [#17688](https://github.com/sgl-project/sglang/pull/17688) | merged | [DSv32] Overlap indexer qk projection and activation quant | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-01-26 | [#17783](https://github.com/sgl-project/sglang/pull/17783) | merged | [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build | docs/config | `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md` |
| 2026-01-29 | [#17951](https://github.com/sgl-project/sglang/pull/17951) | merged | Add tool call tests for DeepSeek V3.2 in nightly CI | model wrapper, scheduler/runtime, tests/benchmarks | `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py` |
| 2026-02-02 | [#18094](https://github.com/sgl-project/sglang/pull/18094) | open | support deepseekv3.2-piecewise-cuda-graph | model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2026-02-02 | [#18126](https://github.com/sgl-project/sglang/pull/18126) | merged | Fix dsv32 encode_messages | misc | `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` |
| 2026-02-03 | [#18174](https://github.com/sgl-project/sglang/pull/18174) | merged | [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON | misc | `python/sglang/srt/function_call/deepseekv32_detector.py` |
| 2026-02-05 | [#18280](https://github.com/sgl-project/sglang/pull/18280) | merged | [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py` |
| 2026-02-05 | [#18297](https://github.com/sgl-project/sglang/pull/18297) | merged | Deepseekv32 compatibility with transformers v5 | model wrapper, attention/backend, docs/config | `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-05 | [#18319](https://github.com/sgl-project/sglang/pull/18319) | merged | [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-07 | [#18389](https://github.com/sgl-project/sglang/pull/18389) | merged | Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4 | model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-02-10 | [#18526](https://github.com/sgl-project/sglang/pull/18526) | merged | [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr… | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py` |
| 2026-02-10 | [#18542](https://github.com/sgl-project/sglang/pull/18542) | open | fix: fixed aux hidden state index out of range when using eagle3 with nsa cp | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-10 | [#18553](https://github.com/sgl-project/sglang/pull/18553) | merged | Fix Bug on dsv3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py` |
| 2026-02-11 | [#18613](https://github.com/sgl-project/sglang/pull/18613) | merged | [V3.2] Change default CP token split method to `--round-robin-split` | docs/config | `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` |
| 2026-02-16 | [#18876](https://github.com/sgl-project/sglang/pull/18876) | merged | Add DeepSeek3.2 and GlmMoeDsa into moe tune | MoE/router, kernel, tests/benchmarks | `benchmark/kernels/fused_moe_triton/common_utils.py` |
| 2026-02-17 | [#18931](https://github.com/sgl-project/sglang/pull/18931) | merged | Fix NSA FP8 KV cache path for both-trtllm MHA one-shot | model wrapper, attention/backend | `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` |
| 2026-02-18 | [#18978](https://github.com/sgl-project/sglang/pull/18978) | merged | [AMD] Fix mi35x dsv32 mtp nightly | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-19 | [#19016](https://github.com/sgl-project/sglang/pull/19016) | merged | [FIX] NSA backend page_table overflow in speculative decoding target_verify | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-02-20 | [#19041](https://github.com/sgl-project/sglang/pull/19041) | merged | [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj` | attention/backend, kernel, tests/benchmarks | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` |
| 2026-02-20 | [#19062](https://github.com/sgl-project/sglang/pull/19062) | merged | [DSv32] Fix MTP and CP compatibility | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-21 | [#19122](https://github.com/sgl-project/sglang/pull/19122) | merged | [3/n] deepseek_v2.py Refactor: Migrate MLA forward method in deepseek_v2.py | model wrapper, attention/backend, tests/benchmarks | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` |
| 2026-02-22 | [#19148](https://github.com/sgl-project/sglang/pull/19148) | merged | [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache | attention/backend, kernel, scheduler/runtime | `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-02-25 | [#19319](https://github.com/sgl-project/sglang/pull/19319) | merged | [deepseekv3.2] fix get_k_and_s_triton kernel for 128K seqlen case bug | attention/backend, kernel, scheduler/runtime, tests/benchmarks | `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py` |
| 2026-02-25 | [#19367](https://github.com/sgl-project/sglang/pull/19367) | merged | Fix NSA CP positions mismatch in eagle NextN model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-02-26 | [#19428](https://github.com/sgl-project/sglang/pull/19428) | merged | [Feature] add feature mla_ag_after_qlora for dsv3.2 | model wrapper, attention/backend | `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2026-02-28 | [#19536](https://github.com/sgl-project/sglang/pull/19536) | merged | [Perf] Optimize NSA backend metadata under MTP | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` |
| 2026-03-05 | [#19945](https://github.com/sgl-project/sglang/pull/19945) | merged | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | attention/backend, kernel | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` |
| 2026-03-05 | [#19987](https://github.com/sgl-project/sglang/pull/19987) | closed | [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-07 | [#20086](https://github.com/sgl-project/sglang/pull/20086) | merged | [V32/GLM5] Change default setting of V32 nvfp4 on TP4 | misc | `python/sglang/srt/server_args.py` |
| 2026-03-11 | [#20326](https://github.com/sgl-project/sglang/pull/20326) | merged | [Doc] Add DSA/NSA attention backend to support matrix | attention/backend, docs/config | `docs/advanced_features/attention_backend.md` |
| 2026-03-11 | [#20343](https://github.com/sgl-project/sglang/pull/20343) | merged | HiSparse for Sparse Attention | attention/backend, kernel, multimodal/processor, scheduler/runtime | `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py` |
| 2026-03-12 | [#20438](https://github.com/sgl-project/sglang/pull/20438) | merged | [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2 | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-13 | [#20492](https://github.com/sgl-project/sglang/pull/20492) | merged | [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode | model wrapper | `python/sglang/srt/models/deepseek_v2.py` |
| 2026-03-13 | [#20534](https://github.com/sgl-project/sglang/pull/20534) | open | Transfer FP8 K/K_scale for CP indexer prefill gather | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-03-15 | [#20606](https://github.com/sgl-project/sglang/pull/20606) | merged | FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache | attention/backend | `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-03-18 | [#20840](https://github.com/sgl-project/sglang/pull/20840) | merged | [AMD] Fix dpsk-v32 accuracy issue on mi355 | quantization | `python/sglang/srt/layers/quantization/fp8_utils.py` |
| 2026-03-20 | [#20984](https://github.com/sgl-project/sglang/pull/20984) | merged | Fix DeepSeek V32 FP4 test | quantization, tests/benchmarks | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` |
| 2026-03-20 | [#21003](https://github.com/sgl-project/sglang/pull/21003) | merged | Revert "Fix DeepSeek V32 FP4 test" | quantization, tests/benchmarks | `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` |
| 2026-03-23 | [#21192](https://github.com/sgl-project/sglang/pull/21192) | merged | Fix CP in-seq-split method for DeepSeek V32 and update related tests | model wrapper, tests/benchmarks | `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py` |
| 2026-03-24 | [#21249](https://github.com/sgl-project/sglang/pull/21249) | merged | Support allreduce fusion with cp | attention/backend, scheduler/runtime | `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py` |
| 2026-03-24 | [#21259](https://github.com/sgl-project/sglang/pull/21259) | merged | [HiCache & HybridModel] mooncake backend support DSA & mamba model | scheduler/runtime, tests/benchmarks | `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` |
| 2026-03-24 | [#21337](https://github.com/sgl-project/sglang/pull/21337) | merged | Workaround of DSA performance drop on B200 + DP | misc | `python/sglang/srt/server_args.py` |
| 2026-03-25 | [#21405](https://github.com/sgl-project/sglang/pull/21405) | merged | Enable IndexCache for DeepSeek V3.2 | model wrapper, attention/backend, scheduler/runtime, tests/benchmarks | `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` |
| 2026-03-26 | [#21468](https://github.com/sgl-project/sglang/pull/21468) | merged | [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation | docs/config | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-03-27 | [#21511](https://github.com/sgl-project/sglang/pull/21511) | merged | [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend | model wrapper, attention/backend, kernel, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py` |
| 2026-03-28 | [#21585](https://github.com/sgl-project/sglang/pull/21585) | merged | [CI] Move v32 cp test to deepep running suite | tests/benchmarks | `test/registered/cp/test_deepseek_v32_cp_single_node.py` |
| 2026-03-28 | [#21599](https://github.com/sgl-project/sglang/pull/21599) | merged | [SPEC][1/N] feat: add adaptive speculative_num_steps for EAGLE topk=1 | kernel, scheduler/runtime, tests/benchmarks, docs/config | `benchmark/bench_adaptive_speculative.py`, `test/registered/unit/spec/test_adaptive_spec_params.py`, `test/registered/spec/eagle/test_adaptive_speculative.py` |
| 2026-03-29 | [#21623](https://github.com/sgl-project/sglang/pull/21623) | open | [Test] Add unit tests for encoding_dsv32.py | tests/benchmarks | `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` |
| 2026-03-31 | [#21783](https://github.com/sgl-project/sglang/pull/21783) | merged | [DSA] Support trtllm sparse mla kernel for prefill batches | attention/backend, tests/benchmarks | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py` |
| 2026-04-02 | [#21914](https://github.com/sgl-project/sglang/pull/21914) | merged | [DSA] Set trtllm kernels as default for Blackwell | misc | `python/sglang/srt/server_args.py` |
| 2026-04-02 | [#21932](https://github.com/sgl-project/sglang/pull/21932) | merged | [HiSparse] Optimize the scheduling of decode backup. | scheduler/runtime | `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-04-03 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | model wrapper, attention/backend, MoE/router, tests/benchmarks | `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2026-04-03 | [#22065](https://github.com/sgl-project/sglang/pull/22065) | merged | [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models. | misc | `python/sglang/srt/server_args.py` |
| 2026-04-05 | [#22128](https://github.com/sgl-project/sglang/pull/22128) | merged | Allow piecewise CUDA graph with speculative decoding | kernel, scheduler/runtime, tests/benchmarks | `test/registered/piecewise_cuda_graph/test_pcg_with_speculative_decoding.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-04-06 | [#22179](https://github.com/sgl-project/sglang/pull/22179) | merged | [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation | docs/config | `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22232](https://github.com/sgl-project/sglang/pull/22232) | merged | Reduce unnecessary kernels and copies in the NSA indexer | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-07 | [#22238](https://github.com/sgl-project/sglang/pull/22238) | merged | [HiSparse]: Add readme docs for HiSparse Feature | docs/config | `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md` |
| 2026-04-07 | [#22258](https://github.com/sgl-project/sglang/pull/22258) | merged | [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization | attention/backend | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-08 | [#22372](https://github.com/sgl-project/sglang/pull/22372) | merged | [DSA] Hopper FP8 FlashMLA KV padding | attention/backend, docs/config | `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py` |
| 2026-04-08 | [#22390](https://github.com/sgl-project/sglang/pull/22390) | merged | [DSA] Enable all reduce fusion for DSA models | misc | `python/sglang/srt/server_args.py` |
| 2026-04-09 | [#22424](https://github.com/sgl-project/sglang/pull/22424) | merged | [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches | attention/backend | `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-09 | [#22425](https://github.com/sgl-project/sglang/pull/22425) | merged | [HiSparse]: Add HiSpares-DSA Model's nightly CI | model wrapper, tests/benchmarks | `test/registered/8-gpu-models/test_dsa_models_hisparse.py` |
| 2026-04-09 | [#22430](https://github.com/sgl-project/sglang/pull/22430) | merged | [Fix] Fix several bugs on DSA models | attention/backend | `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` |
| 2026-04-14 | [#22792](https://github.com/sgl-project/sglang/pull/22792) | open | nsa indexer: use aiter indexer_k_quant_and_cache | model wrapper, attention/backend, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` |
| 2026-04-15 | [#22850](https://github.com/sgl-project/sglang/pull/22850) | merged | [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion) | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-16 | [#22914](https://github.com/sgl-project/sglang/pull/22914) | merged | [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel | model wrapper, attention/backend, scheduler/runtime | `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` |
| 2026-04-16 | [#22950](https://github.com/sgl-project/sglang/pull/22950) | closed | [fix] Parser-gated two-phase cache stripping for reasoning radix caches (fixes #22373) | scheduler/runtime, tests/benchmarks, docs/config | `test/registered/unit/mem_cache/test_radix_cache_thinking.py`, `test/registered/unit/mem_cache/test_radix_cache_thinking_gated.py`, `python/sglang/srt/mem_cache/mamba_radix_cache.py` |
| 2026-04-20 | [#23219](https://github.com/sgl-project/sglang/pull/23219) | merged | [AMD] Enable MTP for GLM-5-mxfp4 model | model wrapper | `python/sglang/srt/models/deepseek_nextn.py` |
| 2026-04-20 | [#23268](https://github.com/sgl-project/sglang/pull/23268) | open | 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache | attention/backend | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` |
| 2026-04-21 | [#23315](https://github.com/sgl-project/sglang/pull/23315) | merged | Opt-in strip of thinking tokens from radix cache | scheduler/runtime, tests/benchmarks | `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`, `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/server_args.py` |
| 2026-04-21 | [#23336](https://github.com/sgl-project/sglang/pull/23336) | open | [SPEC V2][2/N] feat: adaptive spec support spec v2 | multimodal/processor, scheduler/runtime | `python/sglang/srt/speculative/eagle_worker_v2.py`, `python/sglang/srt/speculative/eagle_info_v2.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` |
| 2026-04-21 | [#23351](https://github.com/sgl-project/sglang/pull/23351) | open | Support piecewise CUDA graph with NSA | attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py` |

### 逐 PR 代码 diff 阅读记录

### PR #10912 - [PD] Add PD support for hybrid model (Qwen3-Next, DeepSeek V3.2 Exp)

- 链接：https://github.com/sgl-project/sglang/pull/10912
- 状态/时间：`merged`，created 2025-09-25, merged 2025-10-16；作者 `ShangmingCai`。
- 代码 diff 已读范围：`13` 个文件，`+727/-186`；代码面：attention/backend, scheduler/runtime, tests/benchmarks；关键词：kv, cache, spec, attention, config, scheduler, test, cuda, fp8, lora。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +248/-137 (385 lines); hunk: def __init__(; def fork_from(self, src_index: torch.Tensor) -> Optional[torch.Tensor]:; 符号: __init__, for, get_speculative_mamba2_params_all_layers, fork_from
  - `python/sglang/srt/disaggregation/mooncake/conn.py` modified +148/-17 (165 lines); hunk: class TransferKVChunk:; class TransferInfo:; 符号: TransferKVChunk:, TransferInfo:, from_zmq, from_zmq
  - `python/sglang/srt/disaggregation/decode.py` modified +113/-8 (121 lines); hunk: from collections import deque; ); 符号: clear, HybridMambaDecodeReqToTokenPool, __init__, clear
  - `test/srt/test_disaggregation_hybrid_attention.py` added +83/-0 (83 lines); hunk: +import os; 符号: TestDisaggregationHybridAttentionMamba, setUpClass, start_prefill, start_decode
  - `python/sglang/srt/disaggregation/prefill.py` modified +71/-1 (72 lines); hunk: RequestStage,; def _init_kv_manager(self) -> BaseKVManager:; 符号: _init_kv_manager, send_kv_chunk
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py`；patch 关键词为 kv, cache, spec, attention, config, scheduler。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/disaggregation/mooncake/conn.py`, `python/sglang/srt/disaggregation/decode.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11061 - Support DeepSeek V3.2 Exp

- 链接：https://github.com/sgl-project/sglang/pull/11061
- 状态/时间：`merged`，created 2025-09-29, merged 2025-10-06；作者 `fzyzcjy`。
- 代码 diff 已读范围：`29` 个文件，`+4542/-141`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：quant, attention, cuda, fp8, cache, kv, topk, config, mla, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` added +887/-0 (887 lines); hunk: +from __future__ import annotations; 符号: NSAFlashMLAMetadata:, slice, copy_, NSAMetadata:
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` added +785/-0 (785 lines); hunk: +from typing import Optional, Tuple; 符号: fast_log2_ceil, fast_pow2, fast_round_scale, act_quant_kernel
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` added +761/-0 (761 lines); hunk: +from __future__ import annotations; 符号: BaseIndexerMetadata, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` added +354/-0 (354 lines); hunk: +from typing import TYPE_CHECKING; 符号: GetK:, execute, slow, torch_fast
  - `python/sglang/srt/models/deepseek_v2.py` modified +329/-17 (346 lines); hunk: # Adapted from:; import torch; 符号: AttnForwardMethod, handle_attention_ascend, _get_sum_extend_prefix_lens, _is_extend_without_speculative
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 quant, attention, cuda, fp8, cache, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11063 - Add DeepSeek-V3.2 Tool Call Template

- 链接：https://github.com/sgl-project/sglang/pull/11063
- 状态/时间：`merged`，created 2025-09-29, merged 2025-10-05；作者 `Xu-Wenqing`。
- 代码 diff 已读范围：`1` 个文件，`+100/-0`；代码面：docs/config；关键词：kv。
- 代码 diff 细节：
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` added +100/-0 (100 lines); hunk: +{% if not add_generation_prompt is defined %}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `examples/chat_template/tool_chat_template_deepseekv32.jinja`；patch 关键词为 kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `examples/chat_template/tool_chat_template_deepseekv32.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11109 - [Draft] Support MTP for DeepSeek-V3.2

- 链接：https://github.com/sgl-project/sglang/pull/11109
- 状态/时间：`closed`，created 2025-09-30, closed 2025-10-17；作者 `Fridge003`。
- 代码 diff 已读范围：`4` 个文件，`+180/-25`；代码面：attention/backend, scheduler/runtime, docs/config；关键词：topk, attention, kv, eagle, flash, mla, spec, cache, config, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +146/-21 (167 lines); hunk: def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; def __init__(self, model_runner: ModelRunner):; 符号: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/speculative/eagle_worker.py` modified +18/-0 (18 lines); hunk: def _create_decode_backend(self):; def _create_draft_extend_backend(self):; 符号: _create_decode_backend, _create_draft_extend_backend, _create_flashmla_decode_backend, _create_nsa_decode_backend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +11/-3 (14 lines); hunk: def _get_topk_ragged(; def _get_topk_ragged(; 符号: _get_topk_ragged, _get_topk_ragged, _forward
  - `python/sglang/srt/configs/model_config.py` modified +5/-1 (6 lines); hunk: def is_deepseek_nsa(config: PretrainedConfig) -> bool:; 符号: is_deepseek_nsa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 topk, attention, kv, eagle, flash, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/speculative/eagle_worker.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11191 - [Feature] Support Sparse Attention and KV cache scheduling between CPU and GPU for GQA/DSA.

- 链接：https://github.com/sgl-project/sglang/pull/11191
- 状态/时间：`open`，created 2025-10-03；作者 `yukavio`。
- 代码 diff 已读范围：`52` 个文件，`+18474/-70`；代码面：attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, flash, cuda, config, kv, cache, scheduler, mla, doc, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py` added +2560/-0 (2560 lines)
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py` added +1547/-0 (1547 lines); hunk: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; 符号: FlashAttentionBackwardSm80:, __init__, can_implement, _check_type
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` added +1402/-0 (1402 lines); hunk: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; 符号: FlashAttentionForwardSm90, __init__, _get_smem_layout_atom, _get_tiled_mma
  - `python/sglang/srt/sparse_attention/kernels/attention/interface.py` added +1266/-0 (1266 lines); hunk: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; 符号: maybe_contiguous, _flash_attn_fwd, _flash_attn_bwd, FlashAttnFunc
  - `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd.py` added +1259/-0 (1259 lines); hunk: +# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.; 符号: FlashAttentionForwardBase:, __init__, can_implement, _check_type
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py`；patch 关键词为 attention, flash, cuda, config, kv, cache。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm100.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_bwd.py`, `python/sglang/srt/sparse_attention/kernels/attention/flash_fwd_sm90.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11194 - [Feature] Add a fast-topk to sgl-kernel for DeepSeek v3.2

- 链接：https://github.com/sgl-project/sglang/pull/11194
- 状态/时间：`merged`，created 2025-10-03, merged 2025-10-05；作者 `DarkSharpness`。
- 代码 diff 已读范围：`7` 个文件，`+588/-1`；代码面：MoE/router, kernel, tests/benchmarks；关键词：topk, cuda, mla, spec, awq, test。
- 代码 diff 细节：
  - `sgl-kernel/csrc/elementwise/topk.cu` added +422/-0 (422 lines); hunk: +/**; 符号: int, int, size_t, FastTopKParams
  - `sgl-kernel/tests/test_topk.py` added +120/-0 (120 lines); hunk: +import pytest; 符号: _ref_torch_impl, _ref_torch_transform_decode_impl, assert_equal, test_topk_kernel
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +29/-0 (29 lines); hunk: def fast_topk(values, topk, dim):; 符号: fast_topk, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +8/-0 (8 lines); hunk: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/csrc/common_extension.cc` modified +7/-0 (7 lines); hunk: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`；patch 关键词为 topk, cuda, mla, spec, awq, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11308 - [CI] Add Basic Test for DeepSeek V3.2

- 链接：https://github.com/sgl-project/sglang/pull/11308
- 状态/时间：`merged`，created 2025-10-07, merged 2025-10-13；作者 `Fridge003`。
- 代码 diff 已读范围：`4` 个文件，`+137/-4`；代码面：tests/benchmarks；关键词：test, deepep, attention, cuda, flash, kv, mla, quant。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_basic.py` added +78/-0 (78 lines); hunk: +import unittest; 符号: TestDeepseekV3Basic, setUpClass, tearDownClass, test_a_gsm8k
  - `.github/workflows/pr-test.yml` modified +30/-3 (33 lines); hunk: jobs:; jobs:
  - `scripts/ci/ci_install_dependency.sh` modified +26/-1 (27 lines); hunk: set -euxo pipefail; if [ "$IS_BLACKWELL" != "1" ]; then
  - `test/srt/run_suite.py` modified +3/-0 (3 lines); hunk: class TestFile:; 符号: TestFile:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh`；patch 关键词为 test, deepep, attention, cuda, flash, kv。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_basic.py`, `.github/workflows/pr-test.yml`, `scripts/ci/ci_install_dependency.sh` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11309 - [DeepSeek-V3.2] Include indexer kv cache when estimating kv cache size

- 链接：https://github.com/sgl-project/sglang/pull/11309
- 状态/时间：`merged`，created 2025-10-07, merged 2025-10-09；作者 `trevor-m`。
- 代码 diff 已读范围：`3` 个文件，`+25/-7`；代码面：scheduler/runtime；关键词：cache, kv, quant, attention, config, fp8, mla, spec。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-4 (18 lines); hunk: def __init__(; def load_cpu_copy(self, kv_cache_cpu, indices):; 符号: __init__, get_kv_size_bytes, load_cpu_copy, NSATokenToKVPool
  - `python/sglang/srt/model_executor/model_runner.py` modified +11/-0 (11 lines); hunk: def profile_max_num_token(self, total_gpu_memory: int):; 符号: profile_max_num_token
  - `python/sglang/srt/server_args.py` modified +0/-3 (3 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, kv, quant, attention, config, fp8。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11450 - [DPSKv3.2] Rewrite nsa tilelang act_quant kernel to triton

- 链接：https://github.com/sgl-project/sglang/pull/11450
- 状态/时间：`merged`，created 2025-10-11, merged 2025-10-11；作者 `byjiang1996`。
- 代码 diff 已读范围：`3` 个文件，`+420/-1`；代码面：attention/backend, quantization, kernel, tests/benchmarks；关键词：attention, quant, triton, fp8, benchmark, cuda, kv, test, vision。
- 代码 diff 细节：
  - `test/srt/layers/attention/nsa/test_act_quant_triton.py` added +281/-0 (281 lines); hunk: +"""; 符号: benchmark_kernel, check_accuracy, test_act_quant_comprehensive_benchmark
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` added +136/-0 (136 lines); hunk: +from typing import Optional, Tuple; 符号: _act_quant_kernel, act_quant
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-1 (4 lines); hunk: def _forward(; 符号: _forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, quant, triton, fp8, benchmark, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/layers/attention/nsa/test_act_quant_triton.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11510 - [Bugfix] Fix Qwen3/DSV3/DSV3.2 model support

- 链接：https://github.com/sgl-project/sglang/pull/11510
- 状态/时间：`merged`，created 2025-10-12, merged 2025-10-16；作者 `iforgetmyname`。
- 代码 diff 已读范围：`12` 个文件，`+102/-33`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：cache, test, cuda, attention, config, vision, deepep, doc, mla, quant。
- 代码 diff 细节：
  - `.github/workflows/pr-test-npu.yml` modified +33/-13 (46 lines); hunk: jobs:; jobs:
  - `python/sglang/srt/server_args.py` modified +20/-0 (20 lines); hunk: def _handle_gpu_memory_settings(self, gpu_mem):; def _handle_gpu_memory_settings(self, gpu_mem):; 符号: _handle_gpu_memory_settings, _handle_gpu_memory_settings
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +17/-0 (17 lines); hunk: def forward_extend(; def forward_extend(; 符号: forward_extend, forward_extend, forward_decode_graph
  - `scripts/ci/npu_ci_install_dependency.sh` modified +13/-3 (16 lines); hunk: set -euo pipefail; TORCHVISION_VERSION=0.21.0
  - `docker/Dockerfile.npu` modified +10/-2 (12 lines); hunk: ARG PYTHON_VERSION=py3.11; RUN git clone --depth 1 https://github.com/vllm-project/vllm.git --branch $VLLM_
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py`；patch 关键词为 cache, test, cuda, attention, config, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `.github/workflows/pr-test-npu.yml`, `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/ascend_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11557 - Fix DeepSeek-v3.2 default config (ValueError: not enough values to unpack (expected 4, got 3))

- 链接：https://github.com/sgl-project/sglang/pull/11557
- 状态/时间：`merged`，created 2025-10-13, merged 2025-10-13；作者 `trevor-m`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：misc；关键词：attention, config, cuda, kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 attention, config, cuda, kv, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11565 - [DSv32] Use torch.compile for _get_logits_head_gate

- 链接：https://github.com/sgl-project/sglang/pull/11565
- 状态/时间：`merged`，created 2025-10-13, merged 2025-10-14；作者 `trevor-m`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：attention/backend；关键词：attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-0 (1 lines); hunk: def _forward_fake(; 符号: _forward_fake, _get_logits_head_gate
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11596 - [Spec Decoding] Support MTP for dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/11596
- 状态/时间：`closed`，created 2025-10-14, closed 2025-10-15；作者 `Paiiiiiiiiiiiiii`。
- 代码 diff 已读范围：`8` 个文件，`+515/-534`；代码面：attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, kv, cuda, eagle, spec, topk, attention, config, mla, doc。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +396/-65 (461 lines); hunk: from sglang.srt.model_executor.model_runner import ModelRunner; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; 符号: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `.github/workflows/pr-test-amd.yml` removed +0/-352 (352 lines); hunk: -name: PR Test (AMD)
  - `.github/workflows/release-docker-dev.yml` removed +0/-108 (108 lines); hunk: -name: Build and Push Development Docker Images
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +82/-8 (90 lines); hunk: except ImportError as e:; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; 符号: _get_topk_paged, _get_verify_topk_paged, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/speculative/draft_utils.py` modified +16/-0 (16 lines); hunk: def init_forward_metadata(*args, **kwargs):; def create_draft_extend_backend(self):; 符号: init_forward_metadata, create_draft_extend_backend, create_draft_extend_backend, _create_nsa_decode_backend
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml`；patch 关键词为 cache, kv, cuda, eagle, spec, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `.github/workflows/pr-test-amd.yml`, `.github/workflows/release-docker-dev.yml` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11652 - [Spec Decoding] Support MTP for dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/11652
- 状态/时间：`merged`，created 2025-10-15, merged 2025-10-19；作者 `Paiiiiiiiiiiiiii`。
- 代码 diff 已读范围：`6` 个文件，`+445/-79`；代码面：attention/backend, kernel, scheduler/runtime, docs/config；关键词：kv, cache, cuda, eagle, spec, topk, attention, config, flash, mla。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +385/-68 (453 lines); hunk: from sglang.srt.model_executor.model_runner import ModelRunner; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; 符号: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +23/-10 (33 lines); hunk: def _get_topk_paged(; def _get_topk_ragged(; 符号: _get_topk_paged, _get_topk_ragged, _get_topk_ragged, forward_indexer
  - `python/sglang/srt/speculative/draft_utils.py` modified +16/-0 (16 lines); hunk: def create_decode_backend(self):; def create_draft_extend_backend(self):; 符号: create_decode_backend, create_draft_extend_backend, create_draft_extend_backend, _create_nsa_decode_backend
  - `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` modified +8/-0 (8 lines); hunk: def __init__(self, eagle_worker: EAGLEWorker):; def __init__(self, eagle_worker: EAGLEWorker):; 符号: __init__, __init__, capture_one_batch_size, capture_one_batch_size
  - `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` modified +8/-0 (8 lines); hunk: def __init__(self, eagle_worker: EAGLEWorker):; def capture_one_batch_size(self, bs: int, forward: Callable):; 符号: __init__, capture_one_batch_size, capture_one_batch_size, capture_one_batch_size
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py`；patch 关键词为 kv, cache, cuda, eagle, spec, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/speculative/draft_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11682 - Cleaning indexer for DeepSeek V3.2

- 链接：https://github.com/sgl-project/sglang/pull/11682
- 状态/时间：`merged`，created 2025-10-15, merged 2025-10-17；作者 `Fridge003`。
- 代码 diff 已读范围：`2` 个文件，`+3/-66`；代码面：attention/backend；关键词：attention, topk, cuda, fp8, kv, lora, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +3/-65 (68 lines); hunk: except ImportError as e:; def __init__(; 符号: __init__, _forward_fake, _get_logits_head_gate, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-1 (1 lines); hunk: # temp NSA debugging environ
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py`；patch 关键词为 attention, topk, cuda, fp8, kv, lora。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11761 - (beta)support context parallel with deepseekv3.2-DSA

- 链接：https://github.com/sgl-project/sglang/pull/11761
- 状态/时间：`closed`，created 2025-10-17, closed 2025-10-23；作者 `lixiaolx`。
- 代码 diff 已读范围：`0` 个文件，`+0/-0`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - GitHub 未返回文件级 patch。
- 支持/优化点判断：该 PR 的实际 diff 主要落在 未返回 patch 文件；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 未返回 patch 文件 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11815 - [DeepseekV32] Add fast_topk_transform_ragged_fused kernel

- 链接：https://github.com/sgl-project/sglang/pull/11815
- 状态/时间：`merged`，created 2025-10-19, merged 2025-10-20；作者 `hlu1`。
- 代码 diff 已读范围：`6` 个文件，`+201/-20`；代码面：MoE/router, kernel, tests/benchmarks；关键词：topk, cuda, kv, spec, mla, quant, test。
- 代码 diff 细节：
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +81/-8 (89 lines); hunk: __device__ void naive_topk_transform(; __global__ __launch_bounds__(kThreadsPerBlock) // prefill; 符号: __launch_bounds__, __launch_bounds__
  - `sgl-kernel/tests/test_topk.py` modified +75/-4 (79 lines); hunk: +from typing import Optional; def _ref_torch_transform_decode_impl(; 符号: _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl, assert_equal
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +24/-1 (25 lines); hunk: def fast_topk_transform_fused(; 符号: fast_topk_transform_fused, fast_topk_transform_ragged_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +11/-6 (17 lines); hunk: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/python/sgl_kernel/__init__.py` modified +6/-1 (7 lines); hunk: def _find_cuda_home():; 符号: _find_cuda_home
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py`；patch 关键词为 topk, cuda, kv, spec, mla, quant。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/tests/test_topk.py`, `sgl-kernel/python/sgl_kernel/top_k.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11835 - [CI] Add CI test for DeepSeek V3.2 MTP

- 链接：https://github.com/sgl-project/sglang/pull/11835
- 状态/时间：`merged`，created 2025-10-19, merged 2025-10-20；作者 `Fridge003`。
- 代码 diff 已读范围：`4` 个文件，`+112/-3`；代码面：tests/benchmarks；关键词：test, eagle, kv, spec, attention, awq, cache, quant, topk。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_mtp.py` added +105/-0 (105 lines); hunk: +import unittest; 符号: TestDeepseekV32MTP, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/test_deepseek_v32_basic.py` modified +3/-3 (6 lines); hunk: DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_a_gsm8k(; 符号: TestDeepseekV3Basic, TestDeepseekV32Basic, setUpClass, test_a_gsm8k
  - `python/sglang/srt/server_args.py` modified +3/-0 (3 lines); hunk: def _handle_speculative_decoding(self):; 符号: _handle_speculative_decoding
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: class TestFile:; 符号: TestFile:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py`；patch 关键词为 test, eagle, kv, spec, attention, awq。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11877 - [Doc] Add documentation for DeepSeek V3.2

- 链接：https://github.com/sgl-project/sglang/pull/11877
- 状态/时间：`merged`，created 2025-10-20, merged 2025-10-25；作者 `Fridge003`。
- 代码 diff 已读范围：`4` 个文件，`+723/-3`；代码面：docs/config；关键词：doc, spec, attention, config, cache, cuda, deepep, eagle, flash, kv。
- 代码 diff 细节：
  - `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md` added +570/-0 (570 lines); hunk: +# DeepSeekV32-Exp RBG Based PD Deploy
  - `docs/basic_usage/deepseek_v32.md` added +150/-0 (150 lines); hunk: +# DeepSeek V3.2 Usage
  - `docs/advanced_features/separate_reasoning.ipynb` modified +2/-2 (4 lines); hunk: "\| Model \| Reasoning tags \| Parser \| Notes \|
",; "- Both are handled by the same `deepseek-r1` parser
",
  - `docs/basic_usage/deepseek.md` modified +1/-1 (2 lines); hunk: python3 -m sglang.launch_server \
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb`；patch 关键词为 doc, spec, attention, config, cache, cuda。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/references/multi_node_deployment/rbg_pd/deepseekv32_pd.md`, `docs/basic_usage/deepseek_v32.md`, `docs/advanced_features/separate_reasoning.ipynb` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11892 - DeepSeek-V3.2: Add Adaptive MHA Attention Pathway for Short-Sequence Prefill

- 链接：https://github.com/sgl-project/sglang/pull/11892
- 状态/时间：`merged`，created 2025-10-21, merged 2025-11-06；作者 `YAMY1234`。
- 代码 diff 已读范围：`3` 个文件，`+188/-4`；代码面：model wrapper, attention/backend；关键词：attention, cache, kv, mla, cuda, lora, quant, spec, topk, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +84/-0 (84 lines); hunk: def _get_q_k_bf16(; def _get_topk_ragged(; 符号: _get_q_k_bf16, _get_k_bf16, _get_topk_paged, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +61/-2 (63 lines); hunk: "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."; def forward_extend(; 符号: forward_extend, _forward_flashmla_kv, _forward_standard_mha, _forward_tilelang
  - `python/sglang/srt/models/deepseek_v2.py` modified +43/-2 (45 lines); hunk: def handle_attention_aiter(attn, forward_batch):; def forward_normal_prepare(; 符号: handle_attention_aiter, handle_attention_nsa, forward_normal_prepare
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, kv, mla, cuda, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11936 - [Test] Add dsv3.2 nsa backend testing

- 链接：https://github.com/sgl-project/sglang/pull/11936
- 状态/时间：`merged`，created 2025-10-21, merged 2025-10-26；作者 `Johnsonms`。
- 代码 diff 已读范围：`2` 个文件，`+125/-0`；代码面：tests/benchmarks；关键词：test, attention, awq, flash, kv, mla, quant。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_nsabackend.py` added +124/-0 (124 lines); hunk: +import unittest; 符号: TestDeepseekV32NasBackend_flashmla, setUpClass, tearDownClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: class TestFile:; 符号: TestFile:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py`；patch 关键词为 test, attention, awq, flash, kv, mla。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_nsabackend.py`, `test/srt/run_suite.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12017 - (beta)support context parallel with deepseekv3.2-DSA

- 链接：https://github.com/sgl-project/sglang/pull/12017
- 状态/时间：`closed`，created 2025-10-23, closed 2025-10-24；作者 `lixiaolx`。
- 代码 diff 已读范围：`11` 个文件，`+595/-81`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, config, cuda, kv, quant, cache, lora, moe, topk, expert。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +133/-50 (183 lines); hunk: get_attention_tp_rank,; ParallelLMHead,; 符号: forward, __init__, forward, forward
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +174/-5 (179 lines); hunk: except ImportError as e:; DUAL_STREAM_TOKEN_THRESHOLD = 1024 if is_cuda() else 0; 符号: BaseIndexerMetadata, __init__, _get_q_k_bf16, _get_q_k_bf16
  - `python/sglang/srt/utils/common.py` modified +121/-1 (122 lines); hunk: from sglang.srt.environ import envs; def require_mlp_tp_gather(server_args):; 符号: require_mlp_tp_gather, require_attn_tp_gather, decorator, calculate_cp_seq_idx
  - `python/sglang/srt/models/deepseek_nextn.py` modified +50/-9 (59 lines); hunk: """Inference-only DeepSeek NextN Speculative Decoding."""; enable_nextn_moe_bf16_cast_to_fp8,; 符号: __init__, forward, forward, __init__
  - `python/sglang/srt/layers/communicator.py` modified +25/-9 (34 lines); hunk: is_sm100_supported,; def _scattered_to_tp_attn_full(; 符号: _scattered_to_tp_attn_full, _scatter_hidden_states_and_residual, _scatter_hidden_states, _gather
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py`；patch 关键词为 attention, config, cuda, kv, quant, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/utils/common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12044 - Enable mixed type LayerNorm kernel for NSA indexer

- 链接：https://github.com/sgl-project/sglang/pull/12044
- 状态/时间：`merged`，created 2025-10-24, merged 2025-11-04；作者 `akhilg-nv`。
- 代码 diff 已读范围：`3` 个文件，`+166/-25`；代码面：attention/backend, tests/benchmarks；关键词：cuda, attention, flash, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/layernorm.py` modified +91/-3 (94 lines); hunk: import torch; _is_cpu_amx_available = cpu_has_amx_support(); 符号: forward_with_allreduce_fusion, LayerNorm, __init__, forward_cuda
  - `python/sglang/test/test_layernorm.py` modified +73/-1 (74 lines); hunk: import torch; def test_gemma_rms_norm(self):; 符号: test_gemma_rms_norm, TestLayerNorm, setUpClass, _run_layer_norm_test
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-21 (23 lines); hunk: from typing import TYPE_CHECKING, Any, Dict, Optional; def rotate_activation(x: torch.Tensor) -> torch.Tensor:; 符号: rotate_activation, V32LayerNorm, __init__, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 cuda, attention, flash, test。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/layernorm.py`, `python/sglang/test/test_layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12052 - Fix Illegal Instruction/IMA errors when using DP attention with DeepSeek-V3.2 models

- 链接：https://github.com/sgl-project/sglang/pull/12052
- 状态/时间：`closed`，created 2025-10-24, closed 2025-10-25；作者 `YAMY1234`。
- 代码 diff 已读范围：`1` 个文件，`+18/-1`；代码面：attention/backend；关键词：attention, scheduler。
- 代码 diff 细节：
  - `python/sglang/srt/layers/dp_attention.py` modified +18/-1 (19 lines); hunk: def _dp_gather_via_all_reduce(; 符号: _dp_gather_via_all_reduce
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/dp_attention.py`；patch 关键词为 attention, scheduler。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/dp_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12065 - (1/n)support context parallel with deepseekv3.2-DSA

- 链接：https://github.com/sgl-project/sglang/pull/12065
- 状态/时间：`merged`，created 2025-10-24, merged 2025-11-17；作者 `lixiaolx`。
- 代码 diff 已读范围：`17` 个文件，`+1247/-54`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, kv, cuda, topk, spec, cache, config, moe, quant, expert。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +305/-0 (305 lines); hunk: # temp NSA debugging environ; def print_nsa_bool_env_vars():; 符号: print_nsa_bool_env_vars, compute_nsa_seqlens, is_nsa_enable_prefill_cp, NSAContextParallelMetadata:
  - `python/sglang/srt/layers/communicator_nsa_cp.py` added +284/-0 (284 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: nsa_enable_prefill_cp, NSACPLayerCommunicator, __init__, NSACPCommunicateSimpleFn
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +221/-8 (229 lines); hunk: from __future__ import annotations; except ImportError as e:; 符号: __init__, _get_q_k_bf16, _get_q_k_bf16, _forward_cuda_k_only
  - `python/sglang/srt/models/deepseek_v2.py` modified +134/-32 (166 lines); hunk: is_mla_preprocess_enabled,; def handle_attention_nsa(attn, forward_batch):; 符号: handle_attention_nsa, __init__, forward, forward
  - `test/srt/test_deepseek_v32_cp_single_node.py` added +99/-0 (99 lines); hunk: +"""; 符号: TestDeepseekV32CP, setUpClass, tearDownClass, test_a_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, kv, cuda, topk, spec, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12094 - Fuse wk and weight_proj in Indexer for DeepSeekV3.2-FP4

- 链接：https://github.com/sgl-project/sglang/pull/12094
- 状态/时间：`merged`，created 2025-10-24, merged 2025-10-30；作者 `trevor-m`。
- 代码 diff 已读范围：`2` 个文件，`+110/-22`；代码面：model wrapper, attention/backend；关键词：attention, config, lora, quant, cache, cuda, fp4, fp8, kv, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-22 (67 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +65/-0 (65 lines); hunk: def add_forward_absorb_core_attention_backend(backend_name):; def __init__(; 符号: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, config, lora, quant, cache, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12123 - Fix DeepSeek chat templates to handle tool call arguments type checking (#11700)

- 链接：https://github.com/sgl-project/sglang/pull/12123
- 状态/时间：`merged`，created 2025-10-25, merged 2025-10-30；作者 `Kangyan-Zhou`。
- 代码 diff 已读范围：`4` 个文件，`+331/-9`；代码面：tests/benchmarks, docs/config；关键词：kv, test。
- 代码 diff 细节：
  - `test/srt/test_deepseek_chat_templates.py` added +319/-0 (319 lines); hunk: +"""; 符号: tool, TestDeepSeekChatTemplateToolCalls, setUpClass, _render_template
  - `examples/chat_template/tool_chat_template_deepseekv3.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_tool = false -%}
  - `examples/chat_template/tool_chat_template_deepseekv31.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_first = false %}
  - `examples/chat_template/tool_chat_template_deepseekv32.jinja` modified +4/-3 (7 lines); hunk: {%- set ns.is_first = false %}
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_chat_templates.py`, `examples/chat_template/tool_chat_template_deepseekv3.jinja`, `examples/chat_template/tool_chat_template_deepseekv31.jinja` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12296 - Update deepseek_v32.md

- 链接：https://github.com/sgl-project/sglang/pull/12296
- 状态/时间：`merged`，created 2025-10-28, merged 2025-10-28；作者 `hlu1`。
- 代码 diff 已读范围：`1` 个文件，`+4/-5`；代码面：docs/config；关键词：attention, benchmark, cache, config, doc, eagle, flash, fp8, kv, mla。
- 代码 diff 细节：
  - `docs/basic_usage/deepseek_v32.md` modified +4/-5 (9 lines); hunk: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/deepseek_v32.md`；patch 关键词为 attention, benchmark, cache, config, doc, eagle。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12582 - [sgl-kernel][Deepseek V3.2] Add row_starts to topk kernel

- 链接：https://github.com/sgl-project/sglang/pull/12582
- 状态/时间：`merged`，created 2025-11-04, merged 2025-11-08；作者 `hlu1`。
- 代码 diff 已读范围：`5` 个文件，`+209/-61`；代码面：MoE/router, kernel, tests/benchmarks；关键词：topk, cuda, kv, mla, spec, test。
- 代码 diff 细节：
  - `sgl-kernel/tests/test_topk.py` modified +85/-24 (109 lines); hunk: -from typing import Optional; ); 符号: _ref_torch_impl, _ref_torch_impl, _ref_torch_transform_decode_impl, _ref_torch_transform_ragged_impl
  - `sgl-kernel/csrc/elementwise/topk.cu` modified +51/-24 (75 lines); hunk: constexpr int kThreadsPerBlock = 1024;; __device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {; 符号: int, size_t, FastTopKParams, __launch_bounds__
  - `sgl-kernel/python/sgl_kernel/top_k.py` modified +61/-7 (68 lines); hunk: +from typing import Optional; def fast_topk(values, topk, dim):; 符号: fast_topk, fast_topk_v2, fast_topk_v2, fast_topk_transform_fused
  - `sgl-kernel/include/sgl_kernel_ops.h` modified +9/-3 (12 lines); hunk: void copy_to_gpu_no_ce(const at::Tensor& input, at::Tensor& output);
  - `sgl-kernel/csrc/common_extension.cc` modified +3/-3 (6 lines); hunk: TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py`；patch 关键词为 topk, cuda, kv, mla, spec, test。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/tests/test_topk.py`, `sgl-kernel/csrc/elementwise/topk.cu`, `sgl-kernel/python/sgl_kernel/top_k.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12583 - [Deepseek V3.2] Fix accuracy bug in the Indexer

- 链接：https://github.com/sgl-project/sglang/pull/12583
- 状态/时间：`merged`，created 2025-11-04, merged 2025-11-12；作者 `hlu1`。
- 代码 diff 已读范围：`6` 个文件，`+96/-17`；代码面：attention/backend, tests/benchmarks；关键词：test, attention, kv, fp8, topk, cache, cuda, spec。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_nsabackend.py` modified +52/-2 (54 lines); hunk: def test_a_gsm8k(; def test_a_gsm8k(; 符号: test_a_gsm8k, test_a_gsm8k, TestDeepseekV32NasBackend_fp8kvcache, setUpClass
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +32/-8 (40 lines); hunk: def _get_topk_ragged(; def _get_topk_ragged(; 符号: _get_topk_ragged, _get_topk_ragged, _forward_cuda_k_only
  - `test/srt/test_deepseek_v32_mtp.py` modified +4/-4 (8 lines); hunk: write_github_step_summary,; def test_a_gsm8k(; 符号: TestDeepseekV32MTP, setUpClass, test_a_gsm8k, test_bs_1_speed
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +6/-1 (7 lines); hunk: def topk_transform(; def topk_transform(; 符号: topk_transform, topk_transform, topk_transform
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunk: jobs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py`；patch 关键词为 test, attention, kv, fp8, topk, cache。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_nsabackend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/srt/test_deepseek_v32_mtp.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12645 - [Bug] Fix NSA Backend KV-Buffer Shape Mismatch in DeepSeek-V3.2

- 链接：https://github.com/sgl-project/sglang/pull/12645
- 状态/时间：`merged`，created 2025-11-04, merged 2025-11-04；作者 `Johnsonms`。
- 代码 diff 已读范围：`1` 个文件，`+3/-1`；代码面：scheduler/runtime；关键词：cache, fp8, kv, lora, mla, quant。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +3/-1 (4 lines); hunk: def load_cpu_copy(self, kv_cache_cpu, indices):; def __init__(; 符号: load_cpu_copy, NSATokenToKVPool, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 cache, fp8, kv, lora, mla, quant。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12788 - [DeepSeek-V3.2][NSA] Enable MHA Pathway for Short Sequence Prefill on B200 (SM100)

- 链接：https://github.com/sgl-project/sglang/pull/12788
- 状态/时间：`merged`，created 2025-11-06, merged 2025-11-07；作者 `YAMY1234`。
- 代码 diff 已读范围：`2` 个文件，`+53/-6`；代码面：model wrapper, attention/backend；关键词：attention, cache, kv, topk, config, cuda, flash, mla, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +46/-2 (48 lines); hunk: import torch; from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; 符号: NSAFlashMLAMetadata:, __init__, get_device_int32_arange, _forward_standard_mha
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-4 (11 lines); hunk: def handle_attention_nsa(attn, forward_batch):; 符号: handle_attention_nsa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, kv, topk, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12816 - [Deepseek V3.2] Only skip Indexer logits computation when is_extend_without_speculative

- 链接：https://github.com/sgl-project/sglang/pull/12816
- 状态/时间：`merged`，created 2025-11-07, merged 2025-11-07；作者 `hlu1`。
- 代码 diff 已读范围：`3` 个文件，`+20/-18`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：spec, attention, cache, kv, quant, topk, cuda, flash, fp8, lora。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +6/-14 (20 lines); hunk: def _get_sum_extend_prefix_lens(forward_batch):; def _handle_attention_backend(; 符号: _get_sum_extend_prefix_lens, _is_extend_without_speculative, _support_mha_one_shot, _handle_attention_backend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunk: def _forward_cuda_k_only(; def forward_cuda(; 符号: _forward_cuda_k_only, forward_cuda
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +7/-0 (7 lines); hunk: def is_cpu_graph(self):; 符号: is_cpu_graph, is_split_prefill, is_extend_without_speculative, CaptureHiddenMode
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`；patch 关键词为 spec, attention, cache, kv, quant, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12868 - [Docs][DeepseekV3.2] Update deepseekv3.2 docs for mha short seq prefill

- 链接：https://github.com/sgl-project/sglang/pull/12868
- 状态/时间：`merged`，created 2025-11-08, merged 2025-11-08；作者 `YAMY1234`。
- 代码 diff 已读范围：`1` 个文件，`+3/-2`；代码面：docs/config；关键词：attention, benchmark, cache, config, doc, eagle, flash, fp8, kv, mla。
- 代码 diff 细节：
  - `docs/basic_usage/deepseek_v32.md` modified +3/-2 (5 lines); hunk: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --ep
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/deepseek_v32.md`；patch 关键词为 attention, benchmark, cache, config, doc, eagle。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12964 - [DeepseekV3.2] Deepseek fp8 support for MHA path

- 链接：https://github.com/sgl-project/sglang/pull/12964
- 状态/时间：`merged`，created 2025-11-10, merged 2025-11-20；作者 `YAMY1234`。
- 代码 diff 已读范围：`2` 个文件，`+55/-9`；代码面：model wrapper, attention/backend；关键词：attention, cache, fp8, kv, quant, topk, lora, mla, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +46/-8 (54 lines); hunk: NPUFusedMLAPreprocess,; def handle_attention_nsa(attn, forward_batch):; 符号: handle_attention_nsa, handle_attention_nsa, forward_normal_prepare, _get_mla_kv_buffer
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-1 (10 lines); hunk: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_forward_metadata
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, fp8, kv, quant, topk。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13022 - [Deepseek V3.2] Use torch.compile to speed up torch.cat in nsa

- 链接：https://github.com/sgl-project/sglang/pull/13022
- 状态/时间：`merged`，created 2025-11-10, merged 2025-11-17；作者 `hlu1`。
- 代码 diff 已读范围：`1` 个文件，`+22/-1`；代码面：attention/backend；关键词：attention, cache, flash, kv, mla, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +22/-1 (23 lines); hunk: class TopkTransformMethod(IntEnum):; def forward_extend(; 符号: TopkTransformMethod, _compiled_cat, _cat, NSAIndexerMetadata
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, flash, kv, mla, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13459 - [Deepseek V3.2] Change indexer weights_proj to fp32

- 链接：https://github.com/sgl-project/sglang/pull/13459
- 状态/时间：`merged`，created 2025-11-17, merged 2025-11-20；作者 `hlu1`。
- 代码 diff 已读范围：`3` 个文件，`+92/-124`；代码面：model wrapper, attention/backend, docs/config；关键词：attention, config, fp8, kv, lora, quant, benchmark, cache, cuda, doc。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +26/-53 (79 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-71 (71 lines); hunk: def add_forward_absorb_core_attention_backend(backend_name):; def __init__(; 符号: add_forward_absorb_core_attention_backend, is_nsa_indexer_wk_and_weights_proj_fused, AttnForwardMethod, __init__
  - `docs/basic_usage/deepseek_v32.md` modified +66/-0 (66 lines); hunk: Latency: 25.109 s; Repeat: 8, mean: 0.797; 符号: file:, chat_template_thinking
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md`；patch 关键词为 attention, config, fp8, kv, lora, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13531 - DeepSeek V3.2 indexer RoPE fix

- 链接：https://github.com/sgl-project/sglang/pull/13531
- 状态/时间：`closed`，created 2025-11-18, closed 2025-11-18；作者 `Johnsonms`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13544 - [DeepSeekV3.2] Centralize NSA dispatch logic in NativeSparseAttnBackend

- 链接：https://github.com/sgl-project/sglang/pull/13544
- 状态/时间：`merged`，created 2025-11-18, merged 2025-11-25；作者 `YAMY1234`。
- 代码 diff 已读范围：`2` 个文件，`+74/-78`；代码面：model wrapper, attention/backend；关键词：attention, cache, fp8, kv, mla, quant, spec, topk, benchmark, config。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +69/-42 (111 lines); hunk: NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8,; def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:; 符号: compute_cu_seqlens, NativeSparseAttnBackend, __init__, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-36 (41 lines); hunk: def handle_attention_aiter(attn, forward_batch):; 符号: handle_attention_aiter, handle_attention_nsa
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, fp8, kv, mla, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13546 - [Deepseek V3.2] Optimize use of dual_stream in nsa_indexer/attention

- 链接：https://github.com/sgl-project/sglang/pull/13546
- 状态/时间：`closed`，created 2025-11-18, closed 2026-04-10；作者 `hlu1`。
- 代码 diff 已读范围：`5` 个文件，`+254/-161`；代码面：model wrapper, attention/backend；关键词：attention, cache, kv, cuda, fp8, lora, mla, quant, topk, config。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +191/-130 (321 lines); hunk: from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple; from sglang.srt.layers import deep_gemm_wrapper; 符号: rotate_activation, V32LayerNorm, __init__, _forward_compiled
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +57/-25 (82 lines); hunk: from sglang.srt.environ import envs; def __init__(; 符号: __init__, forward_decode, forward_decode, _prepare_kv_cache
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-3 (6 lines); hunk: from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer; def rebuild_cp_kv_cache(self, latent_cache, forward_batch, k_nope, k_pe):; 符号: rebuild_cp_kv_cache, forward
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunk: from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder; def forward(; 符号: forward
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +1/-1 (2 lines); hunk: def cp_attn_tp_all_gather_reorganazied_into_tensor(; 符号: cp_attn_tp_all_gather_reorganazied_into_tensor, cp_all_gather_rerange_output, cp_all_gather_rearrange_output
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, cache, kv, cuda, fp8, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13646 - [DeepSeekV3.2] Enable pure TP & Partial DP Attention

- 链接：https://github.com/sgl-project/sglang/pull/13646
- 状态/时间：`merged`，created 2025-11-20, merged 2025-11-30；作者 `YAMY1234`。
- 代码 diff 已读范围：`7` 个文件，`+286/-24`；代码面：attention/backend, tests/benchmarks, docs/config；关键词：attention, kv, flash, mla, spec, cache, config, test, cuda, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +95/-14 (109 lines); hunk: def _get_topk_paged(; def _get_topk_ragged(; 符号: _get_topk_paged, _should_chunk_mqa_logits, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +72/-7 (79 lines); hunk: def topk_transform(; def __init__(; 符号: topk_transform, __init__, forward_extend, forward_decode
  - `test/nightly/test_deepseek_v32_nsabackend.py` modified +57/-0 (57 lines); hunk: def test_a_gsm8k(; 符号: test_a_gsm8k, TestDeepseekV32NasBackend_pure_tp, setUpClass, tearDownClass
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass, setUpClass
  - `test/nightly/test_deepseek_v32_perf.py` modified +25/-0 (25 lines); hunk: def setUpClass(cls):; def setUpClass(cls):; 符号: setUpClass, setUpClass, setUpClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py`；patch 关键词为 attention, kv, flash, mla, spec, cache。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/nightly/test_deepseek_v32_nsabackend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13812 - [Performance] Optimize NSA Indexer K/S Buffer Access with Fused Triton Kernels

- 链接：https://github.com/sgl-project/sglang/pull/13812
- 状态/时间：`merged`，created 2025-11-23, merged 2025-12-03；作者 `Johnsonms`。
- 代码 diff 已读范围：`4` 个文件，`+896/-8`；代码面：attention/backend, scheduler/runtime, tests/benchmarks；关键词：fp8, triton, attention, kv, cache, cuda, quant, test, topk。
- 代码 diff 细节：
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` added +554/-0 (554 lines); hunk: +"""; 符号: MockNSATokenToKVPool:, __init__, create_test_buffer, TestGetK:
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +318/-2 (320 lines); hunk: class GetK:; def torch_fast(; 符号: GetK:, execute, slow, torch_fast
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +22/-0 (22 lines); hunk: def get_index_k_scale_continuous(; 符号: get_index_k_scale_continuous, get_index_k_scale_buffer, set_index_k_scale_buffer
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-6 (8 lines); hunk: def _get_topk_ragged(; 符号: _get_topk_ragged
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py`；patch 关键词为 fp8, triton, attention, kv, cache, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/manual/layers/attention/nsa/test_index_buf_accessor.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `python/sglang/srt/mem_cache/memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13959 - [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache

- 链接：https://github.com/sgl-project/sglang/pull/13959
- 状态/时间：`merged`，created 2025-11-26, merged 2026-01-02；作者 `xu-yfei`。
- 代码 diff 已读范围：`14` 个文件，`+603/-264`；代码面：model wrapper, attention/backend, tests/benchmarks, docs/config；关键词：attention, kv, cache, spec, topk, cuda, fp8, triton, mla, moe。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +209/-5 (214 lines); hunk: # temp NSA debugging environ; def is_nsa_enable_prefill_cp():; 符号: is_nsa_enable_prefill_cp, is_nsa_prefill_cp_in_seq_split, is_nsa_prefill_cp_round_robin_split, can_nsa_prefill_cp_round_robin_split
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +60/-133 (193 lines); hunk: import torch; LayerScatterModes,; 符号: __init__, _post_init_communicate, get_fn, _scattered_to_tp_attn_full
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +149/-20 (169 lines); hunk: from dataclasses import dataclass; NSA_ENABLE_MTP_PRECOMPUTE_METADATA,; 符号: NSAMetadata:, TopkTransformMethod, get_seqlens_expanded, get_cu_seqlens_k
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +45/-68 (113 lines); hunk: NSA_DUAL_STREAM,; def get_seqlens_expanded(self) -> torch.Tensor:; 符号: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
  - `test/manual/test_deepseek_v32_cp_single_node.py` modified +74/-0 (74 lines); hunk: def test_a_gsm8k(; 符号: test_a_gsm8k, TestDeepseekV32CPMode1, setUpClass, tearDownClass
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, kv, cache, spec, topk, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/communicator_nsa_cp.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14304 - [FIX][DS32]openai protocol: support openai message role: developer

- 链接：https://github.com/sgl-project/sglang/pull/14304
- 状态/时间：`merged`，created 2025-12-02, merged 2025-12-11；作者 `jimmy-evo`。
- 代码 diff 已读范围：`1` 个文件，`+4/-3`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/protocol.py` modified +4/-3 (7 lines); hunk: class ToolCall(BaseModel):; 符号: ToolCall, ChatCompletionMessageGenericParam, _normalize_role
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/protocol.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/protocol.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14307 - [SMG][DS32][fix] support dsv32, add role developer

- 链接：https://github.com/sgl-project/sglang/pull/14307
- 状态/时间：`merged`，created 2025-12-02, merged 2025-12-11；作者 `jimmy-evo`。
- 代码 diff 已读范围：`3` 个文件，`+36/-9`；代码面：MoE/router；关键词：router。
- 代码 diff 细节：
  - `sgl-model-gateway/src/protocols/chat.rs` modified +12/-9 (21 lines); hunk: pub enum ChatMessage {; impl GenerationRequest for ChatCompletionRequest {
  - `sgl-model-gateway/src/routers/grpc/harmony/builder.rs` modified +20/-0 (20 lines); hunk: impl HarmonyBuilder {
  - `sgl-model-gateway/src/routers/http/pd_router.rs` modified +4/-0 (4 lines); hunk: impl RouterTrait for PDRouter {
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs`；patch 关键词为 router。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `sgl-model-gateway/src/protocols/chat.rs`, `sgl-model-gateway/src/routers/grpc/harmony/builder.rs`, `sgl-model-gateway/src/routers/http/pd_router.rs` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14332 - feat: V32 tool call parsing for no-dsml tag

- 链接：https://github.com/sgl-project/sglang/pull/14332
- 状态/时间：`open`，created 2025-12-03；作者 `Eva20150932-atlascloud`。
- 代码 diff 已读范围：`2` 个文件，`+481/-44`；代码面：tests/benchmarks；关键词：kv, benchmark, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_function_call_parser.py` modified +334/-0 (334 lines); hunk: def setUp(self):; def test_streaming_json_format(self):; 符号: setUp, test_streaming_json_format, test_detect_and_parse_xml_format_without_dsml, test_detect_and_parse_json_format_without_dsml
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +147/-44 (191 lines); hunk: class DeepSeekV32Detector(BaseFormatDetector):; class DeepSeekV32Detector(BaseFormatDetector):; 符号: DeepSeekV32Detector, DeepSeekV32Detector, DeepSeekV32Detector, DeepSeekV32Detector
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv, benchmark, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14353 - feat(dsv32): better error handling for DeepSeek-v3.2 encoder

- 链接：https://github.com/sgl-project/sglang/pull/14353
- 状态/时间：`merged`，created 2025-12-03, merged 2025-12-19；作者 `jimmy-evo`。
- 代码 diff 已读范围：`2` 个文件，`+53/-32`；代码面：misc；关键词：kv, spec。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/encoding_dsv32.py` modified +45/-32 (77 lines); hunk: import re; def find_last_user_index(messages: List[Dict[str, Any]]) -> int:; 符号: DS32EncodingError, find_last_user_index, render_message, render_message
  - `python/sglang/srt/entrypoints/openai/serving_base.py` modified +8/-0 (8 lines); hunk: from fastapi import HTTPException, Request; async def handle_request(; 符号: handle_request
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py`；patch 关键词为 kv, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/encoding_dsv32.py`, `python/sglang/srt/entrypoints/openai/serving_base.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14524 - [Test] Add test suite for NSA backend

- 链接：https://github.com/sgl-project/sglang/pull/14524
- 状态/时间：`open`，created 2025-12-06；作者 `Johnsonms`。
- 代码 diff 已读范围：`1` 个文件，`+709/-0`；代码面：attention/backend, tests/benchmarks；关键词：attention, cache, config, cuda, eagle, flash, kv, lora, mla, scheduler。
- 代码 diff 细节：
  - `python/sglang/test/attention/test_nsa_backend.py` added +709/-0 (709 lines); hunk: +import unittest; 符号: MockNSAConfig:, __init__, MockModelRunner:, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/attention/test_nsa_backend.py`；patch 关键词为 attention, cache, config, cuda, eagle, flash。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/attention/test_nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14541 - [NPU]dsv3.2 cp for npu

- 链接：https://github.com/sgl-project/sglang/pull/14541
- 状态/时间：`merged`，created 2025-12-06, merged 2025-12-11；作者 `liupeng374`。
- 代码 diff 已读范围：`8` 个文件，`+281/-134`；代码面：attention/backend, tests/benchmarks；关键词：attention, cache, kv, cuda, topk, flash, lora, mla, config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +117/-94 (211 lines); hunk: cp_all_gather_rerange_output,; def forward_npu(; 符号: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +111/-22 (133 lines); hunk: is_mla_preprocess_enabled,; class ForwardMetadata:; 符号: ForwardMetadata:, update_verify_buffers_to_fill_after_draft, init_forward_metadata, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +25/-4 (29 lines); hunk: class NSAContextParallelMetadata:; def prepare_input_dp_with_cp_dsa(; 符号: NSAContextParallelMetadata:, prepare_input_dp_with_cp_dsa
  - `python/sglang/srt/layers/communicator_nsa_cp.py` modified +7/-8 (15 lines); hunk: def _gather_hidden_states_and_residual(; 符号: _gather_hidden_states_and_residual, _scatter_hidden_states_and_residual
  - `python/sglang/srt/distributed/parallel_state.py` modified +3/-6 (9 lines); hunk: def cp_all_gather_into_tensor_async(; 符号: cp_all_gather_into_tensor_async, all_gather
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`；patch 关键词为 attention, cache, kv, cuda, topk, flash。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14572 - [NPU] optimization for dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/14572
- 状态/时间：`merged`，created 2025-12-07, merged 2025-12-12；作者 `ZhengdQin`。
- 代码 diff 已读范围：`11` 个文件，`+141/-68`；代码面：model wrapper, attention/backend, MoE/router, quantization；关键词：config, kv, topk, attention, cuda, expert, lora, moe, quant, router。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +51/-15 (66 lines); hunk: from sglang.srt.layers.layernorm import LayerNorm; def forward_npu(; 符号: forward_npu, forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-18 (52 lines); hunk: def forward_dsa_prepare_npu(; def forward_dsa_core_npu(; 符号: forward_dsa_prepare_npu, forward_dsa_core_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +25/-4 (29 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, forward_deepep
  - `python/sglang/srt/hardware_backend/npu/moe/topk.py` modified +4/-11 (15 lines); hunk: def fused_topk_npu(; def fused_topk_npu(; 符号: fused_topk_npu, fused_topk_npu
  - `python/sglang/srt/layers/layernorm.py` modified +1/-13 (14 lines); hunk: def forward_npu(; 符号: forward_npu, forward_cpu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 config, kv, topk, attention, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14573 - [Tool Call] Fix DeepSeekV32Detector skipping functions with no params in streaming mode

- 链接：https://github.com/sgl-project/sglang/pull/14573
- 状态/时间：`merged`，created 2025-12-07, merged 2025-12-08；作者 `momaek`。
- 代码 diff 已读范围：`2` 个文件，`+144/-7`；代码面：tests/benchmarks；关键词：kv, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_function_call_parser.py` modified +142/-0 (142 lines); hunk: def test_streaming_json_format(self):; 符号: test_streaming_json_format, test_detect_and_parse_no_parameters, test_streaming_no_parameters, test_streaming_no_parameters_with_whitespace
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +2/-7 (9 lines); hunk: def parse_streaming_increment(; 符号: parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14619 - [Sparse & HICache]: Enables hierarchical sparse KV cache management and scheduling for DeepSeek V32.

- 链接：https://github.com/sgl-project/sglang/pull/14619
- 状态/时间：`closed`，created 2025-12-08, closed 2026-03-23；作者 `hzh0425`。
- 代码 diff 已读范围：`30` 个文件，`+3077/-118`；代码面：model wrapper, attention/backend, kernel, multimodal/processor, scheduler/runtime；关键词：cache, kv, attention, config, topk, triton, cuda, flash, spec, benchmark。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py` added +622/-0 (622 lines); hunk: +import torch; 符号: nsa_sparse_diff_triton_kernel, invoke_nsa_sparse_diff_kernel, benchmark_kernel
  - `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunk: +from abc import ABC, abstractmethod; 符号: BaseSparseAlgorithm, for, provides, __init__
  - `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` added +341/-0 (341 lines); hunk: +import logging; 符号: RequestTrackers:, __init__, register, clear
  - `python/sglang/srt/mem_cache/sparsity/core/sparse_kvcache_manager.py` added +237/-0 (237 lines); hunk: +from __future__ import annotations; 符号: SparseKVCacheManager:, __init__, transfer_sparse_top_k_cache, offload_sparse_decode_req_tokens
  - `python/sglang/srt/mem_cache/common.py` modified +195/-39 (234 lines); hunk: import triton; def evict_from_tree_cache(tree_cache: BasePrefixCache \| None, num_tokens: int):; 符号: evict_from_tree_cache, truncate_kv_cache_after_prefill, alloc_paged_token_slots_extend, alloc_paged_token_slots_decode
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py`；patch 关键词为 cache, kv, attention, config, topk, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/sparsity/ops/triton_kernel.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14741 - [1/N][Sparse With Hicache]: Add Sparse Interface

- 链接：https://github.com/sgl-project/sglang/pull/14741
- 状态/时间：`merged`，created 2025-12-09, merged 2025-12-25；作者 `hzh0425`。
- 代码 diff 已读范围：`4` 个文件，`+642/-0`；代码面：scheduler/runtime；关键词：cache, attention, config, kv, spec, topk, cuda, lora, triton。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py` added +383/-0 (383 lines); hunk: +from abc import ABC, abstractmethod; 符号: BaseSparseAlgorithm, for, provides, __init__
  - `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py` added +166/-0 (166 lines); hunk: +"""; 符号: QuestAlgorithm, __init__, _initialize_representation_pools, _compute_page_representations
  - `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` added +80/-0 (80 lines); hunk: +from typing import Any, Optional; 符号: DeepSeekNSAAlgorithm, __init__, retrieve_topk, initialize_representation_pool
  - `python/sglang/srt/mem_cache/sparsity/algorithms/__init__.py` added +13/-0 (13 lines); hunk: +from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py`；patch 关键词为 cache, attention, config, kv, spec, topk。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/sparsity/algorithms/base_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/quest_algorithm.py`, `python/sglang/srt/mem_cache/sparsity/algorithms/deepseek_nsa.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14750 - [Tool Call][DSV32] Streamline function call parameters

- 链接：https://github.com/sgl-project/sglang/pull/14750
- 状态/时间：`merged`，created 2025-12-09, merged 2025-12-26；作者 `Muqi1029`。
- 代码 diff 已读范围：`2` 个文件，`+60/-29`；代码面：tests/benchmarks；关键词：kv, spec, test。
- 代码 diff 细节：
  - `test/registered/function_call/test_function_call_parser.py` modified +37/-14 (51 lines); hunk: def setUp(self):; def test_streaming_xml_format(self):; 符号: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_xml_format
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +23/-15 (38 lines); hunk: import logging; ToolCallItem,; 符号: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/function_call/test_function_call_parser.py`, `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14781 - [Performance] optimize NSA backend metadata computation for multi-step speculative decoding

- 链接：https://github.com/sgl-project/sglang/pull/14781
- 状态/时间：`merged`，created 2025-12-10, merged 2025-12-18；作者 `Johnsonms`。
- 代码 diff 已读范围：`3` 个文件，`+440/-16`；代码面：attention/backend；关键词：attention, cache, spec, cuda, flash, kv, mla, quant, topk, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py` added +324/-0 (324 lines); hunk: +"""Multi-step precompute utilities for Native Sparse Attention backend.; 符号: PrecomputedMetadata:, compute_cu_seqlens, NativeSparseAttnBackendMTPPrecomputeMixin:, providing
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +111/-16 (127 lines); hunk: from sglang.srt.environ import envs; def topk_transform(; 符号: topk_transform, compute_cu_seqlens, NativeSparseAttnBackend, NativeSparseAttnBackend
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +5/-0 (5 lines); hunk: NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "true"); 符号: print_nsa_bool_env_vars
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`；patch 关键词为 attention, cache, spec, cuda, flash, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_backend_mtp_precompute.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14901 - fix ds3.2 nsa backend prefill TBO

- 链接：https://github.com/sgl-project/sglang/pull/14901
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-21；作者 `Chen-0210`。
- 代码 diff 已读范围：`5` 个文件，`+76/-2`；代码面：model wrapper, attention/backend, tests/benchmarks；关键词：attention, cuda, deepep, kv, moe, test, awq, fp8, mla, quant。
- 代码 diff 细节：
  - `test/srt/ep/test_deepep_large.py` modified +55/-0 (55 lines); hunk: from sglang.srt.utils import kill_process_tree; popen_launch_server,; 符号: TestDeepseek, test_gsm8k, TestDeepseekV32TBO, setUpClass
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-1 (9 lines); hunk: is_nsa_enable_prefill_cp,; def handle_attention_nsa(attn, forward_batch):; 符号: handle_attention_nsa, _get_mla_kv_buffer_from_fp8
  - `python/sglang/srt/server_args.py` modified +9/-0 (9 lines); hunk: def __post_init__(self):; def _handle_other_validations(self):; 符号: __post_init__, _handle_deprecated_args, _handle_other_validations, _handle_two_batch_overlap
  - `python/sglang/srt/layers/attention/tbo_backend.py` modified +3/-0 (3 lines); hunk: def forward_extend(self, *args, **kwargs):; 符号: forward_extend, forward_decode, get_indexer_metadata, _init_forward_metadata_cuda_graph_split
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunk: # TestFile("ep/test_mooncake_ep_small.py", 450),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py`；patch 关键词为 attention, cuda, deepep, kv, moe, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/ep/test_deepep_large.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14904 - [DeepSeek V3.2] Proper drop_thinking logic

- 链接：https://github.com/sgl-project/sglang/pull/14904
- 状态/时间：`closed`，created 2025-12-11, closed 2025-12-13；作者 `vladnosiv`。
- 代码 diff 已读范围：`1` 个文件，`+5/-1`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +5/-1 (6 lines); hunk: def _apply_jinja_template(; 符号: _apply_jinja_template
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15040 - [DSv32] Move deep_gemm.get_paged_mqa_logits_metadata to init time as metadata

- 链接：https://github.com/sgl-project/sglang/pull/15040
- 状态/时间：`merged`，created 2025-12-13, merged 2025-12-19；作者 `qianlihuang`。
- 代码 diff 已读范围：`2` 个文件，`+91/-5`；代码面：attention/backend；关键词：attention, topk, cache, cuda, flash, fp8, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +84/-1 (85 lines); hunk: from sglang.srt.layers.attention.trtllm_mla_backend import _concat_mla_absorb_q_general; class NSAMetadata:; 符号: NSAMetadata:, _cat, NSAIndexerMetadata, get_seqlens_int32
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-4 (11 lines); hunk: def _get_topk_paged(; 符号: _get_topk_paged
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, topk, cache, cuda, flash, fp8。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15051 - feat(ds32): support <function_call> tag for deepseek 3.2 tool call

- 链接：https://github.com/sgl-project/sglang/pull/15051
- 状态/时间：`closed`，created 2025-12-13, closed 2025-12-16；作者 `jimmy-evo`。
- 代码 diff 已读范围：`1` 个文件，`+56/-10`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +56/-10 (66 lines); hunk: def __init__(self):; def _parse_parameters_from_xml(self, invoke_content: str) -> dict:; 符号: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15064 - fix: dpskv32 chat history processing, default drop_thinking to true

- 链接：https://github.com/sgl-project/sglang/pull/15064
- 状态/时间：`merged`，created 2025-12-13, merged 2025-12-13；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+1/-3`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +1/-3 (4 lines); hunk: def _apply_jinja_template(; 符号: _apply_jinja_template
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15086 - [NSA] Fix NSA backend assertion error when running DeepSeek-V3.2 PP with radix-cache

- 链接：https://github.com/sgl-project/sglang/pull/15086
- 状态/时间：`merged`，created 2025-12-13, merged 2025-12-15；作者 `YAMY1234`。
- 代码 diff 已读范围：`2` 个文件，`+19/-5`；代码面：attention/backend, quantization, scheduler/runtime；关键词：attention, cache, kv, quant, fp8, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +17/-2 (19 lines); hunk: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_forward_metadata
  - `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` modified +2/-3 (5 lines); hunk: def dequantize_k_cache_paged(; 符号: dequantize_k_cache_paged
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py`；patch 关键词为 attention, cache, kv, quant, fp8, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/dequant_k_cache.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15088 - [DeepSeekV3.2] Add pure TP+MTP test

- 链接：https://github.com/sgl-project/sglang/pull/15088
- 状态/时间：`merged`，created 2025-12-14, merged 2025-12-17；作者 `ashtonchew`。
- 代码 diff 已读范围：`2` 个文件，`+107/-7`；代码面：tests/benchmarks, docs/config；关键词：config, eagle, spec, topk, attention, cache, cuda, doc, flash, kv。
- 代码 diff 细节：
  - `test/nightly/test_deepseek_v32_tp.py` modified +100/-6 (106 lines); hunk: import unittest; write_github_step_summary,; 符号: test_a_gsm8k, TestDeepseekV32_TP_MTP, setUpClass, tearDownClass
  - `docs/basic_usage/deepseek_v32.md` modified +7/-1 (8 lines); hunk: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md`；patch 关键词为 config, eagle, spec, topk, attention, cache。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/nightly/test_deepseek_v32_tp.py`, `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15217 - fix(DeepSeek-V3.2 function_call): fix streaming content loss in DeepSeekV32Detector

- 链接：https://github.com/sgl-project/sglang/pull/15217
- 状态/时间：`closed`，created 2025-12-16, closed 2025-12-16；作者 `momaek`。
- 代码 diff 已读范围：`1` 个文件，`+3/-3`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +3/-3 (6 lines); hunk: def parse_streaming_increment(; 符号: parse_streaming_increment
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15242 - [sgl-kernel] Update flashmla to include fp8 sparse_mla optimizations

- 链接：https://github.com/sgl-project/sglang/pull/15242
- 状态/时间：`merged`，created 2025-12-16, merged 2025-12-16；作者 `hlu1`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend, kernel；关键词：flash, mla。
- 代码 diff 细节：
  - `sgl-kernel/cmake/flashmla.cmake` modified +1/-1 (2 lines); hunk: include(FetchContent)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `sgl-kernel/cmake/flashmla.cmake`；patch 关键词为 flash, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `sgl-kernel/cmake/flashmla.cmake` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15278 - feat: DeepSeek-V3.2 Streaming tool call output

- 链接：https://github.com/sgl-project/sglang/pull/15278
- 状态/时间：`merged`，created 2025-12-16, merged 2025-12-18；作者 `JustinTong0323`。
- 代码 diff 已读范围：`2` 个文件，`+111/-69`；代码面：tests/benchmarks；关键词：kv, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +95/-65 (160 lines); hunk: import json; ToolCallItem,; 符号: __init__, has_tool_call, _parse_parameters_from_xml, _parse_parameters_from_xml
  - `test/registered/function_call/test_function_call_parser.py` modified +16/-4 (20 lines); hunk: def setUp(self):; def test_streaming_xml_format(self):; 符号: setUp, test_detect_and_parse_xml_format, test_streaming_xml_format, test_streaming_json_format
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`；patch 关键词为 kv, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15307 - [Deepseek V3.2] Support Overlap Spec + NSA

- 链接：https://github.com/sgl-project/sglang/pull/15307
- 状态/时间：`merged`，created 2025-12-17, merged 2025-12-17；作者 `b8zhong`。
- 代码 diff 已读范围：`3` 个文件，`+25/-8`；代码面：attention/backend, docs/config；关键词：topk, attention, cuda, spec, cache, config, doc, eagle, flash, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +19/-6 (25 lines); hunk: def init_forward_metadata(self, forward_batch: ForwardBatch):; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_forward_metadata, init_forward_metadata_capture_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `docs/basic_usage/deepseek_v32.md` modified +4/-0 (4 lines); hunk: python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --dp
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-2 (4 lines); hunk: def _get_topk_paged(; def forward_cuda(; 符号: _get_topk_paged, forward_cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 topk, attention, cuda, spec, cache, config。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15310 - [Deepseek V3.2] Enable TRTLLM Allreduce Fusion

- 链接：https://github.com/sgl-project/sglang/pull/15310
- 状态/时间：`closed`，created 2025-12-17, closed 2026-01-06；作者 `b8zhong`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：misc；关键词：kv, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +1/-0 (1 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 kv, moe, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15322 - dsv32 support o_proj tp

- 链接：https://github.com/sgl-project/sglang/pull/15322
- 状态/时间：`open`，created 2025-12-17；作者 `lawtherWu`。
- 代码 diff 已读范围：`14` 个文件，`+472/-23`；代码面：model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime；关键词：moe, config, attention, cache, expert, quant, spec, topk, cuda, deepep。
- 代码 diff 细节：
  - `python/sglang/srt/layers/communicator.py` modified +179/-5 (184 lines); hunk: import torch; get_moe_a2a_backend,; 符号: enable_moe_dense_fully_dp, get_max_bs_across_dp, LayerCommunicator:, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +74/-14 (88 lines); hunk: from sglang.srt.distributed import (; from sglang.srt.layers.attention.tbo_backend import TboAttnBackend; 符号: forward_deepep, __init__, __init__, __init__
  - `python/sglang/srt/distributed/parallel_state.py` modified +85/-0 (85 lines); hunk: def get_moe_tp_group() -> GroupCoordinator:; def initialize_model_parallel(; 符号: get_moe_tp_group, get_o_proj_tp_group, get_o_proj_dp_group, initialize_model_parallel
  - `python/sglang/srt/layers/linear.py` modified +73/-1 (74 lines); hunk: from sglang.srt.distributed import (; def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):; 符号: weight_loader, weight_loader, extra_repr, TP2DPandTPRowParallelLinear
  - `python/sglang/srt/distributed/communication_op.py` modified +20/-1 (21 lines); hunk: import torch; def broadcast_tensor_dict(; 符号: tensor_model_parallel_all_reduce, broadcast_tensor_dict, o_proj_tensor_model_parallel_reduce_scatter_tensor, o_proj_tensor_model_parallel_all_reduce
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py`；patch 关键词为 moe, config, attention, cache, expert, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/distributed/parallel_state.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15381 - [NPU]DeepSeek-V3.2 support npu mlaprolog

- 链接：https://github.com/sgl-project/sglang/pull/15381
- 状态/时间：`merged`，created 2025-12-18, merged 2026-01-26；作者 `lawtherWu`。
- 代码 diff 已读范围：`5` 个文件，`+195/-61`；代码面：attention/backend, quantization；关键词：attention, kv, cache, lora, mla, quant, config, topk。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +116/-55 (171 lines); hunk: +import re; def forward_dsa_prepare_npu(; 符号: forward_dsa_prepare_npu, forward_dsa_prepare_npu, forward_dsa_core_npu, npu_mla_preprocess
  - `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` modified +63/-1 (64 lines); hunk: +import re; def __init__(; 符号: __init__, __init__, preprocess_weights, preprocess_weights
  - `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` modified +8/-2 (10 lines); hunk: def apply(; 符号: apply
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +3/-3 (6 lines); hunk: def forward_extend(; def forward_extend(; 符号: forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-0 (5 lines); hunk: def forward_npu(; def forward_npu(; 符号: forward_npu, forward_npu, forward_npu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`；patch 关键词为 attention, kv, cache, lora, mla, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`, `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15429 - [Deepseek V3.2] Fix Deepseek MTP in V1 mode

- 链接：https://github.com/sgl-project/sglang/pull/15429
- 状态/时间：`merged`，created 2025-12-19, merged 2025-12-19；作者 `b8zhong`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunk: def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15807 - [2/N][Sparse With Hicache]: Support separating nsa memory management for KV cache and index_k in decode side.

- 链接：https://github.com/sgl-project/sglang/pull/15807
- 状态/时间：`closed`，created 2025-12-25, closed 2026-03-23；作者 `hzh0425`。
- 代码 diff 已读范围：`10` 个文件，`+516/-39`；代码面：attention/backend, scheduler/runtime；关键词：cache, kv, attention, cuda, mla, spec, topk, config, flash, fp8。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/allocator.py` modified +156/-0 (156 lines); hunk: def get_cpu_copy(self, indices):; 符号: get_cpu_copy, load_cpu_copy, NSAHybridTokenToKVPoolAllocator, __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +89/-0 (89 lines); hunk: ); class NSAMetadata:; 符号: NSAMetadata:, get_seqlens_int32, get_page_table_64, get_seqlens_expanded
  - `python/sglang/srt/mem_cache/common.py` modified +68/-5 (73 lines); hunk: import triton; def alloc_for_decode(batch: ScheduleBatch, token_per_req: int) -> torch.Tensor:; 符号: alloc_for_decode, _alloc_for_nsa_index_k, release_kv_cache, release_kv_cache
  - `python/sglang/srt/model_executor/model_runner.py` modified +48/-17 (65 lines); hunk: def init_memory_pool(; def init_memory_pool(; 符号: init_memory_pool, init_memory_pool, init_memory_pool, init_memory_pool
  - `python/sglang/srt/disaggregation/decode.py` modified +56/-4 (60 lines); hunk: from sglang.srt.managers.utils import GenerationBatchResult; def clear(self):; 符号: clear, NSADecodeReqToTokenPool, __init__, write_index_token
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py`；patch 关键词为 cache, kv, attention, cuda, mla, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/allocator.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/mem_cache/common.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15938 - Clean Some Environment Variables for DeepSeek V32

- 链接：https://github.com/sgl-project/sglang/pull/15938
- 状态/时间：`merged`，created 2025-12-27, merged 2026-01-07；作者 `Fridge003`。
- 代码 diff 已读范围：`8` 个文件，`+39/-108`；代码面：attention/backend, quantization, scheduler/runtime, docs/config；关键词：attention, cache, cuda, fp8, quant, topk, flash, mla, spec, config。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/quant_k_cache.py` modified +6/-42 (48 lines); hunk: import triton; def quantize_k_cache_separate(; 符号: quantize_k_cache, quantize_k_cache_separate, quantize_k_cache_separate, _quantize_k_cache_slow
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-19 (27 lines); hunk: transform_index_page_table_prefill,; def topk_transform(; 符号: topk_transform, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +0/-24 (24 lines); hunk: get_attention_tp_size,; 符号: print_nsa_bool_env_vars, compute_nsa_seqlens
  - `python/sglang/srt/server_args.py` modified +8/-13 (21 lines); hunk: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments, _handle_model_specific_adjustments, _handle_model_specific_adjustments
  - `docs/references/environment_variables.md` modified +10/-0 (10 lines); hunk: SGLang supports various environment variables that can be used to configure its
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py`；patch 关键词为 attention, cache, cuda, fp8, quant, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/quant_k_cache.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16079 - [Performance] Change sparse MLA and dense MHA switching threshold DSv3.2

- 链接：https://github.com/sgl-project/sglang/pull/16079
- 状态/时间：`closed`，created 2025-12-29, closed 2026-03-25；作者 `zhangxiaolei123456`。
- 代码 diff 已读范围：`1` 个文件，`+4/-2`；代码面：attention/backend；关键词：attention, cache, config, flash, fp8, kv, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-2 (6 lines); hunk: from sglang.srt.layers.dp_attention import get_attention_tp_size; def __init__(; 符号: __init__, set_nsa_prefill_impl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, config, flash, fp8, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16091 - [Tool Call] Stream DeepSeek-V3.2 function call parameters in JSON format.

- 链接：https://github.com/sgl-project/sglang/pull/16091
- 状态/时间：`merged`，created 2025-12-29, merged 2026-03-03；作者 `Muqi1029`。
- 代码 diff 已读范围：`2` 个文件，`+31/-22`；代码面：tests/benchmarks；关键词：kv, test。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +17/-21 (38 lines); hunk: def __init__(self):; def has_tool_call(self, text: str) -> bool:; 符号: __init__, has_tool_call, has_tool_call, _parse_parameters_from_xml
  - `test/registered/function_call/test_function_call_parser.py` modified +14/-1 (15 lines); hunk: def test_streaming_xml_format(self):; def test_streaming_xml_format(self):; 符号: test_streaming_xml_format, test_streaming_xml_format, test_streaming_json_format, test_streaming_json_format
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py`；patch 关键词为 kv, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv32_detector.py`, `test/registered/function_call/test_function_call_parser.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16119 - [cp] bug fix for dsv3.2 cp

- 链接：https://github.com/sgl-project/sglang/pull/16119
- 状态/时间：`merged`，created 2025-12-30, merged 2025-12-30；作者 `liupeng374`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention, cache, kv, lora, mla。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +1/-1 (2 lines); hunk: def forward_dsa_prepare_npu(; 符号: forward_dsa_prepare_npu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`；patch 关键词为 attention, cache, kv, lora, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16156 - [cp] assert dsv3.2 cp in pd decode mode

- 链接：https://github.com/sgl-project/sglang/pull/16156
- 状态/时间：`merged`，created 2025-12-30, merged 2025-12-31；作者 `liupeng374`。
- 代码 diff 已读范围：`1` 个文件，`+4/-0`；代码面：misc；关键词：spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +4/-0 (4 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16305 - Multiple updates of DeepSeek V32 and context parallel

- 链接：https://github.com/sgl-project/sglang/pull/16305
- 状态/时间：`merged`，created 2026-01-02, merged 2026-01-02；作者 `Fridge003`。
- 代码 diff 已读范围：`7` 个文件，`+190/-35`；代码面：tests/benchmarks, docs/config；关键词：test, attention, kv, spec, cache, deepep, eagle, fp8, moe, benchmark。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_mtp.py` modified +81/-1 (82 lines); hunk: FULL_DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_bs_1_speed(self):; 符号: TestDeepseekV32MTP, TestDeepseekV32DPMTP, setUpClass, test_bs_1_speed
  - `test/srt/test_deepseek_v32_basic.py` modified +56/-1 (57 lines); hunk: DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2-Exp"; def test_bs_1_speed(self):; 符号: TestDeepseekV32Basic, TestDeepseekV32DP, setUpClass, test_bs_1_speed
  - `docs/basic_usage/deepseek_v32.md` modified +30/-21 (51 lines); hunk: DeepSeek-V3.2-Speciale:
  - `python/sglang/srt/server_args.py` modified +16/-5 (21 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `test/srt/run_suite.py` modified +3/-2 (5 lines); hunk: TestFile("test_deepseek_v3_mtp.py", 275),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md`；patch 关键词为 test, attention, kv, spec, cache, deepep。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_mtp.py`, `test/srt/test_deepseek_v32_basic.py`, `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16306 - [1/n]deepseek_v2.py Refactor: attention backend handlers and forward method definition

- 链接：https://github.com/sgl-project/sglang/pull/16306
- 状态/时间：`merged`，created 2026-01-02, merged 2026-01-08；作者 `Fridge003`。
- 代码 diff 已读范围：`5` 个文件，`+255/-228`；代码面：model wrapper, attention/backend；关键词：attention, cache, cuda, kv, mla, flash, fp8, quant, spec, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +18/-228 (246 lines); hunk: import logging; from sglang.srt.layers.quantization.fp8 import Fp8Config; 符号: add_forward_absorb_core_attention_backend, AttnForwardMethod, _dispatch_mla_subtype, AttentionBackendRegistry:
  - `python/sglang/srt/models/deepseek_common/attention_backend_handler.py` added +182/-0 (182 lines); hunk: +from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph; 符号: AttentionBackendRegistry:, register, get_handler, _dispatch_mla_subtype
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` added +32/-0 (32 lines); hunk: +from enum import IntEnum, auto; 符号: AttnForwardMethod
  - `python/sglang/srt/models/deepseek_common/utils.py` added +23/-0 (23 lines); hunk: +from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
  - `python/sglang/srt/models/deepseek_common/__init__.py` added +0/-0 (0 lines)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py`；patch 关键词为 attention, cache, cuda, kv, mla, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_backend_handler.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_methods.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16380 - [DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled

- 链接：https://github.com/sgl-project/sglang/pull/16380
- 状态/时间：`merged`，created 2026-01-04, merged 2026-01-09；作者 `xu-yfei`。
- 代码 diff 已读范围：`2` 个文件，`+72/-36`；代码面：attention/backend, scheduler/runtime；关键词：attention, cache, config, cuda, fp8, kv, lora, scheduler, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +57/-33 (90 lines); hunk: from __future__ import annotations; import torch_npu; 符号: __init__, __init__, _with_real_sm_count, _get_logits_head_gate
  - `python/sglang/srt/managers/scheduler_pp_mixin.py` modified +15/-3 (18 lines); hunk: def event_loop_pp_disagg_decode(self: Scheduler):; def _pp_send_dict_to_next_stage(; 符号: event_loop_pp_disagg_decode, init_pp_loop_state, _pp_send_dict_to_next_stage, _pp_recv_proxy_tensors
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py`；patch 关键词为 attention, cache, config, cuda, fp8, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/scheduler_pp_mixin.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16520 - fix: unimplemented methods in BaseIndexerMetadata

- 链接：https://github.com/sgl-project/sglang/pull/16520
- 状态/时间：`merged`，created 2026-01-05, merged 2026-01-06；作者 `dougyster`。
- 代码 diff 已读范围：`1` 个文件，`+39/-1`；代码面：kernel, tests/benchmarks；关键词：cache, kv, test, topk。
- 代码 diff 细节：
  - `test/registered/kernels/test_nsa_indexer.py` modified +39/-1 (40 lines); hunk: import unittest; def get_seqlens_expanded(self) -> torch.Tensor:; 符号: get_seqlens_expanded, get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_token_to_batch_idx
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/kernels/test_nsa_indexer.py`；patch 关键词为 cache, kv, test, topk。影响判断：CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/kernels/test_nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16637 - [DSv32] Overlap indexer weights_proj during dual_stream decode

- 链接：https://github.com/sgl-project/sglang/pull/16637
- 状态/时间：`merged`，created 2026-01-07, merged 2026-01-10；作者 `zianglih`。
- 代码 diff 已读范围：`2` 个文件，`+64/-22`；代码面：model wrapper, attention/backend；关键词：cuda, lora, attention, cache, fp8, kv, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +34/-11 (45 lines); hunk: def forward_absorb_prepare(; def forward_absorb_prepare(; 符号: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-11 (41 lines); hunk: def _with_real_sm_count(self):; def forward_cuda(; 符号: _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate, forward_cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 cuda, lora, attention, cache, fp8, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16758 - [DeepSeek V3.2] Enable trtllm NSA with bf16 kvcache

- 链接：https://github.com/sgl-project/sglang/pull/16758
- 状态/时间：`merged`，created 2026-01-08, merged 2026-01-23；作者 `akhilg-nv`。
- 代码 diff 已读范围：`2` 个文件，`+118/-31`；代码面：attention/backend；关键词：attention, cache, config, cuda, flash, kv, mla, fp4, fp8, lora。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +64/-28 (92 lines); hunk: "fa3",; class ServerArgs:; 符号: ServerArgs:, _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +54/-3 (57 lines); hunk: def topk_transform(; def __init__(; 符号: topk_transform, NativeSparseAttnBackend, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, config, cuda, flash, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16841 - [AMD] enable CUDA graph for NSA backend and fix NSA FP8 fused RMSNorm group quant

- 链接：https://github.com/sgl-project/sglang/pull/16841
- 状态/时间：`merged`，created 2026-01-10, merged 2026-01-14；作者 `hubertlu-tw`。
- 代码 diff 已读范围：`7` 个文件，`+260/-81`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime；关键词：attention, cache, kv, fp8, cuda, triton, quant, spec, topk, lora。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +171/-55 (226 lines); hunk: from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu; if TYPE_CHECKING:; 符号: BaseIndexerMetadata, get_page_table_64, get_page_table_1, get_seqlens_expanded
  - `python/sglang/srt/models/deepseek_v2.py` modified +58/-18 (76 lines); hunk: def forward_normal_prepare(; def forward_absorb_prepare(; 符号: forward_normal_prepare, forward_absorb_prepare
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +10/-3 (13 lines); hunk: import triton; def _set_k_and_s_triton(; 符号: _set_k_and_s_triton
  - `python/sglang/srt/server_args.py` modified +10/-3 (13 lines); hunk: def _handle_model_specific_adjustments(self):; def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments, _handle_model_specific_adjustments
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +6/-2 (8 lines); hunk: set_mla_kv_buffer_triton,; GB = 1024 * 1024 * 1024; 符号: get_tensor_size_bytes, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`；patch 关键词为 attention, cache, kv, fp8, cuda, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16881 - [DSv32] Add returning DSA topk indices

- 链接：https://github.com/sgl-project/sglang/pull/16881
- 状态/时间：`closed`，created 2026-01-11, closed 2026-01-11；作者 `zianglih`。
- 代码 diff 已读范围：`15` 个文件，`+205/-2`；代码面：model wrapper, attention/backend, MoE/router, multimodal/processor, scheduler/runtime, tests/benchmarks；关键词：expert, topk, cache, config, lora, moe, scheduler, cuda, kv, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/routed_experts_capturer.py` modified +118/-0 (118 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, get_buffer_size_bytes, get_dsa_topk_indices_buffer_size_bytes
  - `python/sglang/srt/managers/detokenizer_manager.py` modified +17/-2 (19 lines); hunk: def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):; def _extract_routed_experts(self, recv_obj: BatchTokenIDOutput) -> List[List[int; 符号: _decode_batch_token_id_output, _extract_routed_experts, _extract_routed_experts, handle_batch_token_id_out
  - `python/sglang/srt/managers/scheduler_output_processor_mixin.py` modified +16/-0 (16 lines); hunk: def maybe_collect_routed_experts(self: Scheduler, req: Req):; def process_batch_result_prefill(; 符号: maybe_collect_routed_experts, maybe_collect_dsa_topk_indices, maybe_collect_customized_info, process_batch_result_prefill
  - `python/sglang/srt/managers/io_struct.py` modified +12/-0 (12 lines); hunk: class GenerateReqInput(BaseReq, APIServingTimingMixin):; def __getitem__(self, i):; 符号: GenerateReqInput, __getitem__, TokenizedGenerateReqInput, BatchTokenIDOutput
  - `python/sglang/srt/managers/schedule_batch.py` modified +11/-0 (11 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, reset_for_retract, ScheduleBatch
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py`；patch 关键词为 expert, topk, cache, config, lora, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/routed_experts_capturer.py`, `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/scheduler_output_processor_mixin.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16907 - Fix model loading for DeepSeek-V3.2-AWQ

- 链接：https://github.com/sgl-project/sglang/pull/16907
- 状态/时间：`merged`，created 2026-01-11, merged 2026-02-15；作者 `bingps`。
- 代码 diff 已读范围：`1` 个文件，`+8/-4`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +8/-4 (12 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16916 - add doc for dsv32 cp+pp

- 链接：https://github.com/sgl-project/sglang/pull/16916
- 状态/时间：`merged`，created 2026-01-12, merged 2026-01-12；作者 `whybeyoung`。
- 代码 diff 已读范围：`1` 个文件，`+114/-0`；代码面：docs/config；关键词：benchmark, cache, config, cuda, doc, kv, moe, test。
- 代码 diff 细节：
  - `docs/basic_usage/deepseek_v32.md` modified +114/-0 (114 lines); hunk: Latency: 29.545 s; Example usage:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/deepseek_v32.md`；patch 关键词为 benchmark, cache, config, cuda, doc, kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16961 - [DeepSeek v3.2] Opt MTP decode cuda batch sizes and nsa implementation

- 链接：https://github.com/sgl-project/sglang/pull/16961
- 状态/时间：`merged`，created 2026-01-12, merged 2026-01-19；作者 `xu-yfei`。
- 代码 diff 已读范围：`2` 个文件，`+26/-12`；代码面：attention/backend, kernel, scheduler/runtime；关键词：attention, cache, config, cuda, flash, kv, mla, moe, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +14/-5 (19 lines); hunk: def forward_extend(; def forward_extend(; 符号: forward_extend, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +12/-7 (19 lines); hunk: def set_torch_compile_config():; def get_batch_sizes_to_capture(model_runner: ModelRunner):; 符号: set_torch_compile_config, get_batch_sizes_to_capture, get_batch_sizes_to_capture, get_batch_sizes_to_capture
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`；patch 关键词为 attention, cache, config, cuda, flash, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #16990 - [Ascend] fix dsv3.2 weight cast bug

- 链接：https://github.com/sgl-project/sglang/pull/16990
- 状态/时间：`merged`，created 2026-01-13, merged 2026-01-13；作者 `MichelleWu351`。
- 代码 diff 已读范围：`1` 个文件，`+3/-2`；代码面：quantization；关键词：flash, moe, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/unquant.py` modified +3/-2 (5 lines); hunk: def create_weights(; def process_weights_after_loading(self, layer: torch.nn.Module) -> None:; 符号: create_weights, process_weights_after_loading, apply, process_weights_after_loading
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/unquant.py`；patch 关键词为 flash, moe, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/unquant.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17007 - [NPU]bugfix: fix for dsv3.2 and dsvl2

- 链接：https://github.com/sgl-project/sglang/pull/17007
- 状态/时间：`merged`，created 2026-01-13, merged 2026-01-23；作者 `JiaruiChang5268`。
- 代码 diff 已读范围：`5` 个文件，`+129/-46`；代码面：model wrapper, attention/backend, tests/benchmarks, docs/config；关键词：attention, cache, kv, config, lora, mla, quant, test, benchmark, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +80/-46 (126 lines); hunk: def forward_mha_prepare_npu(; def forward_dsa_prepare_npu(; 符号: forward_mha_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu
  - `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py` added +29/-0 (29 lines); hunk: +import unittest; 符号: TestDeepSeekV3_2ExpW8A8
  - `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py` added +18/-0 (18 lines); hunk: +import unittest; 符号: TestGemmaModels, test_vlm_mmmu_benchmark
  - `python/sglang/srt/configs/model_config.py` modified +1/-0 (1 lines); hunk: def _derive_model_shapes(self):; 符号: _derive_model_shapes
  - `python/sglang/srt/models/deepseek_v2.py` modified +1/-0 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py`；patch 关键词为 attention, cache, kv, config, lora, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `test/registered/ascend/llm_models/test_ascend_deepseek_v3_2_exp_w8a8.py`, `test/registered/ascend/vlm_models/test_ascend_deepseek_vl2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17024 - [PD] Fix DeepSeek V3.2 indexer cache transfer

- 链接：https://github.com/sgl-project/sglang/pull/17024
- 状态/时间：`closed`，created 2026-01-13, closed 2026-03-19；作者 `ShangmingCai`。
- 代码 diff 已读范围：`1` 个文件，`+6/-10`；代码面：misc；关键词：cache, kv。
- 代码 diff 细节：
  - `python/sglang/srt/disaggregation/prefill.py` modified +6/-10 (16 lines); hunk: def send_kv_chunk(; def send_kv_chunk(; 符号: send_kv_chunk, send_kv_chunk
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/disaggregation/prefill.py`；patch 关键词为 cache, kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/disaggregation/prefill.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17076 - [DeepSeek V3.2] [Bugfix] slice indexer and padding fa3 when can not run cuda graph

- 链接：https://github.com/sgl-project/sglang/pull/17076
- 状态/时间：`merged`，created 2026-01-14, merged 2026-02-02；作者 `xu-yfei`。
- 代码 diff 已读范围：`4` 个文件，`+58/-7`；代码面：attention/backend, kernel, tests/benchmarks；关键词：attention, cache, kv, triton, fp8, test, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +28/-4 (32 lines); hunk: import triton.language as tl; def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):; 符号: nsa_cp_round_robin_split_data, pad_nsa_cache_seqlens, cal_padded_tokens, pad_nsa_cache_seqlens
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-2 (22 lines); hunk: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; def _get_topk_paged(; 符号: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx, _get_topk_paged
  - `test/registered/kernels/test_nsa_indexer.py` modified +7/-1 (8 lines); hunk: import unittest; def get_indexer_seq_len_cpu(self) -> torch.Tensor:; 符号: get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-0 (3 lines); hunk: def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:; 符号: get_indexer_kvcache_range, get_indexer_seq_len_cpu, get_nsa_extend_len_cpu, get_token_to_batch_idx
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py`；patch 关键词为 attention, cache, kv, triton, fp8, test。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/kernels/test_nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #17199 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/17199
- 状态/时间：`closed`，created 2026-01-16, closed 2026-02-26；作者 `JiaruiChang5268`。
- 代码 diff 已读范围：`5` 个文件，`+191/-82`；代码面：model wrapper, attention/backend；关键词：lora, kv, attention, cache, mla, config, cuda, topk, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` modified +119/-70 (189 lines); hunk: import torch_npu; def __init__(self, model_runner: ModelRunner):; 符号: __init__, init_forward_metadata, _generate_alibi_bias, generate_alibi_bias
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +34/-6 (40 lines); hunk: cp_split_and_rebuild_position,; def forward_mha_prepare_npu(; 符号: forward_mha_prepare_npu, forward_mha_prepare_npu, forward_dsa_prepare_npu, forward_dsa_prepare_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-3 (22 lines); hunk: from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode; def forward_npu(; 符号: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +15/-2 (17 lines); hunk: _is_cpu = is_cpu(); def forward(; 符号: forward, forward, forward_prepare, forward_prepare
  - `python/sglang/srt/layers/communicator.py` modified +4/-1 (5 lines); hunk: _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip(); def __init__(self):; 符号: __init__, init_context, get_fn
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 lora, kv, attention, cache, mla, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17205 - [OPT] DeepSeekV3.2: optimize indexer weight_proj-mma performance

- 链接：https://github.com/sgl-project/sglang/pull/17205
- 状态/时间：`merged`，created 2026-01-16, merged 2026-01-20；作者 `BJWang-ant`。
- 代码 diff 已读范围：`1` 个文件，`+5/-4`；代码面：attention/backend；关键词：attention, config, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +5/-4 (9 lines); hunk: def __init__(; def _with_real_sm_count(self):; 符号: __init__, _with_real_sm_count, _project_and_scale_head_gates, _get_logits_head_gate
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, config, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17310 - [TileLang] Align TileLang NSA kernel with current TileLang and stabilize output

- 链接：https://github.com/sgl-project/sglang/pull/17310
- 状态/时间：`closed`，created 2026-01-18, closed 2026-01-25；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+56/-60`；代码面：attention/backend, kernel；关键词：attention, config, fp8, kv, quant, spec, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +56/-60 (116 lines); hunk: pass_configs = {; def fast_round_scale(amax, fp8_max_inv):; 符号: fast_log2_ceil, fast_pow2, fast_round_scale, fast_round_scale
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`；patch 关键词为 attention, config, fp8, kv, quant, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17409 - [Fix]: correctly fetch ds32 config in tuning_fused_moe_triton

- 链接：https://github.com/sgl-project/sglang/pull/17409
- 状态/时间：`merged`，created 2026-01-20, merged 2026-01-20；作者 `huangzhilin-hzl`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：MoE/router, kernel, tests/benchmarks；关键词：benchmark, config, expert, moe, topk, triton。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +2/-2 (4 lines); hunk: from typing import Dict, List, TypedDict; def get_model_config(; 符号: BenchmarkConfig, get_model_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/common_utils.py`；patch 关键词为 benchmark, config, expert, moe, topk, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/common_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17518 - [HotFix]Fix dtype mismatch in nsa indexer on AMD device

- 链接：https://github.com/sgl-project/sglang/pull/17518
- 状态/时间：`merged`，created 2026-01-21, merged 2026-01-22；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +1/-1 (2 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #17554 - Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels

- 链接：https://github.com/sgl-project/sglang/pull/17554
- 状态/时间：`merged`，created 2026-01-22, merged 2026-02-14；作者 `Johnsonms`。
- 代码 diff 已读范围：`7` 个文件，`+2824/-54`；代码面：attention/backend, kernel, tests/benchmarks；关键词：cache, mla, flash, attention, cuda, spec, config, benchmark, quant, scheduler。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py` added +1067/-0 (1067 lines); hunk: +"""; 符号: create_test_metadata, reference_copy_decode, reference_copy_target_verify, reference_copy_draft_extend
  - `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh` added +722/-0 (722 lines); hunk: +/*; 符号: SourcePointers, DestinationPointers, FusedMetadataCopyParams, FusedMetadataCopyMultiParams
  - `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` added +407/-0 (407 lines); hunk: +"""; 符号: verify_single_backend_fused_metadata_copy, check_tensor_equal, verify_multi_backend_fused_metadata_copy, check_tensor_equal
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +307/-51 (358 lines); hunk: compute_cu_seqlens,; # Reuse this workspace buffer across all NSA backend instances; 符号: NSAFlashMLAMetadata:, init_forward_metadata_replay_cuda_graph_from_precomputed, init_forward_metadata_replay_cuda_graph
  - `python/sglang/jit_kernel/fused_metadata_copy.py` added +316/-0 (316 lines); hunk: +"""; 符号: _jit_fused_metadata_copy_module, _jit_fused_metadata_copy_multi_module, fused_metadata_copy_cuda, parameters
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py`；patch 关键词为 cache, mla, flash, attention, cuda, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/tests/test_fused_metadata_copy.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_metadata_copy.cuh`, `python/sglang/srt/layers/attention/nsa/nsa_mtp_verification.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17633 - [AMD] CI - enable deepseekv3.2 on MI325-8gpu and merge perf/accuracy test suites into stage-b suites

- 链接：https://github.com/sgl-project/sglang/pull/17633
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-28；作者 `yctseng0211`。
- 代码 diff 已读范围：`9` 个文件，`+88/-230`；代码面：attention/backend, MoE/router, kernel, tests/benchmarks；关键词：test, attention, moe, triton。
- 代码 diff 细节：
  - `.github/workflows/pr-test-amd.yml` modified +47/-206 (253 lines); hunk: jobs:; jobs:
  - `scripts/ci/utils/slash_command_handler.py` modified +5/-8 (13 lines); hunk: def handle_rerun_stage(; 符号: handle_rerun_stage
  - `test/registered/eval/test_moe_eval_accuracy_large.py` modified +12/-0 (12 lines); hunk: python -m unittest test_moe_eval_accuracy_large.TestMoEEvalAccuracyLarge.test_mmlu; DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,; 符号: TestMoEEvalAccuracyLarge, setUpClass, setUpClass
  - `test/registered/amd/test_deepseek_v32_basic.py` modified +2/-6 (8 lines); hunk: write_github_step_summary,; def test_bs_1_speed(self):; 符号: test_bs_1_speed
  - `test/registered/amd/test_kimi_k2_instruct.py` modified +6/-2 (8 lines); hunk: from sglang.test.test_utils import (; def test_bs_1_speed(self):; 符号: test_bs_1_speed
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py`；patch 关键词为 test, attention, moe, triton。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `.github/workflows/pr-test-amd.yml`, `scripts/ci/utils/slash_command_handler.py`, `test/registered/eval/test_moe_eval_accuracy_large.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17647 - [Perf] opt nsa backend init forward metada

- 链接：https://github.com/sgl-project/sglang/pull/17647
- 状态/时间：`closed`，created 2026-01-23, closed 2026-03-01；作者 `Baidu-AIAK`。
- 代码 diff 已读范围：`2` 个文件，`+88/-64`；代码面：attention/backend；关键词：attention, cuda, kv, triton, cache, mla, spec, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +27/-64 (91 lines); hunk: pad_nsa_cache_seqlens,; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_forward_metadata, init_forward_metadata_replay_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunk: def pad_sequence_with_mask(; 符号: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`；patch 关键词为 attention, cuda, kv, triton, cache, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17657 - [DeepSeek] Update tests and document for DeepSeek V3.2 NVFP4 checkpoint

- 链接：https://github.com/sgl-project/sglang/pull/17657
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-27；作者 `Fridge003`。
- 代码 diff 已读范围：`3` 个文件，`+88/-0`；代码面：quantization, tests/benchmarks, docs/config；关键词：fp4, fp8, flash, kv, moe, quant, test, attention, cache, config。
- 代码 diff 细节：
  - `test/srt/test_deepseek_v32_fp4_4gpu.py` added +79/-0 (79 lines); hunk: +import unittest; 符号: TestDeepseekV32FP4, setUpClass, tearDownClass, test_a_gsm8k
  - `docs/basic_usage/deepseek_v32.md` modified +8/-0 (8 lines); hunk: python3 -m sglang.launch_server \
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunk: TestFile("test_deepseek_v3_fp4_4gpu.py", 1500),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py`；patch 关键词为 fp4, fp8, flash, kv, moe, quant。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_deepseek_v32_fp4_4gpu.py`, `docs/basic_usage/deepseek_v32.md`, `test/srt/run_suite.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17662 - [DeepSeek-V3.2] Fix TRT-LLM NSA in target_verify/draft_extend

- 链接：https://github.com/sgl-project/sglang/pull/17662
- 状态/时间：`merged`，created 2026-01-23, merged 2026-01-25；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+18/-1`；代码面：attention/backend；关键词：attention, cache, flash, kv, lora, mla, test, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +18/-1 (19 lines); hunk: def forward_extend(; def forward_decode(; 符号: forward_extend, forward_decode, _forward_trtllm, _forward_trtllm
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, flash, kv, lora, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17688 - [DSv32] Overlap indexer qk projection and activation quant

- 链接：https://github.com/sgl-project/sglang/pull/17688
- 状态/时间：`merged`，created 2026-01-25, merged 2026-01-28；作者 `zianglih`。
- 代码 diff 已读范围：`1` 个文件，`+4/-4`；代码面：attention/backend；关键词：attention, cuda, fp8, lora, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +4/-4 (8 lines); hunk: def forward_cuda(; 符号: forward_cuda
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda, fp8, lora, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17783 - [AMD] Update dsv3.2 AMD GPU docs and unify ROCm TileLang build

- 链接：https://github.com/sgl-project/sglang/pull/17783
- 状态/时间：`merged`，created 2026-01-26, merged 2026-01-27；作者 `hubertlu-tw`。
- 代码 diff 已读范围：`2` 个文件，`+81/-88`；代码面：docs/config；关键词：config, doc, test, cache, cuda, kv, quant。
- 代码 diff 细节：
  - `docker/rocm.Dockerfile` modified +71/-87 (158 lines); hunk: # Usage (to build SGLang ROCm docker image):; ARG LLVM_COMMIT="6520ace8227ffe2728148d5f3b9872a870b0a560"
  - `docs/basic_usage/deepseek_v32.md` modified +10/-1 (11 lines); hunk: Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](h; python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3.2-Exp --tp 8 --
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md`；patch 关键词为 config, doc, test, cache, cuda, kv。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docker/rocm.Dockerfile`, `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17951 - Add tool call tests for DeepSeek V3.2 in nightly CI

- 链接：https://github.com/sgl-project/sglang/pull/17951
- 状态/时间：`merged`，created 2026-01-29, merged 2026-01-29；作者 `harvenstar`。
- 代码 diff 已读范围：`3` 个文件，`+363/-5`；代码面：model wrapper, scheduler/runtime, tests/benchmarks；关键词：test, spec, cuda, eagle, kv。
- 代码 diff 细节：
  - `python/sglang/test/tool_call_test_runner.py` added +320/-0 (320 lines); hunk: +import json; 符号: ToolCallTestParams:, ToolCallTestResult:, _call, _test_basic_format
  - `python/sglang/test/run_combined_tests.py` modified +32/-1 (33 lines); hunk: run_performance_test,; def run_combined_tests(; 符号: run_combined_tests, run_combined_tests, run_combined_tests, run_combined_tests
  - `test/registered/8-gpu-models/test_deepseek_v32.py` modified +11/-4 (15 lines); hunk: from sglang.test.performance_test_runner import PerformanceTestParams; '{"enable_multithread_load": true}',; 符号: test_deepseek_v32_all_variants, test_deepseek_v32_all_variants
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py`；patch 关键词为 test, spec, cuda, eagle, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/test/tool_call_test_runner.py`, `python/sglang/test/run_combined_tests.py`, `test/registered/8-gpu-models/test_deepseek_v32.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18094 - support deepseekv3.2-piecewise-cuda-graph

- 链接：https://github.com/sgl-project/sglang/pull/18094
- 状态/时间：`open`，created 2026-02-02；作者 `BJWang-ant`。
- 代码 diff 已读范围：`15` 个文件，`+243/-91`；代码面：model wrapper, attention/backend, MoE/router, kernel, scheduler/runtime, docs/config；关键词：cuda, kv, attention, moe, topk, config, deepep, cache, expert, mla。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +148/-48 (196 lines); hunk: MaybeTboDeepEPDispatcher,; prepare_input_dp_with_cp_dsa,; 符号: forward, forward_deepep, _post_combine_hook, __init__
  - `python/sglang/srt/layers/radix_attention.py` modified +19/-19 (38 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +12/-3 (15 lines); hunk: def forward_impl(self, hidden_states: torch.Tensor, topk_output: TopKOutput):; 符号: forward_impl, moe_forward_piecewise_cuda_graph_impl
  - `python/sglang/srt/layers/moe/topk.py` modified +12/-3 (15 lines); hunk: def is_power_of_two(n):; def biased_grouped_topk_gpu(; 符号: is_power_of_two, _mask_topk_ids_padded_region, _biased_grouped_topk_postprocess, biased_grouped_topk_gpu
  - `python/sglang/srt/layers/communicator.py` modified +7/-5 (12 lines); hunk: def prepare_attn(; 符号: prepare_attn, _tp_reduce_scatter
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；patch 关键词为 cuda, kv, attention, moe, topk, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/radix_attention.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18126 - Fix dsv32 encode_messages

- 链接：https://github.com/sgl-project/sglang/pull/18126
- 状态/时间：`merged`，created 2026-02-02, merged 2026-02-14；作者 `whybeyoung`。
- 代码 diff 已读范围：`2` 个文件，`+30/-5`；代码面：misc；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/parser/jinja_template_utils.py` modified +16/-5 (21 lines); hunk: def process_content_for_template_format(; def process_content_for_template_format(; 符号: process_content_for_template_format, format, process_content_for_template_format, process_content_for_template_format
  - `python/sglang/srt/entrypoints/openai/serving_chat.py` modified +14/-0 (14 lines); hunk: def _apply_jinja_template(; 符号: _apply_jinja_template
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py`；patch 关键词为 n/a。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/parser/jinja_template_utils.py`, `python/sglang/srt/entrypoints/openai/serving_chat.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18174 - [Bugfix] Catch errors when DeepSeek-V3.2 generates malformed JSON

- 链接：https://github.com/sgl-project/sglang/pull/18174
- 状态/时间：`merged`，created 2026-02-03, merged 2026-03-03；作者 `Muqi1029`。
- 代码 diff 已读范围：`1` 个文件，`+6/-3`；代码面：misc；关键词：kv。
- 代码 diff 细节：
  - `python/sglang/srt/function_call/deepseekv32_detector.py` modified +6/-3 (9 lines); hunk: def _parse_parameters_from_xml(; 符号: _parse_parameters_from_xml
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/function_call/deepseekv32_detector.py`；patch 关键词为 kv。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/function_call/deepseekv32_detector.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18280 - [DeepSeek v3.2][Bugfix] get_index_k_scale_buffer support cp

- 链接：https://github.com/sgl-project/sglang/pull/18280
- 状态/时间：`merged`，created 2026-02-05, merged 2026-03-17；作者 `xu-yfei`。
- 代码 diff 已读范围：`4` 个文件，`+22/-4`；代码面：attention/backend, kernel, tests/benchmarks；关键词：attention, cache, kv, topk, fp8, test, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +9/-3 (12 lines); hunk: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; def _get_topk_ragged(; 符号: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu, _get_topk_ragged
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +8/-0 (8 lines); hunk: class NSAMetadata:; def get_cu_seqlens_k(self) -> torch.Tensor:; 符号: NSAMetadata:, get_cu_seqlens_k, get_indexer_kvcache_range, get_indexer_seq_len
  - `test/registered/kernels/test_nsa_indexer.py` modified +4/-0 (4 lines); hunk: def get_indexer_seq_len_cpu(self) -> torch.Tensor:; 符号: get_indexer_seq_len_cpu, get_indexer_seq_len, get_nsa_extend_len_cpu
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +1/-1 (2 lines); hunk: def _get_k_and_s_triton(; 符号: _get_k_and_s_triton
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py`；patch 关键词为 attention, cache, kv, topk, fp8, test。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `test/registered/kernels/test_nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18297 - Deepseekv32 compatibility with transformers v5

- 链接：https://github.com/sgl-project/sglang/pull/18297
- 状态/时间：`merged`，created 2026-02-05, merged 2026-02-10；作者 `JustinTong0323`。
- 代码 diff 已读范围：`5` 个文件，`+33/-19`；代码面：model wrapper, attention/backend, docs/config；关键词：attention, config, mla, quant, cuda, kv, lora, moe, spec, topk。
- 代码 diff 细节：
  - `python/sglang/srt/configs/model_config.py` modified +13/-14 (27 lines); hunk: def _derive_model_shapes(self):; 符号: _derive_model_shapes
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-4 (17 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +4/-0 (4 lines); hunk: def __init__(; def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):; 符号: __init__, set_nsa_prefill_impl
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +2/-1 (3 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunk: class Envs:; 符号: Envs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, config, mla, quant, cuda, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18319 - [AMD] Use `tilelang` as default NSA attention backend dispatch on AMD Instinct

- 链接：https://github.com/sgl-project/sglang/pull/18319
- 状态/时间：`merged`，created 2026-02-05, merged 2026-02-27；作者 `fxmarty-amd`。
- 代码 diff 已读范围：`2` 个文件，`+7/-2`；代码面：attention/backend；关键词：attention, cache, flash, fp8, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +4/-1 (5 lines); hunk: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; 符号: _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunk: def forward_extend(; 符号: forward_extend, forward_decode
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, flash, fp8, kv, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18389 - Nsa trtllm mla sparse fp8 support with Deepseek v3.2 NVFP4

- 链接：https://github.com/sgl-project/sglang/pull/18389
- 状态/时间：`merged`，created 2026-02-07, merged 2026-02-16；作者 `rainj-me`。
- 代码 diff 已读范围：`10` 个文件，`+352/-183`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：fp8, kv, mla, attention, cache, quant, lora, flash, cuda, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +172/-66 (238 lines); hunk: nsa_cp_round_robin_split_q_seqs,; def __init__(; 符号: __init__, forward_extend, forward_extend, forward_extend
  - `python/sglang/srt/layers/attention/trtllm_mla_backend.py` modified +13/-97 (110 lines); hunk: FlashInferMLAMultiStepDraftBackend,; logger = logging.getLogger(__name__); 符号: init_forward_metadata, init_mha_chunk_metadata, quantize_and_rope_for_fp8, pad_draft_extend_query
  - `python/sglang/srt/layers/attention/utils.py` modified +99/-0 (99 lines); hunk: import triton; def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:; 符号: create_flashinfer_kv_indices_triton, canonicalize_stride, mla_quantize_and_rope_for_fp8, concat_mla_absorb_q_general
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +40/-0 (40 lines); hunk: def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):; def init_memory_pool(self: ModelRunner, total_gpu_memory: int):; 符号: handle_max_mamba_cache, calculate_mla_kv_cache_dim, set_num_tokens_hybrid_swa, init_memory_pool
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +14/-18 (32 lines); hunk: def __init__(; def set_kv_buffer(; 符号: __init__, set_kv_buffer, set_mla_kv_buffer, set_kv_buffer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py`；patch 关键词为 fp8, kv, mla, attention, cache, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/trtllm_mla_backend.py`, `python/sglang/srt/layers/attention/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18526 - [AMD] Enable cudagraph for aiter nsa backend and add aiter impl for nsa pr…

- 链接：https://github.com/sgl-project/sglang/pull/18526
- 状态/时间：`merged`，created 2026-02-10, merged 2026-02-27；作者 `wufann`。
- 代码 diff 已读范围：`2` 个文件，`+130/-3`；代码面：attention/backend, kernel；关键词：attention, kv, topk, triton, cache, eagle, flash, mla, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +70/-3 (73 lines); hunk: _is_hip = is_hip(); def __init__(; 符号: __init__, forward_extend, _forward_aiter, _forward_aiter
  - `python/sglang/srt/layers/attention/nsa/triton_kernel.py` modified +60/-0 (60 lines); hunk: def act_quant(; 符号: act_quant, _get_valid_kv_indices_kernel, get_valid_kv_indices
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py`；patch 关键词为 attention, kv, topk, triton, cache, eagle。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/nsa/triton_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18542 - fix: fixed aux hidden state index out of range when using eagle3 with nsa cp

- 链接：https://github.com/sgl-project/sglang/pull/18542
- 状态/时间：`open`，created 2026-02-10；作者 `echo-rain`。
- 代码 diff 已读范围：`1` 个文件，`+9/-1`；代码面：model wrapper；关键词：cuda, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +9/-1 (10 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 cuda, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18553 - Fix Bug on dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/18553
- 状态/时间：`merged`，created 2026-02-10, merged 2026-02-11；作者 `BourneSun0527`。
- 代码 diff 已读范围：`2` 个文件，`+16/-8`；代码面：attention/backend；关键词：attention, eagle, fp8, quant, scheduler, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-6 (18 lines); hunk: import torch; def forward_npu(; 符号: forward_npu, forward_npu
  - `python/sglang/srt/managers/overlap_utils.py` modified +4/-2 (6 lines); hunk: import torch; 符号: _resolve_future_token_ids
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py`；patch 关键词为 attention, eagle, fp8, quant, scheduler, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/managers/overlap_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18613 - [V3.2] Change default CP token split method to `--round-robin-split`

- 链接：https://github.com/sgl-project/sglang/pull/18613
- 状态/时间：`merged`，created 2026-02-11, merged 2026-02-11；作者 `Fridge003`。
- 代码 diff 已读范围：`3` 个文件，`+5/-5`；代码面：docs/config；关键词：cache, kv, attention, doc, fp8, moe, spec。
- 代码 diff 细节：
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunk: DeepSeek-V3.2-Speciale:; Example:
  - `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunk: class ServerArgs:; def add_cli_args(parser: argparse.ArgumentParser):; 符号: ServerArgs:, add_cli_args
  - `docs/advanced_features/server_arguments.md` modified +1/-1 (2 lines); hunk: Please consult the documentation below and [server_args.py](https://github.com/s
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md`；patch 关键词为 cache, kv, attention, doc, fp8, moe。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`, `docs/advanced_features/server_arguments.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18876 - Add DeepSeek3.2 and GlmMoeDsa into moe tune

- 链接：https://github.com/sgl-project/sglang/pull/18876
- 状态/时间：`merged`，created 2026-02-16, merged 2026-03-10；作者 `yuan-luo`。
- 代码 diff 已读范围：`1` 个文件，`+4/-0`；代码面：MoE/router, kernel, tests/benchmarks；关键词：benchmark, config, expert, kv, moe, triton。
- 代码 diff 细节：
  - `benchmark/kernels/fused_moe_triton/common_utils.py` modified +4/-0 (4 lines); hunk: def get_model_config(; def get_model_config(; 符号: get_model_config, get_model_config
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/kernels/fused_moe_triton/common_utils.py`；patch 关键词为 benchmark, config, expert, kv, moe, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/kernels/fused_moe_triton/common_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18931 - Fix NSA FP8 KV cache path for both-trtllm MHA one-shot

- 链接：https://github.com/sgl-project/sglang/pull/18931
- 状态/时间：`merged`，created 2026-02-17, merged 2026-02-20；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+8/-1`；代码面：model wrapper, attention/backend；关键词：attention, cache, fp8, kv, mla, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` modified +8/-1 (9 lines); hunk: def forward_normal_prepare(; 符号: forward_normal_prepare
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`；patch 关键词为 attention, cache, fp8, kv, mla, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18978 - [AMD] Fix mi35x dsv32 mtp nightly

- 链接：https://github.com/sgl-project/sglang/pull/18978
- 状态/时间：`merged`，created 2026-02-18, merged 2026-02-19；作者 `bingxche`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：attention/backend；关键词：attention。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-1 (2 lines); hunk: # Control whether to use fused metadata copy kernel (default: enabled)
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19016 - [FIX] NSA backend page_table overflow in speculative decoding target_verify

- 链接：https://github.com/sgl-project/sglang/pull/19016
- 状态/时间：`merged`，created 2026-02-19, merged 2026-03-06；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+3/-1`；代码面：attention/backend；关键词：attention, cuda, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +3/-1 (4 lines); hunk: def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):; 符号: init_cuda_graph_state
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cuda, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19041 - [DSv32] [GLM5] Improve Model Quality by Avoiding FP32 Precision Loss in `weights_proj`

- 链接：https://github.com/sgl-project/sglang/pull/19041
- 状态/时间：`merged`，created 2026-02-20, merged 2026-02-22；作者 `zianglih`。
- 代码 diff 已读范围：`4` 个文件，`+48/-9`；代码面：attention/backend, kernel, tests/benchmarks；关键词：config, cuda, attention, scheduler, test, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +17/-7 (24 lines); hunk: def _with_real_sm_count(self):; 符号: _with_real_sm_count, _project_and_scale_head_gates, _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates
  - `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py` modified +15/-0 (15 lines); hunk: class DeepGemmKernelType(IntEnum):; def create(kernel_type: DeepGemmKernelType, **kwargs):; 符号: DeepGemmKernelType, create, get_memory_requirement, execute
  - `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` modified +14/-0 (14 lines); hunk: def gemm_nt_f8f8bf16(; 符号: gemm_nt_f8f8bf16, gemm_nt_bf16bf16f32, update_deep_gemm_config
  - `test/registered/kernels/test_nsa_indexer.py` modified +2/-2 (4 lines); hunk: from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler; "context_len": 2048,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py`；patch 关键词为 config, cuda, attention, scheduler, test, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/layers/deep_gemm_wrapper/compile_utils.py`, `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19062 - [DSv32] Fix MTP and CP compatibility

- 链接：https://github.com/sgl-project/sglang/pull/19062
- 状态/时间：`merged`，created 2026-02-20, merged 2026-02-21；作者 `vladnosiv`。
- 代码 diff 已读范围：`1` 个文件，`+5/-5`；代码面：model wrapper；关键词：attention, config。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +5/-5 (10 lines); hunk: prepare_input_dp_with_cp_dsa,; def __init__(; 符号: __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`；patch 关键词为 attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #19148 - [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache

- 链接：https://github.com/sgl-project/sglang/pull/19148
- 状态/时间：`merged`，created 2026-02-22, merged 2026-02-26；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+307/-21`；代码面：attention/backend, kernel, scheduler/runtime；关键词：fp8, cache, cuda, kv, quant, attention, topk。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh` added +124/-0 (124 lines); hunk: +#include <sgl_kernel/tensor.h>; 符号: FusedStoreCacheParam, void, int64_t, FusedStoreCacheIndexerKernel
  - `python/sglang/jit_kernel/fused_store_index_cache.py` added +103/-0 (103 lines); hunk: +"""; 符号: _jit_nsa_fused_store_module, can_use_nsa_fused_store, fused_store_index_k_cache
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +79/-21 (100 lines); hunk: import torch; def _forward_cuda_k_only(; 符号: _forward_cuda_k_only, forward_indexer, _store_index_k_cache, forward_cuda
  - `python/sglang/jit_kernel/utils.py` modified +1/-0 (1 lines); hunk: def __str__(self) -> str:; 符号: __str__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 fp8, cache, cuda, kv, quant, attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/csrc/nsa/fused_store_index_cache.cuh`, `python/sglang/jit_kernel/fused_store_index_cache.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19319 - [deepseekv3.2] fix get_k_and_s_triton kernel for 128K seqlen case bug

- 链接：https://github.com/sgl-project/sglang/pull/19319
- 状态/时间：`merged`，created 2026-02-25, merged 2026-03-11；作者 `BJWang-ant`。
- 代码 diff 已读范围：`5` 个文件，`+380/-81`；代码面：attention/backend, kernel, scheduler/runtime, tests/benchmarks；关键词：triton, attention, kv, cache, cuda, fp8, test, topk。
- 代码 diff 细节：
  - `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py` added +191/-0 (191 lines); hunk: +import torch; 符号: golden_torch_gen, get_k_and_s_triton
  - `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py` modified +105/-48 (153 lines); hunk: def execute(cls, *args, **kwargs):; def _get_s_triton_kernel(; 符号: execute, triton, _get_s_triton_kernel, _get_k_and_s_triton
  - `test/manual/layers/attention/nsa/test_index_buf_accessor.py` modified +46/-9 (55 lines); hunk: def test_get_k_and_s_correctness(; def test_get_k_and_s_correctness(; 符号: test_get_k_and_s_correctness, test_get_k_and_s_correctness, test_get_k_and_s_sequential_pages, test_get_k_and_s_sequential_pages
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +29/-22 (51 lines); hunk: def _should_chunk_mqa_logits(; def _get_topk_ragged(; 符号: _should_chunk_mqa_logits, _get_topk_ragged, _get_topk_ragged, _get_topk_ragged
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +9/-2 (11 lines); hunk: def get_index_k_scale_continuous(; def get_index_k_scale_buffer(; 符号: get_index_k_scale_continuous, get_index_k_scale_buffer, get_index_k_scale_buffer, set_index_k_scale_buffer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py`；patch 关键词为 triton, attention, kv, cache, cuda, fp8。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/manual/layers/attention/nsa/test_get_k_scale_triton_kernel.py`, `python/sglang/srt/layers/attention/nsa/index_buf_accessor.py`, `test/manual/layers/attention/nsa/test_index_buf_accessor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19367 - Fix NSA CP positions mismatch in eagle NextN model

- 链接：https://github.com/sgl-project/sglang/pull/19367
- 状态/时间：`merged`，created 2026-02-25, merged 2026-02-26；作者 `alisonshao`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：model wrapper；关键词：expert。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-0 (2 lines); hunk: can_cp_split,; def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`；patch 关键词为 expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19428 - [Feature] add feature mla_ag_after_qlora for dsv3.2

- 链接：https://github.com/sgl-project/sglang/pull/19428
- 状态/时间：`merged`，created 2026-02-26, merged 2026-03-02；作者 `JiaruiChang5268`。
- 代码 diff 已读范围：`5` 个文件，`+101/-9`；代码面：model wrapper, attention/backend；关键词：lora, attention, kv, quant, cache, config, cuda, fp8, mla, topk。
- 代码 diff 细节：
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +35/-3 (38 lines); hunk: import torch; def forward_mha_prepare_npu(; 符号: forward_mha_prepare_npu, forward_mha_prepare_npu, forward_mla_prepare_npu, forward_mla_prepare_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +33/-2 (35 lines); hunk: fused_store_index_k_cache,; is_nsa_enable_prefill_cp,; 符号: forward_npu, forward_npu, forward_npu, forward_npu
  - `python/sglang/srt/models/deepseek_v2.py` modified +26/-3 (29 lines); hunk: def forward(; def forward_prepare(; 符号: forward, forward_prepare, forward_prepare, __init__
  - `python/sglang/srt/layers/communicator.py` modified +5/-1 (6 lines); hunk: from sglang.srt.distributed.device_communicators.pynccl_allocator import (; _use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip(); 符号: __init__, init_context, get_fn
  - `python/sglang/srt/environ.py` modified +2/-0 (2 lines); hunk: class Envs:; 符号: Envs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 lora, attention, kv, quant, cache, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19536 - [Perf] Optimize NSA backend metadata under MTP

- 链接：https://github.com/sgl-project/sglang/pull/19536
- 状态/时间：`merged`，created 2026-02-28, merged 2026-03-01；作者 `b8zhong`。
- 代码 diff 已读范围：`2` 个文件，`+85/-64`；代码面：attention/backend；关键词：attention, cuda, kv, triton, cache, fp8, mla, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +24/-64 (88 lines); hunk: from sglang.srt.layers.attention.utils import (; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: init_forward_metadata, init_forward_metadata, init_forward_metadata_replay_cuda_graph, init_forward_metadata_replay_cuda_graph
  - `python/sglang/srt/layers/attention/utils.py` modified +61/-0 (61 lines); hunk: def pad_sequence_with_mask(; 符号: pad_sequence_with_mask, seqlens_expand_kernel, seqlens_expand_triton
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py`；patch 关键词为 attention, cuda, kv, triton, cache, fp8。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/srt/layers/attention/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19945 - [AMD] Tilelang sparse fwd for dsv32 mi355/mi300

- 链接：https://github.com/sgl-project/sglang/pull/19945
- 状态/时间：`merged`，created 2026-03-05, merged 2026-03-24；作者 `1am9trash`。
- 代码 diff 已读范围：`1` 个文件，`+141/-95`；代码面：attention/backend, kernel；关键词：attention, kv, mla, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +141/-95 (236 lines); hunk: def sparse_mla_fwd_decode_partial(; def sparse_mla_fwd_decode_partial(; 符号: sparse_mla_fwd_decode_partial, sparse_mla_fwd_decode_partial, main, main
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`；patch 关键词为 attention, kv, mla, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19987 - [AMD] Fix nightly GLM-5 failures: Fix NSA indexer tensor aliasing on ROCm during CUDA graph capture

- 链接：https://github.com/sgl-project/sglang/pull/19987
- 状态/时间：`closed`，created 2026-03-05, closed 2026-03-05；作者 `michaelzhang-ai`。
- 代码 diff 已读范围：`1` 个文件，`+7/-0`；代码面：attention/backend；关键词：attention, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +7/-0 (7 lines); hunk: def _get_q_k_bf16(; 符号: _get_q_k_bf16
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20086 - [V32/GLM5] Change default setting of V32 nvfp4 on TP4

- 链接：https://github.com/sgl-project/sglang/pull/20086
- 状态/时间：`merged`，created 2026-03-07, merged 2026-03-07；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+15/-6`；代码面：misc；关键词：attention, cache, flash, fp8, kv, mla。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +15/-6 (21 lines); hunk: def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; 符号: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 attention, cache, flash, fp8, kv, mla。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20326 - [Doc] Add DSA/NSA attention backend to support matrix

- 链接：https://github.com/sgl-project/sglang/pull/20326
- 状态/时间：`merged`，created 2026-03-11, merged 2026-03-11；作者 `mvanhorn`。
- 代码 diff 已读范围：`1` 个文件，`+19/-1`；代码面：attention/backend, docs/config；关键词：attention, cache, cuda, doc, flash, fp8, kv, mla, spec。
- 代码 diff 细节：
  - `docs/advanced_features/attention_backend.md` modified +19/-1 (20 lines); hunk: Multimodal attention is selected by `--mm-attention-backend`. The "MultiModal" c; GDN models are hybrid: the full-attention layers still require a standard `--a
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/advanced_features/attention_backend.md`；patch 关键词为 attention, cache, cuda, doc, flash, fp8。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/advanced_features/attention_backend.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20343 - HiSparse for Sparse Attention

- 链接：https://github.com/sgl-project/sglang/pull/20343
- 状态/时间：`merged`，created 2026-03-11, merged 2026-03-23；作者 `xiezhq-hermann`。
- 代码 diff 已读范围：`20` 个文件，`+1692/-59`；代码面：attention/backend, kernel, multimodal/processor, scheduler/runtime；关键词：cache, cuda, kv, mla, config, attention, lora, quant, scheduler, test。
- 代码 diff 细节：
  - `python/sglang/srt/managers/hisparse_coordinator.py` added +596/-0 (596 lines); hunk: +# to be combined with the sparse coordinator class and sparse algorithm family; 符号: and, HiSparseAct, HiSparseCoordinator:, __init__
  - `python/sglang/jit_kernel/csrc/hisparse.cuh` added +390/-0 (390 lines); hunk: +#include <sgl_kernel/tensor.h>; 符号: int, int32_t, int32_t, void
  - `python/sglang/srt/mem_cache/hisparse_memory_pool.py` added +341/-0 (341 lines); hunk: +# mapping on device memory, host memory and memory allocator; 符号: HiSparseNSATokenToKVPool, __init__, register_mapping, translate_loc_to_hisparse_device
  - `python/sglang/srt/managers/scheduler.py` modified +85/-23 (108 lines); hunk: from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config; ); 符号: __init__, init_cache_with_memory_pool, handle_batch_embedding_request, stash_chunked_request
  - `python/sglang/jit_kernel/hisparse.py` added +88/-0 (88 lines); hunk: +from __future__ import annotations; 符号: _jit_sparse_module, load_cache_to_device_buffer_mla
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py`；patch 关键词为 cache, cuda, kv, mla, config, attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/jit_kernel/csrc/hisparse.cuh`, `python/sglang/srt/mem_cache/hisparse_memory_pool.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20438 - [Perf] Overlap NSA-CP key all-gather with query computation for DeepSeek-V3.2

- 链接：https://github.com/sgl-project/sglang/pull/20438
- 状态/时间：`merged`，created 2026-03-12, merged 2026-03-24；作者 `Baidu-AIAK`。
- 代码 diff 已读范围：`1` 个文件，`+19/-0`；代码面：attention/backend；关键词：attention, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +19/-0 (19 lines); hunk: def _get_q_k_bf16(; 符号: _get_q_k_bf16
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20492 - [BugFix] bug fix for DeepSeek eagle3 in Attn-DP mode

- 链接：https://github.com/sgl-project/sglang/pull/20492
- 状态/时间：`merged`，created 2026-03-13, merged 2026-03-19；作者 `khalil2ji3mp6`。
- 代码 diff 已读范围：`1` 个文件，`+2/-2`；代码面：model wrapper；关键词：attention, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_v2.py` modified +2/-2 (4 lines); hunk: get_moe_expert_parallel_world_size,; from sglang.srt.layers.dp_attention import (; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_v2.py`；patch 关键词为 attention, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_v2.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20534 - Transfer FP8 K/K_scale for CP indexer prefill gather

- 链接：https://github.com/sgl-project/sglang/pull/20534
- 状态/时间：`open`，created 2026-03-13；作者 `huangzhilin-hzl`。
- 代码 diff 已读范围：`1` 个文件，`+35/-8`；代码面：attention/backend；关键词：attention, cache, cuda, fp8, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +35/-8 (43 lines); hunk: def _get_q_k_bf16(; def _store_index_k_cache(; 符号: _get_q_k_bf16, _get_k_bf16, _store_index_k_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cache, cuda, fp8, kv, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20606 - FIX: (NSA) Compute topk_indices_offset when NSA prefill flashmla_sparse is used with FP8 KV cache

- 链接：https://github.com/sgl-project/sglang/pull/20606
- 状态/时间：`merged`，created 2026-03-15, merged 2026-03-26；作者 `JackChuang`。
- 代码 diff 已读范围：`1` 个文件，`+20/-4`；代码面：attention/backend；关键词：attention, cache, flash, kv, mla, spec, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +20/-4 (24 lines); hunk: def topk_transform(; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: topk_transform, init_forward_metadata, forward_extend, set_nsa_prefill_impl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 attention, cache, flash, kv, mla, spec。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20840 - [AMD] Fix dpsk-v32 accuracy issue on mi355

- 链接：https://github.com/sgl-project/sglang/pull/20840
- 状态/时间：`merged`，created 2026-03-18, merged 2026-03-18；作者 `1am9trash`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：quantization；关键词：fp8, quant, triton。
- 代码 diff 细节：
  - `python/sglang/srt/layers/quantization/fp8_utils.py` modified +1/-0 (1 lines); hunk: def use_aiter_triton_gemm_w8a8_tuned_gfx950(n: int, k: int) -> bool:; 符号: use_aiter_triton_gemm_w8a8_tuned_gfx950
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/quantization/fp8_utils.py`；patch 关键词为 fp8, quant, triton。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/quantization/fp8_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20984 - Fix DeepSeek V32 FP4 test

- 链接：https://github.com/sgl-project/sglang/pull/20984
- 状态/时间：`merged`，created 2026-03-20, merged 2026-03-20；作者 `Fridge003`。
- 代码 diff 已读范围：`3` 个文件，`+20/-1`；代码面：quantization, tests/benchmarks；关键词：test, fp4, quant, cache。
- 代码 diff 细节：
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +9/-0 (9 lines); hunk: +import os; def setUpClass(cls):; 符号: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +9/-0 (9 lines); hunk: +import os; def setUpClass(cls):; 符号: setUpClass, setUpClass
  - `python/sglang/test/test_utils.py` modified +2/-1 (3 lines); hunk: def popen_launch_server(; def popen_launch_server(; 符号: popen_launch_server, popen_launch_server
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`；patch 关键词为 test, fp4, quant, cache。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21003 - Revert "Fix DeepSeek V32 FP4 test"

- 链接：https://github.com/sgl-project/sglang/pull/21003
- 状态/时间：`merged`，created 2026-03-20, merged 2026-03-20；作者 `merrymercy`。
- 代码 diff 已读范围：`3` 个文件，`+1/-20`；代码面：quantization, tests/benchmarks；关键词：test, fp4, quant, cache。
- 代码 diff 细节：
  - `test/registered/quant/test_deepseek_v32_fp4_4gpu.py` modified +0/-9 (9 lines); hunk: -import os; def setUpClass(cls):; 符号: setUpClass, setUpClass
  - `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py` modified +0/-9 (9 lines); hunk: -import os; def setUpClass(cls):; 符号: setUpClass, setUpClass
  - `python/sglang/test/test_utils.py` modified +1/-2 (3 lines); hunk: def popen_launch_server(; def popen_launch_server(; 符号: popen_launch_server, popen_launch_server
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py`；patch 关键词为 test, fp4, quant, cache。影响判断：量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/quant/test_deepseek_v32_fp4_4gpu.py`, `test/registered/quant/test_deepseek_v32_fp4_mtp_4gpu.py`, `python/sglang/test/test_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21192 - Fix CP in-seq-split method for DeepSeek V32 and update related tests

- 链接：https://github.com/sgl-project/sglang/pull/21192
- 状态/时间：`merged`，created 2026-03-23, merged 2026-03-23；作者 `Fridge003`。
- 代码 diff 已读范围：`7` 个文件，`+162/-97`；代码面：model wrapper, tests/benchmarks；关键词：test, cuda, kv, spec, attention, config, eagle, topk, benchmark, cache。
- 代码 diff 细节：
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` added +157/-0 (157 lines); hunk: +import unittest; 符号: TestDeepseekV32CPInSeqSplit, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py` removed +0/-92 (92 lines); hunk: -import unittest; 符号: TestDeepseekV32CPSingleNode, for, test_deepseek_v32_cp_variants
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
  - `test/manual/nightly/test_deepseek_v32_perf.py` modified +1/-1 (2 lines); hunk: from sglang.test.nightly_utils import NightlyBenchmarkRunner
  - `test/registered/8-gpu-models/test_deepseek_v32_basic.py` modified +1/-1 (2 lines); hunk: register_cuda_ci(est_time=360, suite="stage-c-test-8-gpu-h200"); 符号: TestDeepseekV32DP
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py`；patch 关键词为 test, cuda, kv, spec, attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/cp/test_deepseek_v32_cp_single_node.py`, `test/registered/8-gpu-models/test_deepseek_v32_cp_single_node.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21249 - Support allreduce fusion with cp

- 链接：https://github.com/sgl-project/sglang/pull/21249
- 状态/时间：`merged`，created 2026-03-24, merged 2026-04-20；作者 `Shunkangz`。
- 代码 diff 已读范围：`4` 个文件，`+201/-27`；代码面：attention/backend, scheduler/runtime；关键词：attention, flash, cuda, moe, config, expert, spec。
- 代码 diff 细节：
  - `python/sglang/srt/layers/flashinfer_comm_fusion.py` modified +178/-22 (200 lines); hunk: from sglang.srt.distributed import (; logger = logging.getLogger(__name__); 符号: _always_disable_fabric, _FixedTorchDistBackend, __init__, bcast
  - `python/sglang/srt/model_executor/model_runner.py` modified +22/-0 (22 lines); hunk: def initialize(self, pre_model_load_memory: float):; def kernel_warmup(self):; 符号: initialize, kernel_warmup, _pre_initialize_flashinfer_allreduce_workspace, _should_run_flashinfer_autotune
  - `python/sglang/srt/layers/communicator.py` modified +1/-4 (5 lines); hunk: def apply_flashinfer_allreduce_fusion(batch_size: int):; def prepare_attn(; 符号: apply_flashinfer_allreduce_fusion, prepare_attn
  - `python/sglang/srt/server_args.py` modified +0/-1 (1 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py`；patch 关键词为 attention, flash, cuda, moe, config, expert。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/flashinfer_comm_fusion.py`, `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/layers/communicator.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21259 - [HiCache & HybridModel] mooncake backend support DSA & mamba model

- 链接：https://github.com/sgl-project/sglang/pull/21259
- 状态/时间：`merged`，created 2026-03-24, merged 2026-04-14；作者 `huangtingwei9988`。
- 代码 diff 已读范围：`8` 个文件，`+760/-232`；代码面：scheduler/runtime, tests/benchmarks；关键词：cache, kv, mla, config, attention, cuda, quant, test。
- 代码 diff 细节：
  - `python/sglang/srt/mem_cache/memory_pool_host.py` modified +230/-68 (298 lines); hunk: logger = logging.getLogger(__name__); def __init__(; 符号: synchronized, __init__, __init__, __init__
  - `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` modified +180/-36 (216 lines); hunk: import time; HiCacheStorage,; 符号: __init__, register_mem_pool_host, register_mem_host_pool_v2, _tag_keys
  - `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` added +212/-0 (212 lines); hunk: +from __future__ import annotations; 符号: build_nsa_hybrid_stack, layer_mapper, build_mamba_hybrid_stack, kv_layer_mapper
  - `python/sglang/srt/mem_cache/hiradix_cache.py` modified +83/-32 (115 lines); hunk: MatchPrefixParams,; from sglang.srt.mem_cache.memory_pool_host import (; 符号: __init__, __init__, __init__, attach_storage_backend
  - `python/sglang/srt/mem_cache/hi_mamba_radix_cache.py` modified +9/-88 (97 lines); hunk: ); def __init__(self, params: CacheInitParams, server_args: ServerArgs):; 符号: __init__, kv_layer_mapper, mamba_layer_mapper, mamba_layer_mapper
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`；patch 关键词为 cache, kv, mla, config, attention, cuda。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/mem_cache/memory_pool_host.py`, `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`, `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21337 - Workaround of DSA performance drop on B200 + DP

- 链接：https://github.com/sgl-project/sglang/pull/21337
- 状态/时间：`merged`，created 2026-03-24, merged 2026-03-25；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+11/-5`；代码面：misc；关键词：cache, cuda, fp4, fp8, kv, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +11/-5 (16 lines); hunk: def _generate_piecewise_cuda_graph_tokens(self):; def _set_default_nsa_kv_cache_dtype(self, major: int) -> str:; 符号: _generate_piecewise_cuda_graph_tokens, _set_default_nsa_kv_cache_dtype, _set_default_nsa_kv_cache_dtype, _set_default_nsa_kv_cache_dtype
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 cache, cuda, fp4, fp8, kv, quant。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21405 - Enable IndexCache for DeepSeek V3.2

- 链接：https://github.com/sgl-project/sglang/pull/21405
- 状态/时间：`merged`，created 2026-03-25, merged 2026-04-05；作者 `jinyouzhi`。
- 代码 diff 已读范围：`4` 个文件，`+196/-20`；代码面：model wrapper, attention/backend, scheduler/runtime, tests/benchmarks；关键词：topk, cache, cuda, kv, attention, config, lora, mla, expert, fp4。
- 代码 diff 细节：
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunk: +import unittest; 符号: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/deepseek_v2.py` modified +51/-6 (57 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, op_prepare
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunk: def forward_absorb_prepare(; def forward_absorb_prepare(; 符号: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
  - `python/sglang/srt/models/deepseek_nextn.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`；patch 关键词为 topk, cache, cuda, kv, attention, config。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21468 - [NPU] Update DeepSeek-V3.2 model deployment instructions in documentation

- 链接：https://github.com/sgl-project/sglang/pull/21468
- 状态/时间：`merged`，created 2026-03-26, merged 2026-03-30；作者 `MichelleWu351`。
- 代码 diff 已读范围：`1` 个文件，`+96/-148`；代码面：docs/config；关键词：attention, cache, config, cuda, deepep, doc, eagle, mla, moe, quant。
- 代码 diff 细节：
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +96/-148 (244 lines); hunk: you encounter issues or have any questions, please [open an issue](https://githu; We tested it based on the `RANDOM` dataset.
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/platforms/ascend/ascend_npu_best_practice.md`；patch 关键词为 attention, cache, config, cuda, deepep, doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/platforms/ascend/ascend_npu_best_practice.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21511 - [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend

- 链接：https://github.com/sgl-project/sglang/pull/21511
- 状态/时间：`merged`，created 2026-03-27, merged 2026-04-03；作者 `1am9trash`。
- 代码 diff 已读范围：`6` 个文件，`+517/-77`；代码面：model wrapper, attention/backend, kernel, scheduler/runtime；关键词：cache, kv, fp8, mla, attention, quant, lora, topk, triton, config。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py` modified +307/-42 (349 lines); hunk: +from functools import lru_cache; def fast_round_scale(amax, fp8_max_inv):; 符号: fast_round_scale, _pick_inner_iter, act_quant_kernel, main
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +79/-18 (97 lines); hunk: def forward_absorb_prepare(; def forward_absorb_core(; 符号: forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla, _skip_rope_for_nsa_tilelang_fused
  - `python/sglang/srt/mem_cache/utils.py` modified +87/-0 (87 lines); hunk: def set_mla_kv_buffer_triton(; 符号: set_mla_kv_buffer_triton, set_mla_kv_buffer_fp8_quant_kernel, set_mla_kv_buffer_triton_fp8_quant, set_mla_kv_scale_buffer_kernel
  - `python/sglang/srt/mem_cache/memory_pool.py` modified +32/-15 (47 lines); hunk: quantize_k_cache,; _is_cpu = is_cpu(); 符号: get_tensor_size_bytes, set_mla_kv_buffer
  - `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` modified +11/-1 (12 lines); hunk: from sglang.srt.utils.common import (; def __post_init__(self):; 符号: __post_init__, ModelRunnerKVCacheMixin:, calculate_mla_kv_cache_dim
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py`；patch 关键词为 cache, kv, fp8, mla, attention, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`, `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`, `python/sglang/srt/mem_cache/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21585 - [CI] Move v32 cp test to deepep running suite

- 链接：https://github.com/sgl-project/sglang/pull/21585
- 状态/时间：`merged`，created 2026-03-28, merged 2026-03-28；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+1/-1`；代码面：tests/benchmarks；关键词：cuda, deepep, test。
- 代码 diff 细节：
  - `test/registered/cp/test_deepseek_v32_cp_single_node.py` modified +1/-1 (2 lines); hunk: write_github_step_summary,
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/cp/test_deepseek_v32_cp_single_node.py`；patch 关键词为 cuda, deepep, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/cp/test_deepseek_v32_cp_single_node.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #21623 - [Test] Add unit tests for encoding_dsv32.py

- 链接：https://github.com/sgl-project/sglang/pull/21623
- 状态/时间：`open`，created 2026-03-29；作者 `dondetir`。
- 代码 diff 已读范围：`1` 个文件，`+871/-0`；代码面：tests/benchmarks；关键词：config, spec, test。
- 代码 diff 细节：
  - `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` added +871/-0 (871 lines); hunk: +"""Unit tests for encoding_dsv32.py — no server, no model loading.; 符号: _make_tool, _make_tool_call, _parse_dsml_args, TestEncodeArgumentsToDsml
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py`；patch 关键词为 config, spec, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/unit/entrypoints/openai/test_encoding_dsv32.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21783 - [DSA] Support trtllm sparse mla kernel for prefill batches

- 链接：https://github.com/sgl-project/sglang/pull/21783
- 状态/时间：`merged`，created 2026-03-31, merged 2026-04-01；作者 `Fridge003`。
- 代码 diff 已读范围：`3` 个文件，`+12/-14`；代码面：attention/backend, tests/benchmarks；关键词：attention, cache, flash, mla, topk, kv, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +0/-11 (11 lines); hunk: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; def _handle_model_specific_adjustments(self):; 符号: _set_default_nsa_backends, _handle_model_specific_adjustments
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +9/-0 (9 lines); hunk: def forward_extend(; def _forward_trtllm(; 符号: forward_extend, _forward_trtllm, _forward_trtllm
  - `python/sglang/test/run_eval.py` modified +3/-3 (6 lines); hunk: def get_thinking_kwargs(args):; def run_eval(args):; 符号: get_thinking_kwargs, run_eval
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py`；patch 关键词为 attention, cache, flash, mla, topk, kv。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`, `python/sglang/test/run_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21914 - [DSA] Set trtllm kernels as default for Blackwell

- 链接：https://github.com/sgl-project/sglang/pull/21914
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-02；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+2/-7`；代码面：misc；关键词：cache, fp4, fp8, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +2/-7 (9 lines); hunk: def _set_default_nsa_kv_cache_dtype(self, major: int, quantization: str) -> str:; def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; 符号: _set_default_nsa_kv_cache_dtype, _set_default_nsa_backends
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 cache, fp4, fp8, kv, quant。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21932 - [HiSparse] Optimize the scheduling of decode backup.

- 链接：https://github.com/sgl-project/sglang/pull/21932
- 状态/时间：`merged`，created 2026-04-02, merged 2026-04-07；作者 `huangtingwei9988`。
- 代码 diff 已读范围：`2` 个文件，`+42/-9`；代码面：scheduler/runtime；关键词：cuda, topk。
- 代码 diff 细节：
  - `python/sglang/srt/managers/hisparse_coordinator.py` modified +36/-9 (45 lines); hunk: def __init__(; def _eager_backup_previous_token(; 符号: __init__, _eager_backup_previous_token, _eager_backup_previous_token, wait_for_pending_backup
  - `python/sglang/srt/model_executor/model_runner.py` modified +6/-0 (6 lines); hunk: def _forward_raw(; 符号: _forward_raw
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py`；patch 关键词为 cuda, topk。影响判断：scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/managers/hisparse_coordinator.py`, `python/sglang/srt/model_executor/model_runner.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- 链接：https://github.com/sgl-project/sglang/pull/22003
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-20；作者 `Shunkangz`。
- 代码 diff 已读范围：`8` 个文件，`+276/-25`；代码面：model wrapper, attention/backend, MoE/router, tests/benchmarks；关键词：moe, attention, cuda, config, expert, flash, fp4, spec, test。
- 代码 diff 细节：
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunk: get_dp_global_num_tokens,; class ScatterMode(Enum):; 符号: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0 (55 lines); hunk: def test_gsm8k(self):; 符号: test_gsm8k, TestQwen330BCP, setUpClass, tearDownClass
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunk: get_attn_tensor_model_parallel_rank,; def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):; 符号: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/distributed/parallel_state.py` modified +13/-7 (20 lines); hunk: def initialize_model_parallel(; def initialize_model_parallel(; 符号: initialize_model_parallel, initialize_model_parallel, destroy_model_parallel, destroy_model_parallel
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunk: def __init__(; 符号: __init__, get_input_embeddings
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py`；patch 关键词为 moe, attention, cuda, config, expert, flash。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/communicator.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `python/sglang/srt/layers/dp_attention.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22065 - [HiSparse]: Optimize server args checking-HiSparse is temporarily only available for DSA models.

- 链接：https://github.com/sgl-project/sglang/pull/22065
- 状态/时间：`merged`，created 2026-04-03, merged 2026-04-03；作者 `hzh0425`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：misc；关键词：attention, cache, config。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +8/-0 (8 lines); hunk: def check_server_args(self):; 符号: check_server_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 attention, cache, config。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #22179 - [Doc] Fix and improve DeepSeek V3.2/GLM-5 documentation

- 链接：https://github.com/sgl-project/sglang/pull/22179
- 状态/时间：`merged`，created 2026-04-06, merged 2026-04-06；作者 `mmangkad`。
- 代码 diff 已读范围：`1` 个文件，`+11/-12`；代码面：docs/config；关键词：attention, benchmark, cache, config, deepep, doc, eagle, flash, fp8, kv。
- 代码 diff 细节：
  - `docs/basic_usage/deepseek_v32.md` modified +11/-12 (23 lines); hunk: DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attent
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/basic_usage/deepseek_v32.md`；patch 关键词为 attention, benchmark, cache, config, deepep, doc。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22232 - Reduce unnecessary kernels and copies in the NSA indexer

- 链接：https://github.com/sgl-project/sglang/pull/22232
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-07；作者 `1am9trash`。
- 代码 diff 已读范围：`1` 个文件，`+13/-5`；代码面：attention/backend；关键词：attention, cuda, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +13/-5 (18 lines); hunk: def _weights_proj_bf16_in_fp32_out(self, x: torch.Tensor) -> torch.Tensor:; def _get_q_k_bf16(; 符号: _weights_proj_bf16_in_fp32_out, _project_and_scale_head_gates, _get_logits_head_gate, _get_q_k_bf16
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22238 - [HiSparse]: Add readme docs for HiSparse Feature

- 链接：https://github.com/sgl-project/sglang/pull/22238
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-07；作者 `hzh0425`。
- 代码 diff 已读范围：`2` 个文件，`+117/-0`；代码面：docs/config；关键词：attention, config, cuda, doc, kv, cache, flash, mla, topk。
- 代码 diff 细节：
  - `docs/advanced_features/hisparse_guide.md` added +111/-0 (111 lines); hunk: +# HiSparse: Hierarchical Sparse Attention
  - `docs/basic_usage/deepseek_v32.md` modified +6/-0 (6 lines); hunk: python -m sglang.launch_server \
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md`；patch 关键词为 attention, config, cuda, doc, kv, cache。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `docs/advanced_features/hisparse_guide.md`, `docs/basic_usage/deepseek_v32.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22258 - [AMD][HIP] NSA: bf16 passthrough from RMSNorm to eliminate FP8 dequantization

- 链接：https://github.com/sgl-project/sglang/pull/22258
- 状态/时间：`merged`，created 2026-04-07, merged 2026-04-10；作者 `Jacob0226`。
- 代码 diff 已读范围：`2` 个文件，`+68/-25`；代码面：attention/backend；关键词：attention, cuda, fp8, quant, lora。
- 代码 diff 细节：
  - `python/sglang/srt/layers/communicator.py` modified +38/-18 (56 lines); hunk: def __init__(self):; def prepare_attn(; 符号: __init__, init_context, prepare_attn, prepare_attn
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +30/-7 (37 lines); hunk: import contextlib; ceil_align,; 符号: _with_real_sm_count, _weights_proj_bf16_in_fp32_out, _weights_proj_bf16_in_fp32_out, _weights_proj_bf16_in_fp32_out
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda, fp8, quant, lora。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22372 - [DSA] Hopper FP8 FlashMLA KV padding

- 链接：https://github.com/sgl-project/sglang/pull/22372
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-12；作者 `mmangkad`。
- 代码 diff 已读范围：`3` 个文件，`+43/-8`；代码面：attention/backend, docs/config；关键词：cache, flash, fp8, kv, mla, attention, doc, topk, benchmark, config。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +39/-4 (43 lines); hunk: def __init__(; def init_forward_metadata(self, forward_batch: ForwardBatch):; 符号: __init__, init_forward_metadata, init_forward_metadata, _forward_flashmla_kv
  - `docs/basic_usage/deepseek_v32.md` modified +2/-2 (4 lines); hunk: To serve GLM-5, just replace the `--model` argument with `zai-org/GLM-5-FP8`.
  - `python/sglang/srt/server_args.py` modified +2/-2 (4 lines); hunk: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; 符号: _set_default_nsa_backends
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py`；patch 关键词为 cache, flash, fp8, kv, mla, attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa_backend.py`, `docs/basic_usage/deepseek_v32.md`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22390 - [DSA] Enable all reduce fusion for DSA models

- 链接：https://github.com/sgl-project/sglang/pull/22390
- 状态/时间：`merged`，created 2026-04-08, merged 2026-04-09；作者 `Fridge003`。
- 代码 diff 已读范围：`1` 个文件，`+2/-0`；代码面：misc；关键词：kv, moe, spec。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +2/-0 (2 lines); hunk: def _handle_model_specific_adjustments(self):; 符号: _handle_model_specific_adjustments
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`；patch 关键词为 kv, moe, spec。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22424 - [AMD] Use aiter CK layernorm2d for LayerNorm to reduce NSA indexer kernel launches

- 链接：https://github.com/sgl-project/sglang/pull/22424
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-09；作者 `1am9trash`。
- 代码 diff 已读范围：`2` 个文件，`+27/-3`；代码面：attention/backend；关键词：attention, cuda, fp8, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/layernorm.py` modified +15/-1 (16 lines); hunk: gemma_rmsnorm,; def forward_hip(; 符号: forward_hip, forward_npu
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +12/-2 (14 lines); hunk: from sglang.srt.layers.layernorm import LayerNorm; def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cuda, fp8, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/layernorm.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22425 - [HiSparse]: Add HiSpares-DSA Model's nightly CI

- 链接：https://github.com/sgl-project/sglang/pull/22425
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-09；作者 `hzh0425`。
- 代码 diff 已读范围：`1` 个文件，`+84/-0`；代码面：model wrapper, tests/benchmarks；关键词：attention, cache, config, cuda, flash, fp8, kv, mla, test。
- 代码 diff 细节：
  - `test/registered/8-gpu-models/test_dsa_models_hisparse.py` added +84/-0 (84 lines); hunk: +import unittest; 符号: TestGLM5DPHiSparse, setUpClass, tearDownClass, test_a_gsm8k
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/8-gpu-models/test_dsa_models_hisparse.py`；patch 关键词为 attention, cache, config, cuda, flash, fp8。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/8-gpu-models/test_dsa_models_hisparse.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22430 - [Fix] Fix several bugs on DSA models

- 链接：https://github.com/sgl-project/sglang/pull/22430
- 状态/时间：`merged`，created 2026-04-09, merged 2026-04-09；作者 `Fridge003`。
- 代码 diff 已读范围：`2` 个文件，`+5/-5`；代码面：attention/backend；关键词：cache, flash, fp8, kv, mla, attention, topk。
- 代码 diff 细节：
  - `python/sglang/srt/server_args.py` modified +4/-2 (6 lines); hunk: def _set_default_nsa_backends(self, kv_cache_dtype: str, major: int) -> str:; 符号: _set_default_nsa_backends
  - `python/sglang/srt/layers/attention/nsa_backend.py` modified +1/-3 (4 lines); hunk: def get_topk_transform_method(; 符号: get_topk_transform_method
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py`；patch 关键词为 cache, flash, fp8, kv, mla, attention。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/server_args.py`, `python/sglang/srt/layers/attention/nsa_backend.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22792 - nsa indexer: use aiter indexer_k_quant_and_cache

- 链接：https://github.com/sgl-project/sglang/pull/22792
- 状态/时间：`open`，created 2026-04-14；作者 `almaslof`。
- 代码 diff 已读范围：`32` 个文件，`+701/-165`；代码面：model wrapper, attention/backend, quantization, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：attention, config, cache, cuda, kv, test, topk, lora, mla, quant。
- 代码 diff 细节：
  - `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py` modified +208/-44 (252 lines); hunk: def generate_dashboard(; def generate_dashboard(; 符号: generate_dashboard, generate_dashboard, _chart_label, _chart_label
  - `python/tools/get_version_tag.py` added +171/-0 (171 lines); hunk: +#!/usr/bin/env python3; 符号: parse_version_tuple, run_git, get_exact_version_tag, get_latest_version_tag_describe
  - `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` added +117/-0 (117 lines); hunk: +import unittest; 符号: TestDeepseekV32IndexTopkPattern, setUpClass, tearDownClass, test_a_gsm8k
  - `python/sglang/srt/models/deepseek_v2.py` modified +53/-9 (62 lines); hunk: make_layers,; if _is_cuda:; 符号: forward, __init__, __init__, __init__
  - `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` modified +27/-13 (40 lines); hunk: def forward_absorb_prepare(; def forward_absorb_prepare(; 符号: forward_absorb_prepare, forward_absorb_prepare, forward_absorb_core, _fuse_rope_for_trtllm_mla
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`；patch 关键词为 attention, config, cache, cuda, kv, test。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `scripts/ci/utils/diffusion/generate_diffusion_dashboard.py`, `python/tools/get_version_tag.py`, `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22850 - [AMD] Reduce NSA indexer kernels (weights_proj, k-cache store kernel fusion)

- 链接：https://github.com/sgl-project/sglang/pull/22850
- 状态/时间：`merged`，created 2026-04-15, merged 2026-04-19；作者 `1am9trash`。
- 代码 diff 已读范围：`1` 个文件，`+24/-5`；代码面：attention/backend；关键词：attention, cache, cuda, fp8, kv, quant。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +24/-5 (29 lines); hunk: from sglang.srt.environ import envs; _is_npu = is_npu(); 符号: __init__, _weights_proj_bf16_in_fp32_out, _store_index_k_cache
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cache, cuda, fp8, kv, quant。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22914 - [Refactor] Deduplicate NSA utils.py into cp_utils.py for context parallel

- 链接：https://github.com/sgl-project/sglang/pull/22914
- 状态/时间：`merged`，created 2026-04-16, merged 2026-04-20；作者 `Fridge003`。
- 代码 diff 已读范围：`8` 个文件，`+148/-402`；代码面：model wrapper, attention/backend, scheduler/runtime；关键词：attention, cache, kv, config, cuda, expert, fp8, moe, quant, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/utils.py` modified +2/-353 (355 lines); hunk: -# temp NSA debugging environ; def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):; 符号: pad_nsa_cache_seqlens, NSAContextParallelMetadata:, can_cp_split, can_nsa_cp_split
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +103/-12 (115 lines); hunk: import torch; def can_cp_split(seq_len: int, cp_size: int, forward_batch):; 符号: can_cp_split, cp_split_and_rebuild_data, cp_split_and_rebuild_data, cp_split_and_rebuild_position
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +20/-17 (37 lines); hunk: from sglang.srt.distributed.parallel_state import get_pp_group; def _get_q_k_bf16(; 符号: _get_q_k_bf16, _get_q_k_bf16, forward_cuda, forward_npu
  - `python/sglang/srt/models/deepseek_nextn.py` modified +11/-7 (18 lines); hunk: from sglang.srt.environ import envs; from sglang.srt.layers.logits_processor import LogitsProcessor; 符号: forward
  - `python/sglang/srt/models/deepseek_v2.py` modified +11/-7 (18 lines); hunk: from sglang.srt.layers.amx_utils import PackWeightMethod; from sglang.srt.layers.radix_attention import RadixAttention; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`；patch 关键词为 attention, cache, kv, config, cuda, expert。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/utils.py`, `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #23219 - [AMD] Enable MTP for GLM-5-mxfp4 model

- 链接：https://github.com/sgl-project/sglang/pull/23219
- 状态/时间：`merged`，created 2026-04-20, merged 2026-04-20；作者 `1am9trash`。
- 代码 diff 已读范围：`1` 个文件，`+41/-15`；代码面：model wrapper；关键词：attention, config, fp8, processor, quant, spec。
- 代码 diff 细节：
  - `python/sglang/srt/models/deepseek_nextn.py` modified +41/-15 (56 lines); hunk: is_dp_attention_enabled,; def __init__(; 符号: __init__, forward, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/deepseek_nextn.py`；patch 关键词为 attention, config, fp8, processor, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/deepseek_nextn.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23268 - 【NPU】【bugfix】accuracy fix when enable both nsa cp and prefixcache

- 链接：https://github.com/sgl-project/sglang/pull/23268
- 状态/时间：`open`，created 2026-04-20；作者 `cen121212`。
- 代码 diff 已读范围：`2` 个文件，`+21/-5`；代码面：attention/backend；关键词：attention, kv, lora, mla。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +18/-4 (22 lines); hunk: def forward_npu(; 符号: forward_npu
  - `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` modified +3/-1 (4 lines); hunk: def forward_dsa_prepare_npu(; 符号: forward_dsa_prepare_npu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py`；patch 关键词为 attention, kv, lora, mla。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

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

### PR #23351 - Support piecewise CUDA graph with NSA

- 链接：https://github.com/sgl-project/sglang/pull/23351
- 状态/时间：`open`，created 2026-04-21；作者 `nvjullin`。
- 代码 diff 已读范围：`11` 个文件，`+302/-56`；代码面：attention/backend, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：cache, config, cuda, attention, quant, topk, fp8, moe, flash, fp4。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/nsa/nsa_indexer.py` modified +163/-34 (197 lines); hunk: can_use_nsa_fused_store,; DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0; 符号: k_cache_and_topk_result, _logits_head_gate_pcg_fake_impl, logits_head_gate_pcg, BaseIndexerMetadata
  - `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py` added +70/-0 (70 lines); hunk: +import unittest; 符号: TestPCGGlm5Fp4, setUpClass, tearDownClass, test_gsm8k
  - `python/sglang/srt/layers/layernorm.py` modified +20/-1 (21 lines); hunk: if _is_cuda or _is_xpu:; 符号: _layernorm_fake_impl, layernorm
  - `python/sglang/srt/server_args.py` modified +4/-16 (20 lines); hunk: def _handle_piecewise_cuda_graph(self):; def _handle_moe_kernel_config(self):; 符号: _handle_piecewise_cuda_graph, _handle_multi_item_scoring, _handle_moe_kernel_config, _handle_a2a_moe
  - `python/sglang/srt/layers/radix_attention.py` modified +14/-0 (14 lines); hunk: def unified_attention_with_output(; def unified_attention_with_output(; 符号: unified_attention_with_output, unified_attention_with_output
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py`；patch 关键词为 cache, config, cuda, attention, quant, topk。影响判断：attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`, `test/registered/piecewise_cuda_graph/test_pcg_glm5_fp4.py`, `python/sglang/srt/layers/layernorm.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：175；open PR 数：12。
- 仍需跟进的 open PR：[#11191](https://github.com/sgl-project/sglang/pull/11191), [#14332](https://github.com/sgl-project/sglang/pull/14332), [#14524](https://github.com/sgl-project/sglang/pull/14524), [#15322](https://github.com/sgl-project/sglang/pull/15322), [#18094](https://github.com/sgl-project/sglang/pull/18094), [#18542](https://github.com/sgl-project/sglang/pull/18542), [#20534](https://github.com/sgl-project/sglang/pull/20534), [#21623](https://github.com/sgl-project/sglang/pull/21623), [#22792](https://github.com/sgl-project/sglang/pull/22792), [#23268](https://github.com/sgl-project/sglang/pull/23268), [#23336](https://github.com/sgl-project/sglang/pull/23336), [#23351](https://github.com/sgl-project/sglang/pull/23351)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
