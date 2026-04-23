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
- `#22774`：MUSA backend 支持 DeepSeek V2/V3/R1-class layers。
- `#22851`：新增 `--nsa-topk-backend`，接入 FlashInfer 和 PyTorch top-k。
- `#22865`：扩展 sparsity framework 支持非 NSA sparse algorithms。
- `#22938`：MLA refactor 后恢复 MI300X DeepSeek MLA 路径。
- `#23195`：AWQ/compressed-tensors 下 DeepSeek MLA `.weight` 访问保护。
- `#23241`：3FS backend 支持 DSA/mamba。
- `#23257`：`DeepseekV2MoE` 在 CuteDSL EP + DP attention 下的 double-reduce。
- `#23336`：spec v2 adaptive speculative decoding。
- `#23351`：NSA piecewise CUDA graph。
