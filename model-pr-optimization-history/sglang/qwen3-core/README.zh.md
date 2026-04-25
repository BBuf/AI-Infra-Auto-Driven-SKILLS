# sglang Qwen3 Core 模型 PR 优化历史

## 文档口径

- 重做日期: 2026-04-25
- 源码基线: `sgl-project/sglang` 当前追溯 worktree commit `880599cd43`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` | 无直接 PR 号提交 |
| `docs_new/docs/basic_usage/qwen3.mdx` | 无直接 PR 号提交 |
| `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx` | 无直接 PR 号提交 |
| `python/sglang/srt/models/qwen3.py` | [#4693](https://github.com/sgl-project/sglang/pull/4693), [#6250](https://github.com/sgl-project/sglang/pull/6250), [#6990](https://github.com/sgl-project/sglang/pull/6990), [#7312](https://github.com/sgl-project/sglang/pull/7312), [#7681](https://github.com/sgl-project/sglang/pull/7681), [#7740](https://github.com/sgl-project/sglang/pull/7740), [#10574](https://github.com/sgl-project/sglang/pull/10574), [#15223](https://github.com/sgl-project/sglang/pull/15223), [#15390](https://github.com/sgl-project/sglang/pull/15390), [#16115](https://github.com/sgl-project/sglang/pull/16115), [#17535](https://github.com/sgl-project/sglang/pull/17535), [#19532](https://github.com/sgl-project/sglang/pull/19532), ... (14 total) |
| `python/sglang/srt/models/qwen3_moe.py` | [#4693](https://github.com/sgl-project/sglang/pull/4693), [#5917](https://github.com/sgl-project/sglang/pull/5917), [#6120](https://github.com/sgl-project/sglang/pull/6120), [#6250](https://github.com/sgl-project/sglang/pull/6250), [#6533](https://github.com/sgl-project/sglang/pull/6533), [#6598](https://github.com/sgl-project/sglang/pull/6598), [#6652](https://github.com/sgl-project/sglang/pull/6652), [#6709](https://github.com/sgl-project/sglang/pull/6709), [#6820](https://github.com/sgl-project/sglang/pull/6820), [#7740](https://github.com/sgl-project/sglang/pull/7740), [#8751](https://github.com/sgl-project/sglang/pull/8751), [#9973](https://github.com/sgl-project/sglang/pull/9973), ... (26 total) |
| `test/registered/lora/test_lora_qwen3.py` | 无直接 PR 号提交 |
| `test/srt/cpu/test_qwen3.py` | [#12330](https://github.com/sgl-project/sglang/pull/12330) |

## PR 覆盖总览

- git 追溯 PR 数: 34
- 原文档显式引用补充 PR 数: 54
- 当前文档总 PR 数: 88
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2025-04-18 | [#4693](https://github.com/sgl-project/sglang/pull/4693) | merged | [Model] Adding Qwen3 and Qwen3MoE | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-04-30 | [#5917](https://github.com/sgl-project/sglang/pull/5917) | merged | [qwen3] support qwen3 ep moe | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-16 | [#6121](https://github.com/sgl-project/sglang/pull/6121) | merged | feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088 | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py` |
| 2025-05-18 | [#6250](https://github.com/sgl-project/sglang/pull/6250) | merged | Add pipeline parallelism for Qwen2 and Qwen3 Model | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-22 | [#6120](https://github.com/sgl-project/sglang/pull/6120) | merged | Support qwen3 deepep | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-24 | [#6533](https://github.com/sgl-project/sglang/pull/6533) | merged | support eplb for qwen3 | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-25 | [#6546](https://github.com/sgl-project/sglang/pull/6546) | merged | added support for tied weights in qwen pipeline parallelism | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`, `test/srt/test_pp_single_node.py` |
| 2025-05-26 | [#6598](https://github.com/sgl-project/sglang/pull/6598) | merged | qwen3moe support two batch overlap | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-27 | [#6652](https://github.com/sgl-project/sglang/pull/6652) | merged | Fix qwen3 tbo/dp-lm-head | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-05-29 | [#6709](https://github.com/sgl-project/sglang/pull/6709) | merged | Fix PP for Qwen3 MoE | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-06-03 | [#6818](https://github.com/sgl-project/sglang/pull/6818) | merged | Fix wrong weight reference in dynamic EPLB | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py` |
| 2025-06-09 | [#6990](https://github.com/sgl-project/sglang/pull/6990) | merged | support qwen3 emebedding | `python/sglang/srt/models/qwen3.py` |
| 2025-06-10 | [#6964](https://github.com/sgl-project/sglang/pull/6964) | merged | Support both approximate and exact expert distribution collection | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/managers/expert_distribution.py` |
| 2025-06-29 | [#7580](https://github.com/sgl-project/sglang/pull/7580) | merged | Move files related to EPLB | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-03 | [#7681](https://github.com/sgl-project/sglang/pull/7681) | merged | support qwen3 dense model dp attention | `python/sglang/srt/models/qwen3.py` |
| 2025-07-03 | [#7740](https://github.com/sgl-project/sglang/pull/7740) | merged | [optimize] add two stream norm for qwen3 | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-03 | [#7723](https://github.com/sgl-project/sglang/pull/7723) | merged | [Bug] add flashinfer bool check for fusedmoe in Qwen moe models | `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-05 | [#7745](https://github.com/sgl-project/sglang/pull/7745) | merged | [feat] Support EAGLE3 for Qwen | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2025-07-05 | [#7222](https://github.com/sgl-project/sglang/pull/7222) | merged | DP Attention with Auto DeepEP Dispatch | `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-07-07 | [#7762](https://github.com/sgl-project/sglang/pull/7762) | merged | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py` |
| 2025-07-16 | [#7634](https://github.com/sgl-project/sglang/pull/7634) | merged | [Feature] Layer-wise Prefill | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py` |
| 2025-07-19 | [#7966](https://github.com/sgl-project/sglang/pull/7966) | merged | [1/N] MoE Refactor: refactor `select_experts` | `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-07-20 | [#7312](https://github.com/sgl-project/sglang/pull/7312) | merged | Add get_hidden_dim to qwen3.py for correct lora | `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-07-25 | [#8280](https://github.com/sgl-project/sglang/pull/8280) | merged | DP Enhancement | `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py` |
| 2025-07-27 | [#8036](https://github.com/sgl-project/sglang/pull/8036) | merged | [NVIDIA] Add Flashinfer MoE blockscale fp8 backend | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-07-28 | [#8421](https://github.com/sgl-project/sglang/pull/8421) | merged | [3/N] MoE Refactor: Simplify DeepEP Output | `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-07-29 | [#8448](https://github.com/sgl-project/sglang/pull/8448) | merged | Support EPLB in FusedMoE | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-08-01 | [#8450](https://github.com/sgl-project/sglang/pull/8450) | merged | [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE | `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py` |
| 2025-08-01 | [#8658](https://github.com/sgl-project/sglang/pull/8658) | merged | [5/N] MoE Refactor: Update MoE parallelism arguments | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py` |
| 2025-08-06 | [#8751](https://github.com/sgl-project/sglang/pull/8751) | merged | [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-08-06 | [#8753](https://github.com/sgl-project/sglang/pull/8753) | merged | [2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights | `python/sglang/srt/eplb/expert_location.py` |
| 2025-08-09 | [#8987](https://github.com/sgl-project/sglang/pull/8987) | merged | Fix incorrect default get_hidden_dim logic | `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/granite.py`, `python/sglang/srt/models/llama.py` |
| 2025-08-12 | [#9014](https://github.com/sgl-project/sglang/pull/9014) | merged | Fuse writing KV buffer into rope kernel (part 2: srt) | `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py` |
| 2025-08-13 | [#9147](https://github.com/sgl-project/sglang/pull/9147) | open | support Qwen3-MoE-w4afp8 | `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py` |
| 2025-08-14 | [#9101](https://github.com/sgl-project/sglang/pull/9101) | merged | Feature: support qwen and llama4 reducescatter for dp attention padding | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py` |
| 2025-09-02 | [#8118](https://github.com/sgl-project/sglang/pull/8118) | merged | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-03 | [#7912](https://github.com/sgl-project/sglang/pull/7912) | merged | Qwen FP8/NVFP4 ModelOPT Quantization support | `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py` |
| 2025-09-04 | [#9973](https://github.com/sgl-project/sglang/pull/9973) | merged | Optimize Qwen3-moe model by using flashinfer fused allreduce | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-09-15 | [#9338](https://github.com/sgl-project/sglang/pull/9338) | merged | Refactor TopK to ensure readability and extensibility | `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-09-23 | [#10574](https://github.com/sgl-project/sglang/pull/10574) | merged | [Ascend]optimize Qwen3 on Ascend | `python/sglang/srt/models/qwen3.py` |
| 2025-09-26 | [#10749](https://github.com/sgl-project/sglang/pull/10749) | merged | Fuse write kv buffer into rope for qwen3 moe & bailing moe | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-09-29 | [#10975](https://github.com/sgl-project/sglang/pull/10975) | merged | Use more general heuristics to set the default value of --mem-fraction-static | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py` |
| 2025-10-01 | [#10985](https://github.com/sgl-project/sglang/pull/10985) | merged | Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-10-16 | [#10911](https://github.com/sgl-project/sglang/pull/10911) | merged | model: qwen3-omni (thinker-only) | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-10-29 | [#12002](https://github.com/sgl-project/sglang/pull/12002) | merged | Eagle3 DP attention for Qwen3 MoE | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-11-13 | [#12543](https://github.com/sgl-project/sglang/pull/12543) | merged | Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` |
| 2025-11-18 | [#13489](https://github.com/sgl-project/sglang/pull/13489) | merged | Flashinfer TRTLLM-GEN-MoE + Qwen3 | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-11-25 | [#12078](https://github.com/sgl-project/sglang/pull/12078) | merged | [Ascend] qwen optimization | `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` |
| 2025-12-03 | [#12330](https://github.com/sgl-project/sglang/pull/12330) | merged | [CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next | `test/srt/cpu/test_qwen3.py` |
| 2025-12-05 | [#14093](https://github.com/sgl-project/sglang/pull/14093) | merged | Add fused FP8 KV cache write kernel for TRTLLM MHA backend | `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-12-07 | [#13998](https://github.com/sgl-project/sglang/pull/13998) | merged | [apply][2/2] Fused qk_norm_rope for Qwen3-MoE | `python/sglang/srt/models/qwen3_moe.py` |
| 2025-12-15 | [#11984](https://github.com/sgl-project/sglang/pull/11984) | closed | [Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix | `python/sglang/srt/layers/quantization/w4a4_int4.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| 2025-12-17 | [#15223](https://github.com/sgl-project/sglang/pull/15223) | merged | [bug fix][pp] fix qwen3 model load | `python/sglang/srt/models/qwen3.py` |
| 2025-12-24 | [#15390](https://github.com/sgl-project/sglang/pull/15390) | merged | [NPU]qwen3 pp bugfix | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2025-12-28 | [#15835](https://github.com/sgl-project/sglang/pull/15835) | merged | [Feature] JIT Fused QK norm + qk norm clean up | `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py` |
| 2026-01-08 | [#16115](https://github.com/sgl-project/sglang/pull/16115) | merged | [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-01-10 | [#13715](https://github.com/sgl-project/sglang/pull/13715) | merged | Fix EPLB + FP4 Quantization Compatibility Issue | `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py` |
| 2026-01-27 | [#15890](https://github.com/sgl-project/sglang/pull/15890) | merged | [PP] fix wrong weight logic for tie_word_embeddings model | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-01-28 | [#15904](https://github.com/sgl-project/sglang/pull/15904) | merged | [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-01-29 | [#15203](https://github.com/sgl-project/sglang/pull/15203) | merged | [NPU] support GPTQ quantization on npu | `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/linear.py` |
| 2026-02-03 | [#17535](https://github.com/sgl-project/sglang/pull/17535) | merged | Update weight rename check for Qwen3 Embeddings | `python/sglang/srt/models/qwen3.py` |
| 2026-02-08 | [#18189](https://github.com/sgl-project/sglang/pull/18189) | merged | [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-03 | [#19532](https://github.com/sgl-project/sglang/pull/19532) | merged | [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe | `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-08 | [#20127](https://github.com/sgl-project/sglang/pull/20127) | open | [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-03-12 | [#20474](https://github.com/sgl-project/sglang/pull/20474) | open | Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3 | `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `test/srt/xpu/test_qwen3.py` |
| 2026-03-13 | [#20520](https://github.com/sgl-project/sglang/pull/20520) | open | [NPU]TP Communications compression For Qwen3 models for NPU | `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-20 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-22 | [#18233](https://github.com/sgl-project/sglang/pull/18233) | merged | Support Qwen3 MoE context parallel | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-24 | [#21195](https://github.com/sgl-project/sglang/pull/21195) | merged | Enable the qwen3 test | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-25 | [#21412](https://github.com/sgl-project/sglang/pull/21412) | open | [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints | `python/sglang/srt/models/qwen3.py` |
| 2026-03-27 | [#19059](https://github.com/sgl-project/sglang/pull/19059) | merged | [jit_kernel] Add fused_qknorm_rope JIT kernel | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` |
| 2026-03-31 | [#21770](https://github.com/sgl-project/sglang/pull/21770) | open | [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon | `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py` |
| 2026-04-01 | [#21654](https://github.com/sgl-project/sglang/pull/21654) | merged | [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` |
| 2026-04-01 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | `python/sglang/srt/models/qwen3.py` |
| 2026-04-09 | [#22429](https://github.com/sgl-project/sglang/pull/22429) | merged | [NPU]add Qwen3-32b and Qwen3-8b low latency md | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22450](https://github.com/sgl-project/sglang/pull/22450) | open | [NPU] Add Qwen3-14B low latency doc | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22529](https://github.com/sgl-project/sglang/pull/22529) | open | [Model] Support sliding window attention for Qwen3 | `python/sglang/srt/models/qwen3.py` |
| 2026-04-11 | [#22446](https://github.com/sgl-project/sglang/pull/22446) | merged | [NPU] add qwen3-30b-a3b low latency example | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-13 | [#22674](https://github.com/sgl-project/sglang/pull/22674) | open | [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization | `python/sglang/srt/model_loader/loader.py` |
| 2026-04-13 | [#22687](https://github.com/sgl-project/sglang/pull/22687) | merged | [NPU]qwen3-8b and 32b md bugfix | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-14 | [#22739](https://github.com/sgl-project/sglang/pull/22739) | merged | Restore Qwen3 rope config fallback | `python/sglang/srt/models/qwen3.py` |
| 2026-04-15 | [#22837](https://github.com/sgl-project/sglang/pull/22837) | open | [Bug] Qwen3 reasoning detector silently swallows tool_call when is missing | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-04-20 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-04-21 | [#23372](https://github.com/sgl-project/sglang/pull/23372) | open | [NPU] Add CI tests for Speculative Decoding | `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` |
| 2026-04-21 | [#23397](https://github.com/sgl-project/sglang/pull/23397) | open | [alignment-sglang] PR3: Dense Deterministic Math | `python/sglang/srt/layers/on_policy_utils.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/models/qwen3.py` |
| 2026-04-22 | [#23434](https://github.com/sgl-project/sglang/pull/23434) | open | [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model | `python/sglang/srt/models/qwen3_classification.py` |

## 逐 PR diff 审计卡

### PR #4693 - [Model] Adding Qwen3 and Qwen3MoE

- 链接: https://github.com/sgl-project/sglang/pull/4693
- 状态/时间: merged / 2025-04-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `4db463b1ad6e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+780/-14，可读 patch 840 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Adding Qwen3 and Qwen3MoE」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`；PR 正文摘要: This PR provides initial support for the Qwen3 and Qwen3MoE. Once the model file becomes available, I will further refine the integration. ref https://github.com/huggingface/tra...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention，涉及 `Qwen3MoeSparseMoeBlock, __init__, forward`；`python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward，涉及 `Qwen3Attention, __init__, _apply_qk_norm`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention
  - `python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -0,0 +1,423 @@
+# Adapted from qwen2_moe.py
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
diff -- python/sglang/srt/models/qwen3.py
@@ -0,0 +1,335 @@
+# Adapted from qwen2.py
+from functools import partial
+from typing import Any, Dict, Iterable, Optional, Tuple
+import torch
+from torch import nn
+from sglang.srt.distributed import (
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` added +423/-0; `python/sglang/srt/models/qwen3.py` added +335/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #5917 - [qwen3] support qwen3 ep moe

- 链接: https://github.com/sgl-project/sglang/pull/5917
- 状态/时间: merged / 2025-04-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `e330f2b86cd2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+16/-6，可读 patch 86 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[qwen3] support qwen3 ep moe」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Performance Test in 8xH200 Server launch and benchmarking command line:。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunks: -40,6 +40,7; -48,6 +49,7; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunks: -40,6 +40,7; -48,6 +49,7; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -40,6 +40,7 @@
+from sglang.srt.layers.moe.ep_moe.layer import EPMoE
@@ -48,6 +49,7 @@
+from sglang.srt.managers.schedule_batch import global_server_args_dict
@@ -73,12 +75,13 @@ def __init__(
-        self.experts = FusedMoE(
+        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +8/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6121 - feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088

- 链接: https://github.com/sgl-project/sglang/pull/6121
- 状态/时间: merged / 2025-05-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+449/-70，可读 patch 756 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`；PR 正文摘要: This is the prerequisites of EP, which is introduced in PR #5917 As described in #6088, DP attention is not supported for Qwen MoE models, but #5917 introduces EP MoE for them....。
- 实现要点: `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunks: -16,6 +16,8; -28,14 +30,23; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunks: -17,12 +17,15; -32,14 +35,23; symbols: __init__, forward, _FFNInputMode，涉及 `__init__, forward, _FFNInputMode`；`python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunks: -142,16 +142,6 @@ def get_local_attention_dp_size():; symbols: get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size，涉及 `get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size`；`python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunks: -269,6 +269,7 @@ def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model...; symbols: _maybe_prepare_dp_attn_batch，涉及 `_maybe_prepare_dp_attn_batch`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunks: -16,6 +16,8; -28,14 +30,23; symbols: __init__, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunks: -17,12 +17,15; -32,14 +35,23; symbols: __init__, forward, _FFNInputMode
  - `python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunks: -142,16 +142,6 @@ def get_local_attention_dp_size():; symbols: get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size
  - `python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunks: -269,6 +269,7 @@ def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model...; symbols: _maybe_prepare_dp_attn_batch
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -16,6 +16,8 @@
+from dataclasses import dataclass
+from enum import Enum, auto
@@ -28,14 +30,23 @@
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_gather,
+    attn_tp_reduce_scatter,
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -17,12 +17,15 @@
+from dataclasses import dataclass
+from enum import Enum, auto
+from transformers.configuration_utils import PretrainedConfig
@@ -32,14 +35,23 @@
+from sglang.srt.layers.dp_attention import (
+    attn_tp_all_gather,
diff -- python/sglang/srt/layers/dp_attention.py
@@ -142,16 +142,6 @@ def get_local_attention_dp_size():
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +227/-32; `python/sglang/srt/models/qwen3_moe.py` modified +221/-28; `python/sglang/srt/layers/dp_attention.py` modified +0/-10; `python/sglang/bench_one_batch.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/bench_one_batch.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6250 - Add pipeline parallelism for Qwen2 and Qwen3 Model

- 链接: https://github.com/sgl-project/sglang/pull/6250
- 状态/时间: merged / 2025-05-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `11553c1a3727`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+340/-73，可读 patch 736 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add pipeline parallelism for Qwen2 and Qwen3 Model」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunks: -1,12 +1,14; -19,15 +21,18; symbols: Qwen3Attention, __init__, forward, start_layer，涉及 `Qwen3Attention, __init__, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunks: -17,6 +17,7; -28,6 +29,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, start_layer，涉及 `Qwen3MoeSparseMoeBlock, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunks: -1,12 +1,14; -19,15 +21,18; symbols: Qwen3Attention, __init__, forward, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunks: -17,6 +17,7; -28,6 +29,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, start_layer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -1,12 +1,14 @@
+import logging
+    get_pp_group,
@@ -19,15 +21,18 @@
+from sglang.srt.layers.utils import get_layer_id
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -17,6 +17,7 @@
+import logging
@@ -28,6 +29,7 @@
+    get_pp_group,
@@ -57,19 +59,22 @@
+from sglang.srt.layers.utils import get_layer_id
-from sglang.srt.model_executor.forward_batch_info import ForwardBatch
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +52/-10; `python/sglang/srt/models/qwen3_moe.py` modified +49/-10
- 验证与风险: diff 自带测试面 `test/srt/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6120 - Support qwen3 deepep

- 链接: https://github.com/sgl-project/sglang/pull/6120
- 状态/时间: merged / 2025-05-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `fc0e3b91744b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+125/-8，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support qwen3 deepep」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Support qwen3's deepep. For now, we've simply copied the deepep code from DS, and the accuracy test has passed.。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunks: -32,6 +32,7; -54,8 +55,10; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunks: -32,6 +32,7; -54,8 +55,10; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -32,6 +32,7 @@
+    parallel_state,
@@ -54,8 +55,10 @@
-from sglang.srt.layers.moe.ep_moe.layer import EPMoE
+from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
+from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
+from sglang.srt.layers.moe.topk import select_experts
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +121/-7
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6533 - support eplb for qwen3

- 链接: https://github.com/sgl-project/sglang/pull/6533
- 状态/时间: merged / 2025-05-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `e6f113569e51`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+46/-25，可读 patch 187 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support eplb for qwen3」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: support eplb for qwen3moe, ~~need merge #6120 first, then do other modification. (add ExpertLocationDispatchInfo)~~ simple test。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunks: -55,7 +55,7; -67,6 +67,8; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, forward_normal，涉及 `Qwen3MoeSparseMoeBlock, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunks: -55,7 +55,7; -67,6 +67,8; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -55,7 +55,7 @@
-from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, EPMoE
+from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
@@ -67,6 +67,8 @@
+from sglang.srt.managers.expert_location import ModelConfigForExpertLocation
+from sglang.srt.managers.expert_location_dispatch import ExpertLocationDispatchInfo
@@ -86,28 +88,25 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +39/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6546 - added support for tied weights in qwen pipeline parallelism

- 链接: https://github.com/sgl-project/sglang/pull/6546
- 状态/时间: merged / 2025-05-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+134/-20，可读 patch 205 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「added support for tied weights in qwen pipeline parallelism」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`, `test/srt/test_pp_single_node.py`；PR 正文摘要: Some Qwen variants have `tie_word_embeddings = true`, however, the current code only caters to case where its value is false. If this value is true, the last rank cannot tie wei...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunks: -21,7 +21,7; -249,15 +249,36 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunks: -386,15 +386,36 @@ def __init__(; -470,7 +491,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunks: -116,6 +116,62 @@ def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test，涉及 `test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass`；`.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ jobs:。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunks: -21,7 +21,7; -249,15 +249,36 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunks: -386,15 +386,36 @@ def __init__(; -470,7 +491,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunks: -116,6 +116,62 @@ def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ jobs:
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -21,7 +21,7 @@
-from sglang.srt.layers.utils import get_layer_id
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -249,15 +249,36 @@ def __init__(
-        if config.tie_word_embeddings:
-            self.lm_head = self.model.embed_tokens
+        # handle the lm head on different pp ranks
diff -- python/sglang/srt/models/qwen2.py
@@ -386,15 +386,36 @@ def __init__(
-        if config.tie_word_embeddings:
-            self.lm_head = self.model.embed_tokens
+        # handle the lm head on different pp ranks
+        if self.pp_group.is_last_rank:
+            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
+                self.lm_head = self.model.embed_tokens
diff -- test/srt/test_pp_single_node.py
@@ -116,6 +116,62 @@ def test_pp_consistency(self):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +39/-10; `python/sglang/srt/models/qwen2.py` modified +38/-9
  - tests: `test/srt/test_pp_single_node.py` modified +56/-0
  - ci: `.github/workflows/pr-test.yml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/srt/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6598 - qwen3moe support two batch overlap

- 链接: https://github.com/sgl-project/sglang/pull/6598
- 状态/时间: merged / 2025-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `f9bab3d59100`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+351/-28，可读 patch 515 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「qwen3moe support two batch overlap」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Support two batch overlap for Qwen3, need merge #6581 first. Current overlap strategy is not stable, maybe we need change during tests.。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunks: -68,6 +68,9; -79,6 +82,7; symbols: __init__, forward_deepep, op_gate, op_select_experts，涉及 `__init__, forward_deepep, op_gate`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunks: -68,6 +68,9; -79,6 +82,7; symbols: __init__, forward_deepep, op_gate, op_select_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -68,6 +68,9 @@
+from sglang.srt.managers.expert_distribution import (
+    get_global_expert_distribution_recorder,
+)
@@ -79,6 +82,7 @@
+from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher
@@ -137,7 +141,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +200/-11
- 验证与风险: diff 自带测试面 `test/srt/test_two_batch_overlap.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6652 - Fix qwen3 tbo/dp-lm-head

- 链接: https://github.com/sgl-project/sglang/pull/6652
- 状态/时间: merged / 2025-05-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `b18416fbf869`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+3/-1，可读 patch 25 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix qwen3 tbo/dp-lm-head」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: 1. fix qwen3 tbo parameter 2. add dp_lm_head for qwen2/qwen3。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -688,6 +688,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -688,6 +688,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -688,6 +688,7 @@ def __init__(
+            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/two_batch_overlap.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6709 - Fix PP for Qwen3 MoE

- 链接: https://github.com/sgl-project/sglang/pull/6709
- 状态/时间: merged / 2025-05-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `e06b07610597`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+60/-4，可读 patch 85 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix PP for Qwen3 MoE」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Try to fix qwen3_moe PP failure issue mentioned by @libratiger in https://github.com/sgl-project/sglang/pull/6533#issuecomment-2911923181 Traverse layers according to `range(sel...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -812,9 +812,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -812,9 +812,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -812,9 +812,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-            layer_id: layer.mlp.get_moe_weights()
-            for layer_id, layer in enumerate(self.model.layers)
-            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
+            layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
+            for layer_id in range(self.start_layer, self.end_layer)
+            if isinstance(self.model.layers[layer_id].mlp, Qwen3MoeSparseMoeBlock)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +3/-3
- 验证与风险: diff 自带测试面 `test/srt/test_pp_single_node.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6818 - Fix wrong weight reference in dynamic EPLB

- 链接: https://github.com/sgl-project/sglang/pull/6818
- 状态/时间: merged / 2025-06-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+27/-13，可读 patch 83 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix wrong weight reference in dynamic EPLB」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py`；PR 正文摘要: will need to test locally later。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunks: -91,6 +91,7; -1661,6 +1662,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights，涉及 `__init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion`；`python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -18,15 +18,10; -811,6 +806,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`；`python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -2257,3 +2257,16 @@ def support_triton(backend: str) -> bool:; symbols: support_triton, cpu_has_amx_support, LazyValue, __init__，涉及 `support_triton, cpu_has_amx_support, LazyValue`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunks: -91,6 +91,7; -1661,6 +1662,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -18,15 +18,10; -811,6 +806,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -2257,3 +2257,16 @@ def support_triton(backend: str) -> bool:; symbols: support_triton, cpu_has_amx_support, LazyValue, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -91,6 +91,7 @@
+    LazyValue,
@@ -1661,6 +1662,18 @@ def __init__(
+        self._routed_experts_weights_of_layer = LazyValue(
+            lambda: {
+                layer_id: layer.mlp.get_moe_weights()
+                for layer_id, layer in enumerate(self.model.layers)
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -18,15 +18,10 @@
-from dataclasses import dataclass
-from enum import Enum, auto
-from functools import partial
-import torch.nn.functional as F
-from transformers.configuration_utils import PretrainedConfig
@@ -811,6 +806,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
diff -- python/sglang/srt/utils.py
@@ -2257,3 +2257,16 @@ def support_triton(backend: str) -> bool:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-8; `python/sglang/srt/models/qwen3_moe.py` modified +1/-5; `python/sglang/srt/utils.py` modified +13/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #6990 - support qwen3 emebedding

- 链接: https://github.com/sgl-project/sglang/pull/6990
- 状态/时间: merged / 2025-06-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `451ffe74d907`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+3/-0，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3 emebedding」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: 6917 Support Qwen3 Embedding model. Add "model" prefix to model weight name. Test with `python3 -m unittest models/test_embedding_models.py`.。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunks: -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunks: -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if "Embedding" in self.config.name_or_path:
+                name = add_prefix(name, "model")
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +2/-0
- 验证与风险: diff 自带测试面 `test/srt/models/test_embedding_models.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #6964 - Support both approximate and exact expert distribution collection

- 链接: https://github.com/sgl-project/sglang/pull/6964
- 状态/时间: merged / 2025-06-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+101/-71，可读 patch 257 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support both approximate and exact expert distribution collection」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/managers/expert_distribution.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunks: -456,22 +456,25 @@ def op_select_experts(self, state):; symbols: op_select_experts，涉及 `op_select_experts`；`python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunks: -255,17 +255,20 @@ def op_select_experts(self, state):; symbols: op_select_experts，涉及 `op_select_experts`；`python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunks: -264,15 +264,23 @@ def init_new(; -347,7 +355,9 @@ def on_forward_pass_start(self, forward_batch: ForwardBatch):; symbols: init_new, __init__, on_forward_pass_start, on_select_experts，涉及 `init_new, __init__, on_forward_pass_start`；`python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -182,7 +182,7 @@ class ServerArgs:; symbols: ServerArgs，涉及 `ServerArgs`。
- 代码 diff 细节:
  - `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunks: -456,22 +456,25 @@ def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunks: -255,17 +255,20 @@ def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunks: -264,15 +264,23 @@ def init_new(; -347,7 +355,9 @@ def on_forward_pass_start(self, forward_batch: ForwardBatch):; symbols: init_new, __init__, on_forward_pass_start, on_select_experts
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -182,7 +182,7 @@ class ServerArgs:; symbols: ServerArgs
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -456,22 +456,25 @@ def op_select_experts(self, state):
-            state.topk_weights_local, state.topk_idx_local = select_experts(
-                hidden_states=hidden_states,
-                router_logits=router_logits,
-                top_k=self.top_k,
-                use_grouped_topk=True,
-                renormalize=self.renormalize,
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -255,17 +255,20 @@ def op_select_experts(self, state):
-            state.topk_weights_local, state.topk_idx_local = select_experts(
-                hidden_states=hidden_states,
-                router_logits=router_logits,
-                top_k=self.top_k,
-                use_grouped_topk=False,
-                renormalize=self.renormalize,
diff -- python/sglang/srt/managers/expert_distribution.py
@@ -264,15 +264,23 @@ def init_new(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +19/-16; `python/sglang/srt/models/qwen3_moe.py` modified +14/-11; `python/sglang/srt/managers/expert_distribution.py` modified +67/-43; `python/sglang/srt/server_args.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7580 - Move files related to EPLB

- 链接: https://github.com/sgl-project/sglang/pull/7580
- 状态/时间: merged / 2025-06-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 22 个文件，+42/-54，可读 patch 289 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Move files related to EPLB」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；PR 正文摘要: suggested by @merrymercy。
- 实现要点: `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunks: -39,6 +39,19; -54,18 +67,6；`python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunks: -31,6 +31,11; -64,11 +69,6；`python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -77,11 +80,6；`python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -63,11 +66,6。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunks: -39,6 +39,19; -54,18 +67,6
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunks: -31,6 +31,11; -64,11 +69,6
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -77,11 +80,6
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -63,11 +66,6
  - `python/sglang/srt/layers/moe/topk.py` modified +3/-3 (6 lines); hunks: -18,12 +18,12
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -39,6 +39,19 @@
+from sglang.srt.eplb.eplb_manager import EPLBManager
+from sglang.srt.eplb.expert_distribution import (
+    ExpertDistributionRecorder,
+    get_global_expert_distribution_recorder,
+    set_global_expert_distribution_recorder,
+)
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -31,6 +31,11 @@
+from sglang.srt.eplb.expert_distribution import (
+    ExpertDistributionRecorder,
+    get_global_expert_distribution_recorder,
+)
+from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
@@ -64,11 +69,6 @@
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -32,6 +32,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +13/-13; `python/sglang/srt/models/qwen2_moe.py` modified +5/-5; `python/sglang/srt/models/deepseek_v2.py` modified +3/-5; `python/sglang/srt/models/qwen3_moe.py` modified +3/-5; `python/sglang/srt/layers/moe/topk.py` modified +3/-3; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-2
- 验证与风险: diff 自带测试面 `test/srt/test_expert_location_updater.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7681 - support qwen3 dense model dp attention

- 链接: https://github.com/sgl-project/sglang/pull/7681
- 状态/时间: merged / 2025-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `646cef2e2ea5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+49/-17，可读 patch 139 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support qwen3 dense model dp attention」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: support qwen3 dense model use dp attention for special use @ShangmingCai。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunks: -14,6 +14,8; -54,18 +56,21 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunks: -14,6 +14,8; -54,18 +56,21 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -14,6 +14,8 @@
+from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
+from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
@@ -54,18 +56,21 @@ def __init__(
-        assert self.total_num_heads % self.tp_size == 0
-        self.num_heads = self.total_num_heads // self.tp_size
+        attn_tp_rank = get_attention_tp_rank()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +42/-16
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7740 - [optimize] add two stream norm for qwen3

- 链接: https://github.com/sgl-project/sglang/pull/7740
- 状态/时间: merged / 2025-07-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `264dc6e74462`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+54/-10，可读 patch 229 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[optimize] add two stream norm for qwen3」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: overlap RMSNorm by two stream Co-authored-by: @ispobock before | max_concurrency | input_throughput | output_throughput | mean_ttft_ms | median_ttft_ms | p99_ttft_ms | mean_tpot...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunks: -23,15 +23,17; -49,6 +51,7 @@ def __init__(; symbols: Qwen3Attention, __init__, _apply_qk_norm，涉及 `Qwen3Attention, __init__, _apply_qk_norm`；`python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunks: -67,6 +67,7; -76,11 +77,12; symbols: Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm，涉及 `Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunks: -23,15 +23,17; -49,6 +51,7 @@ def __init__(; symbols: Qwen3Attention, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunks: -67,6 +67,7; -76,11 +77,12; symbols: Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -23,15 +23,17 @@
+from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
-from sglang.srt.utils import add_prefix
+from sglang.srt.utils import add_prefix, is_cuda
+_is_cuda = is_cuda()
@@ -49,6 +51,7 @@ def __init__(
+        alt_stream: Optional[torch.cuda.Stream] = None,
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -67,6 +67,7 @@
+from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
@@ -76,11 +77,12 @@
-from sglang.srt.utils import DeepEPMode, add_prefix, is_non_idle_and_non_empty
+from sglang.srt.utils import DeepEPMode, add_prefix, is_cuda, is_non_idle_and_non_empty
+_is_cuda = is_cuda()
@@ -352,6 +354,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +24/-5; `python/sglang/srt/models/qwen3_moe.py` modified +24/-5
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7723 - [Bug] add flashinfer bool check for fusedmoe in Qwen moe models

- 链接: https://github.com/sgl-project/sglang/pull/7723
- 状态/时间: merged / 2025-07-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+18/-0，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] add flashinfer bool check for fusedmoe in Qwen moe models」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: The current Qwen MoE models (`qwen2_moe.py` and `qwen3_moe.py`) do not pass the `enable_flashinfer_moe` flag, causing `FuseMoE` to default to `enable_flashinfer_moe=False`. Adde...。
- 实现要点: `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -143,6 +143,15 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunks: -115,6 +115,15 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -143,6 +143,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunks: -115,6 +115,15 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -143,6 +143,15 @@ def __init__(
+            # Additional args for FusedMoE
+            **(
+                dict(
+                    enable_flashinfer_moe=True,
+                    enable_ep_moe=global_server_args_dict["enable_ep_moe"],
+                )
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -115,6 +115,15 @@ def __init__(
+            # Additional args for FusedMoE
+            **(
+                dict(
+                    enable_flashinfer_moe=True,
+                    enable_ep_moe=global_server_args_dict["enable_ep_moe"],
+                )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3_moe.py` modified +9/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7745 - [feat] Support EAGLE3 for Qwen

- 链接: https://github.com/sgl-project/sglang/pull/7745
- 状态/时间: merged / 2025-07-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+81/-6，可读 patch 197 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] Support EAGLE3 for Qwen」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`；PR 正文摘要: refer to https://huggingface.co/Tengyunw/qwen3_8b_eagle3 and https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunks: -2,7 +2,7; -325,6 +325,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, set_embed_and_head，涉及 `__init__, get_input_embeddings, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunks: -18,7 +18,7; -717,6 +717,7 @@ def __init__(; symbols: __init__, forward, start_layer, end_layer，涉及 `__init__, forward, start_layer`；`python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunks: -440,6 +440,9 @@ def __init__(; -459,6 +462,7 @@ def forward(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunks: -293,6 +293,9 @@ def __init__(; -321,7 +324,12 @@ def forward(; symbols: __init__, get_input_embedding, forward，涉及 `__init__, get_input_embedding, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunks: -2,7 +2,7; -325,6 +325,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, set_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunks: -18,7 +18,7; -717,6 +717,7 @@ def __init__(; symbols: __init__, forward, start_layer, end_layer
  - `python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunks: -440,6 +440,9 @@ def __init__(; -459,6 +462,7 @@ def forward(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunks: -293,6 +293,9 @@ def __init__(; -321,7 +324,12 @@ def forward(; symbols: __init__, get_input_embedding, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -2,7 +2,7 @@
-from typing import Any, Dict, Iterable, Optional, Tuple
+from typing import Any, Dict, Iterable, List, Optional, Tuple
@@ -325,6 +325,9 @@ def __init__(
+        # For EAGLE3 support
+        self.capture_aux_hidden_states = False
@@ -346,10 +349,18 @@ def forward(
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -18,7 +18,7 @@
-from typing import Any, Dict, Iterable, Optional, Tuple
+from typing import Any, Dict, Iterable, List, Optional, Tuple
@@ -717,6 +717,7 @@ def __init__(
+        self.capture_aux_hidden_states = False
@@ -735,9 +736,13 @@ def forward(
+        aux_hidden_states = None
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -440,6 +440,9 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +28/-2; `python/sglang/srt/models/qwen3_moe.py` modified +25/-2; `python/sglang/srt/models/qwen2_moe.py` modified +15/-1; `python/sglang/srt/models/qwen2.py` modified +13/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7222 - DP Attention with Auto DeepEP Dispatch

- 链接: https://github.com/sgl-project/sglang/pull/7222
- 状态/时间: merged / 2025-07-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+136/-90，可读 patch 638 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DP Attention with Auto DeepEP Dispatch」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；PR 正文摘要: This PR enables auto DeepEP dispatch for DP attention. Integration with TBO will be supported in future PRs.。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunks: -34,7 +34,7; -686,21 +686,21 @@ def dispatch_a(; symbols: dispatch_a, dispatch_b, combine, combine_a，涉及 `dispatch_a, dispatch_b, combine`；`python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunks: -229,7 +229,7 @@ def forward_deepep(; -240,14 +240,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a，涉及 `forward_deepep, op_dispatch_a, op_experts`；`python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunks: -558,7 +558,7 @@ def forward_deepep(; -569,14 +569,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a，涉及 `forward_deepep, op_dispatch_a, op_experts`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3 (8 lines); hunks: -42,7 +42,7; -1178,12 +1178,14 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunks: -34,7 +34,7; -686,21 +686,21 @@ def dispatch_a(; symbols: dispatch_a, dispatch_b, combine, combine_a
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunks: -229,7 +229,7 @@ def forward_deepep(; -240,14 +240,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunks: -558,7 +558,7 @@ def forward_deepep(; -569,14 +569,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3 (8 lines); hunks: -42,7 +42,7; -1178,12 +1178,14 @@ def forward(; symbols: forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +2/-0 (2 lines); hunks: -254,6 +254,7 @@ class ForwardBatch:; -299,6 +300,7 @@ def init_new(; symbols: ForwardBatch, init_new
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py
@@ -34,7 +34,7 @@
-from sglang.srt.model_executor.forward_batch_info import ForwardMode
+from sglang.srt.model_executor.forward_batch_info import ForwardBatch
@@ -686,21 +686,21 @@ def dispatch_a(
-        forward_mode: ForwardMode = None,
+        forward_batch: ForwardBatch,
-        inner_state = self._get_impl(forward_mode).dispatch_a(
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -229,7 +229,7 @@ def forward_deepep(
-                forward_mode=forward_mode,
+                forward_batch=forward_batch,
@@ -240,14 +240,14 @@ def forward_deepep(
-            forward_mode=forward_mode,
+            forward_batch=forward_batch,
-                forward_mode=forward_mode,
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -558,7 +558,7 @@ def forward_deepep(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13; `python/sglang/srt/models/qwen3_moe.py` modified +7/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-7; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3; `python/sglang/srt/model_executor/forward_batch_info.py` modified +2/-0; `python/sglang/srt/two_batch_overlap.py` modified +7/-3
  - tests: `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-40
- 验证与风险: diff 自带测试面 `test/srt/test_hybrid_dp_ep_tp_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7762 - feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode

- 链接: https://github.com/sgl-project/sglang/pull/7762
- 状态/时间: merged / 2025-07-07
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+1006/-9，可读 patch 1203 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`；PR 正文摘要: This PR supports running DeepSeek-R1-W4AFP8 model with ep-moe mode(deepep mode support is on the way~) Due to the reduced space required for model weights and decreased bandwidt...。
- 实现要点: `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: W4AFp8Config, for, __init__, get_name，涉及 `W4AFp8Config, for, __init__`；`python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunks: -0,0 +1,215; symbols: cutlass_w4a8_moe，涉及 `cutlass_w4a8_moe`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunks: -12,6 +12,7; -20,6 +21,8; symbols: __init__, internal，涉及 `__init__, internal`；`python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunks: -146,6 +146,7 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_i...; -158,9 +159,66 @@ def run_moe_ep_preproess(topk_ids: torch.Tensor, num_expert...; symbols: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess, pre_reorder_triton_kernel_for_cutlass_moe，涉及 `compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: W4AFp8Config, for, __init__, get_name
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunks: -0,0 +1,215; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunks: -12,6 +12,7; -20,6 +21,8; symbols: __init__, internal
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunks: -146,6 +146,7 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_i...; -158,9 +159,66 @@ def run_moe_ep_preproess(topk_ids: torch.Tensor, num_expert...; symbols: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess, pre_reorder_triton_kernel_for_cutlass_moe
  - `python/sglang/srt/layers/quantization/fp8.py` modified +27/-6 (33 lines); hunks: -1,7 +1,7; -200,7 +200,7 @@ class Fp8LinearMethod(LinearMethodBase):; symbols: Fp8LinearMethod, __init__, create_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -0,0 +1,264 @@
+import logging
+from typing import Any, Dict, List, Optional
+import torch
+from torch.nn import Module
+from torch.nn.parameter import Parameter
+from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -0,0 +1,215 @@
+# SPDX-License-Identifier: Apache-2.0
+"""Cutlass W4A8 MoE kernel."""
+from typing import Optional
+import torch
+from sgl_kernel import (
+    cutlass_w4a8_moe_mm,
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -12,6 +12,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +27/-6; `python/sglang/srt/configs/model_config.py` modified +12/-1
- 验证与风险: diff 自带测试面 `python/sglang/test/test_cutlass_w4a8_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7634 - [Feature] Layer-wise Prefill

- 链接: https://github.com/sgl-project/sglang/pull/7634
- 状态/时间: merged / 2025-07-16
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+464/-2，可读 patch 616 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] Layer-wise Prefill」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`；PR 正文摘要: This PR implements layer-wise prefill functionality by adding a new `forward_split_prefill` method to all supported model architectures. This allows launching several decoder la...。
- 实现要点: `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunks: -647,6 +647,69 @@ def forward(; symbols: forward, forward_split_prefill, load_weights，涉及 `forward, forward_split_prefill, load_weights`；`python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunks: -381,6 +381,57 @@ def forward(; symbols: forward, forward_split_prefill, get_hidden_dim，涉及 `forward, forward_split_prefill, get_hidden_dim`；`python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunks: -318,6 +318,54 @@ def forward(; symbols: forward, forward_split_prefill, load_weights，涉及 `forward, forward_split_prefill, load_weights`；`python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunks: -406,6 +406,7 @@ def __init__(; -554,6 +555,49 @@ def forward(; symbols: __init__, forward, forward_split_prefill, start_layer，涉及 `__init__, forward, forward_split_prefill`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunks: -647,6 +647,69 @@ def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunks: -381,6 +381,57 @@ def forward(; symbols: forward, forward_split_prefill, get_hidden_dim
  - `python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunks: -318,6 +318,54 @@ def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunks: -406,6 +406,7 @@ def __init__(; -554,6 +555,49 @@ def forward(; symbols: __init__, forward, forward_split_prefill, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +43/-0 (43 lines); hunks: -745,6 +745,49 @@ def forward(; symbols: forward, forward_split_prefill, start_layer
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -647,6 +647,69 @@ def forward(
+    @torch.no_grad()
+    def forward_split_prefill(
+        self,
+        input_ids: torch.Tensor,
+        positions: torch.Tensor,
+        forward_batch: ForwardBatch,
diff -- python/sglang/srt/models/gemma2.py
@@ -381,6 +381,57 @@ def forward(
+    @torch.no_grad()
+    def forward_split_prefill(
+        self,
+        input_ids: torch.Tensor,
+        positions: torch.Tensor,
+        forward_batch: ForwardBatch,
diff -- python/sglang/srt/models/gemma.py
@@ -318,6 +318,54 @@ def forward(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +63/-0; `python/sglang/srt/models/gemma2.py` modified +51/-0; `python/sglang/srt/models/gemma.py` modified +48/-0; `python/sglang/srt/models/qwen2_moe.py` modified +44/-0; `python/sglang/srt/models/qwen3_moe.py` modified +43/-0; `python/sglang/srt/models/qwen3.py` modified +41/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- 链接: https://github.com/sgl-project/sglang/pull/7966
- 状态/时间: merged / 2025-07-19
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 39 个文件，+557/-872，可读 patch 2848 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/N] MoE Refactor: refactor `select_experts`」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；PR 正文摘要: This pull request extracts the `select_experts` computation from within `FusedMoE` and `EPMoE`, moving it outside these modules. This refactoring offers three key benefits: - En...。
- 实现要点: `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunks: -1,5 +1,7; -21,6 +23,9; symbols: __init__, create_weights, apply, forward_cuda，涉及 `__init__, create_weights, apply`；`python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunks: -12,12 +12,15; -52,6 +55,168; symbols: TopKOutput, TopK, __init__, forward_native，涉及 `TopKOutput, TopK, __init__`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunks: -1,15 +1,17; -20,6 +22,12; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method，涉及 `GPTQMarlinState, CompressedTensorsMoEMethod, __new__`；`python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunks: -3,7 +3,7; -37,6 +37,9; symbols: get_quant_method, apply, create_weights，涉及 `get_quant_method, apply, create_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunks: -1,5 +1,7; -21,6 +23,9; symbols: __init__, create_weights, apply, forward_cuda
  - `python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunks: -12,12 +12,15; -52,6 +55,168; symbols: TopKOutput, TopK, __init__, forward_native
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunks: -1,15 +1,17; -20,6 +22,12; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunks: -3,7 +3,7; -37,6 +37,9; symbols: get_quant_method, apply, create_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74 (87 lines); hunks: -1,17 +1,13; -28,7 +24,7; symbols: __init__, determine_expert_map, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/unquant.py
@@ -1,5 +1,7 @@
+from __future__ import annotations
-from typing import Callable, List, Optional
+from typing import TYPE_CHECKING, Callable, List, Optional
@@ -21,6 +23,9 @@
+if TYPE_CHECKING:
+    from sglang.srt.layers.moe.topk import TopKOutput
diff -- python/sglang/srt/layers/moe/topk.py
@@ -12,12 +12,15 @@
+from __future__ import annotations
-from typing import Callable, Optional
+from typing import TYPE_CHECKING, Callable, NamedTuple, Optional
+from sglang.srt.custom_op import CustomOp
@@ -52,6 +55,168 @@
+if _is_npu:
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -1,15 +1,17 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152; `python/sglang/srt/layers/moe/topk.py` modified +171/-5; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +11/-52
- 验证与风险: diff 自带测试面 `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7312 - Add get_hidden_dim to qwen3.py for correct lora

- 链接: https://github.com/sgl-project/sglang/pull/7312
- 状态/时间: merged / 2025-07-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `877e35d7754c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+240/-2，可读 patch 296 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_hidden_dim to qwen3.py for correct lora」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`；PR 正文摘要: This is fix of https://github.com/sgl-project/sglang/issues/7271 Without it qwen3 lora doesn't work。
- 实现要点: `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models，涉及 `TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models`；`python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunks: -330,6 +330,30 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward，涉及 `__init__, get_input_embeddings, get_hidden_dim`。
- 代码 diff 细节:
  - `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models
  - `python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunks: -330,6 +330,30 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
- 关键代码摘录:

```diff
diff -- test/srt/models/lora/test_lora_qwen3.py
@@ -0,0 +1,209 @@
+# Copyright 2023-2025 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/qwen3.py
@@ -330,6 +330,30 @@ def __init__(
+    def get_hidden_dim(self, module_name: str) -> Tuple[int]:
+        # return input_dim, output_dim
+        if module_name in ["q_proj", "qkv_proj"]:
+            return (
+                self.config.hidden_size,
+                self.config.head_dim * self.config.num_attention_heads,
```

- 已读文件:
  - tests: `test/srt/models/lora/test_lora_qwen3.py` added +209/-0
  - runtime: `python/sglang/srt/models/qwen3.py` modified +24/-0
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/models/lora/test_lora.py`, `test/srt/models/lora/test_lora_qwen3.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8280 - DP Enhancement

- 链接: https://github.com/sgl-project/sglang/pull/8280
- 状态/时间: merged / 2025-07-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 20 个文件，+665/-1116，可读 patch 3002 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「DP Enhancement」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`；PR 正文摘要: This PR is for CI test. All changes are separated in #8276, #8277, #8278, and #8279.。
- 实现要点: `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunks: -38,6 +38,11; -48,6 +53,7; symbols: ForwardBatch, init_new，涉及 `ForwardBatch, init_new`；`python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunks: -3,7 +3,8; -30,6 +31,34; symbols: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode，涉及 `DPPaddingMode, is_max_len, is_sum_len`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunks: -29,9 +29,9; -167,8 +167,15 @@ def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: get_batch_sizes_to_capture, __init__, can_run, capture_one_batch_size，涉及 `get_batch_sizes_to_capture, __init__, can_run`；`python/sglang/srt/layers/logits_processor.py` modified +34/-24 (58 lines); hunks: -27,7 +27,9; -111,7 +113,8 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata，涉及 `LogitsMetadata, from_forward_batch, compute_dp_attention_metadata`。
- 代码 diff 细节:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunks: -38,6 +38,11; -48,6 +53,7; symbols: ForwardBatch, init_new
  - `python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunks: -3,7 +3,8; -30,6 +31,34; symbols: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunks: -29,9 +29,9; -167,8 +167,15 @@ def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: get_batch_sizes_to_capture, __init__, can_run, capture_one_batch_size
  - `python/sglang/srt/layers/logits_processor.py` modified +34/-24 (58 lines); hunks: -27,7 +27,9; -111,7 +113,8 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunks: -1462,9 +1462,13 @@ def apply_torch_tp(self):; -1576,8 +1580,18 @@ def _forward_raw(; symbols: apply_torch_tp, forward_decode, _forward_raw, _preprocess_logits
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -38,6 +38,11 @@
+from sglang.srt.layers.dp_attention import (
+    DPPaddingMode,
+    get_attention_dp_rank,
+    get_attention_tp_size,
+)
@@ -48,6 +53,7 @@
diff -- python/sglang/srt/layers/dp_attention.py
@@ -3,7 +3,8 @@
-from typing import TYPE_CHECKING, List
+from enum import IntEnum, auto
+from typing import TYPE_CHECKING, List, Tuple
@@ -30,6 +31,34 @@
+class DPPaddingMode(IntEnum):
+    # Padding tokens to max length and then gather tokens using `all_gather_into_tensor`
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -29,9 +29,9 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22; `python/sglang/srt/layers/dp_attention.py` modified +72/-24; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25; `python/sglang/srt/layers/logits_processor.py` modified +34/-24; `python/sglang/srt/model_executor/model_runner.py` modified +21/-4; `python/sglang/srt/layers/communicator.py` modified +12/-12
- 验证与风险: diff 自带测试面 `test/srt/test_deepep_small.py`, `test/srt/test_hybrid_dp_ep_tp_mtp.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8036 - [NVIDIA] Add Flashinfer MoE blockscale fp8 backend

- 链接: https://github.com/sgl-project/sglang/pull/8036
- 状态/时间: merged / 2025-07-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+179/-47，可读 patch 439 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Add Flashinfer MoE blockscale fp8 backend」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；PR 正文摘要: Enable flashinfer moe blockscale fp8 backend for low latency scenario. The e2e perf shows up to 3x improvement (see here). cc. @kushanam @pavanimajety。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunks: -47,12 +47,17; -64,6 +69,13; symbols: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical，涉及 `forward, _get_tile_tokens_dim, EPMoE`；`python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunks: -56,7 +56,11; -302,15 +306,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal，涉及 `__init__, forward_normal_dual_stream, forward_normal`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunks: -75,7 +75,7 @@ def __init__(; -92,16 +92,16 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunks: -711,7 +711,7 @@ def __init__(self, quant_config: ModelOptFp4Config):; -865,7 +865,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: __init__, create_weights, process_weights_after_loading，涉及 `__init__, create_weights, process_weights_after_loading`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunks: -47,12 +47,17; -64,6 +69,13; symbols: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical
  - `python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunks: -56,7 +56,11; -302,15 +306,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunks: -75,7 +75,7 @@ def __init__(; -92,16 +92,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunks: -711,7 +711,7 @@ def __init__(self, quant_config: ModelOptFp4Config):; -865,7 +865,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/models/qwen2_moe.py` modified +2/-2 (4 lines); hunks: -147,10 +147,10 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -47,12 +47,17 @@
+    next_power_of_2,
+use_flashinfer_trtllm_moe = (
+    global_server_args_dict["enable_flashinfer_trtllm_moe"]
+    and global_server_args_dict["enable_ep_moe"]
+)
@@ -64,6 +69,13 @@
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -56,7 +56,11 @@
-from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
+from sglang.srt.layers.moe.ep_moe.layer import (
+    DeepEPMoE,
+    get_moe_impl_class,
+    use_flashinfer_trtllm_moe,
+)
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -75,7 +75,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7; `python/sglang/srt/models/deepseek_v2.py` modified +44/-20; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5; `python/sglang/srt/models/qwen2_moe.py` modified +2/-2; `python/sglang/srt/models/qwen3_moe.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8421 - [3/N] MoE Refactor: Simplify DeepEP Output

- 链接: https://github.com/sgl-project/sglang/pull/8421
- 状态/时间: merged / 2025-07-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+319/-276，可读 patch 862 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[3/N] MoE Refactor: Simplify DeepEP Output」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: - Introduce `DispatchOutput` to maintain dispatcher's results. - Move DeepEP's `dispatch` and `combine` operations from model files the moe layer file. After this PR, all forwar...。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunks: -1,7 +1,27; -24,7 +44,6; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, DeepEPDispatchMode，涉及 `DeepEPNormalOutput, format, DeepEPLLOutput`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunks: -1,5 +1,7; -50,6 +52,13; symbols: __init__, forward, dispatch, moe_impl，涉及 `__init__, forward, dispatch`；`python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunks: -144,19 +144,6 @@ def __init__(; -207,41 +194,12 @@ def forward_deepep(; symbols: __init__, forward, forward_deepep, op_gate，涉及 `__init__, forward, forward_deepep`；`python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunks: -594,41 +594,13 @@ def forward_deepep(; -689,8 +661,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b，涉及 `forward_deepep, op_select_experts, op_dispatch_a`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunks: -1,7 +1,27; -24,7 +44,6; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, DeepEPDispatchMode
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunks: -1,5 +1,7; -50,6 +52,13; symbols: __init__, forward, dispatch, moe_impl
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunks: -144,19 +144,6 @@ def __init__(; -207,41 +194,12 @@ def forward_deepep(; symbols: __init__, forward, forward_deepep, op_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunks: -594,41 +594,13 @@ def forward_deepep(; -689,8 +661,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b
  - `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: DispatchOutputFormat, is_standard, is_deepep_normal, is_deepep_ll
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py
@@ -1,7 +1,27 @@
+# TODO(ch-wan): this file will be moved to sglang/srt/layers/moe/token_dispatcher/deepep.py
+from __future__ import annotations
+from typing import (
+    TYPE_CHECKING,
+    List,
+    NamedTuple,
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -1,5 +1,7 @@
+from __future__ import annotations
-from typing import List, Optional, Tuple
+from typing import TYPE_CHECKING, List, Optional, Tuple
@@ -50,6 +52,13 @@
+if TYPE_CHECKING:
+    from sglang.srt.layers.moe.ep_moe.token_dispatcher import (
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -144,19 +144,6 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30; `python/sglang/srt/models/qwen3_moe.py` modified +12/-69; `python/sglang/srt/models/deepseek_v2.py` modified +13/-56; `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0; `python/sglang/srt/layers/moe/token_dispatcher/standard.py` added +19/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/__init__.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8448 - Support EPLB in FusedMoE

- 链接: https://github.com/sgl-project/sglang/pull/8448
- 状态/时间: merged / 2025-07-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+107/-11，可读 patch 407 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support EPLB in FusedMoE」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/glm4_moe.py`；PR 正文摘要: Fix #8398。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunks: -11,6 +11,7; -62,8 +63,9 @@ def __init__(; symbols: __init__, weight_loader, _weight_loader_physical，涉及 `__init__, weight_loader, _weight_loader_physical`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunks: -183,6 +183,7 @@ def __init__(; -196,6 +197,7 @@ def __init__(; symbols: __init__, weight_loader，涉及 `__init__, weight_loader`；`python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -434,6 +434,7 @@ def __init__(; -740,10 +741,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts，涉及 `__init__, determine_num_fused_shared_experts`；`python/sglang/srt/models/granitemoe.py` modified +3/-0 (3 lines); hunks: -43,6 +43,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunks: -11,6 +11,7; -62,8 +63,9 @@ def __init__(; symbols: __init__, weight_loader, _weight_loader_physical
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunks: -183,6 +183,7 @@ def __init__(; -196,6 +197,7 @@ def __init__(; symbols: __init__, weight_loader
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -434,6 +434,7 @@ def __init__(; -740,10 +741,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/models/granitemoe.py` modified +3/-0 (3 lines); hunks: -43,6 +43,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +3/-0 (3 lines); hunks: -78,6 +78,7 @@ class Grok1MoE(nn.Module):; -128,6 +129,7 @@ def __init__(; symbols: Grok1MoE, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -11,6 +11,7 @@
+from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
@@ -62,8 +63,9 @@ def __init__(
+        layer_id: int,
-        layer_id: Optional[int] = None,
+        num_fused_shared_experts: int = 0,
@@ -84,13 +86,15 @@ def __init__(
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -183,6 +183,7 @@ def __init__(
+        num_fused_shared_experts: int = 0,
@@ -196,6 +197,7 @@ def __init__(
+            num_fused_shared_experts=num_fused_shared_experts,
@@ -728,10 +730,19 @@ def weight_loader(
-        physical_expert_ids = (
-            get_global_expert_location_metadata().logical_to_all_physical(
diff -- python/sglang/srt/models/glm4_moe.py
@@ -434,6 +434,7 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3; `python/sglang/srt/models/glm4_moe.py` modified +3/-1; `python/sglang/srt/models/granitemoe.py` modified +3/-0; `python/sglang/srt/models/grok.py` modified +3/-0; `python/sglang/srt/models/llama4.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/eplb/expert_distribution.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8450 - [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE

- 链接: https://github.com/sgl-project/sglang/pull/8450
- 状态/时间: merged / 2025-08-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+131/-46，可读 patch 344 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`；PR 正文摘要: A followup PR to enable Flashinfer MoE blockscale fp8 backend for TP MoE. The previous PR is doing the same but for the EP MoE. cc. @kushanam。
- 实现要点: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunks: -1,10 +1,13; -33,6 +36,15; symbols: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping，涉及 `should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunks: -25,14 +25,22; -49,7 +57,6; symbols: _get_tile_tokens_dim, EPMoE, __init__, forward，涉及 `_get_tile_tokens_dim, EPMoE, __init__`；`python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunks: -72,6 +72,7 @@ def dummy_func(*args, **kwargs):; -490,6 +491,16 @@ def apply(; symbols: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod，涉及 `dummy_func, apply, get_tile_tokens_dim`；`python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunks: -59,7 +59,7; -317,7 +317,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunks: -1,10 +1,13; -33,6 +36,15; symbols: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunks: -25,14 +25,22; -49,7 +57,6; symbols: _get_tile_tokens_dim, EPMoE, __init__, forward
  - `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunks: -72,6 +72,7 @@ def dummy_func(*args, **kwargs):; -490,6 +491,16 @@ def apply(; symbols: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunks: -59,7 +59,7; -317,7 +317,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: -52,7 +52,7; -426,7 +426,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -1,10 +1,13 @@
+import importlib.util
+from functools import lru_cache
+from packaging import version as pkg_version
@@ -33,6 +36,15 @@
+@lru_cache(maxsize=1)
+def should_use_flashinfer_trtllm_moe():
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -25,14 +25,22 @@
-from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
+from sglang.srt.layers.moe.fused_moe_triton.layer import (
+    FlashInferFusedMoE,
+    FusedMoE,
+    should_use_flashinfer_trtllm_moe,
+)
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -72,6 +72,7 @@ def dummy_func(*args, **kwargs):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34; `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0; `python/sglang/srt/models/deepseek_v2.py` modified +3/-4; `python/sglang/srt/models/glm4_moe.py` modified +3/-3; `python/sglang/srt/server_args.py` modified +0/-4
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- 链接: https://github.com/sgl-project/sglang/pull/8658
- 状态/时间: merged / 2025-08-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 38 个文件，+342/-299，可读 patch 1748 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[5/N] MoE Refactor: Update MoE parallelism arguments」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`；PR 正文摘要: This PR introduces `--moe-a2a-backend` and deprecates `--enable-ep-moe` and `--enable-deepep-moe`. Benchmark & Profiling。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunks: -1,28 +1,17; -31,11 +20,9; symbols: __init__, forward, get_moe_impl_class，涉及 `__init__, forward, get_moe_impl_class`；`python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: MoeA2ABackend, _missing_, is_deepep, is_standard，涉及 `MoeA2ABackend, _missing_, is_deepep`；`python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunks: -29,6 +29,7; -61,7 +62,6; symbols: __init__, get_moe_weights，涉及 `__init__, get_moe_weights`；`python/sglang/srt/models/glm4_moe.py` modified +10/-15 (25 lines); hunks: -23,6 +23,7; -50,7 +51,6; symbols: __init__, Glm4MoeDecoderLayer，涉及 `__init__, Glm4MoeDecoderLayer`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunks: -1,28 +1,17; -31,11 +20,9; symbols: __init__, forward, get_moe_impl_class
  - `python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: MoeA2ABackend, _missing_, is_deepep, is_standard
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunks: -29,6 +29,7; -61,7 +62,6; symbols: __init__, get_moe_weights
  - `python/sglang/srt/models/glm4_moe.py` modified +10/-15 (25 lines); hunks: -23,6 +23,7; -50,7 +51,6; symbols: __init__, Glm4MoeDecoderLayer
  - `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +23/-0 (23 lines); hunks: -0,0 +1,23
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -1,28 +1,17 @@
-from typing import TYPE_CHECKING, List, Optional, Tuple
+from typing import TYPE_CHECKING, Optional
-from sglang.srt.distributed import (
-    get_tensor_model_parallel_rank,
-    get_tensor_model_parallel_world_size,
-)
diff -- python/sglang/srt/layers/moe/utils.py
@@ -0,0 +1,43 @@
+from enum import Enum
+class MoeA2ABackend(Enum):
+    STANDARD = ("standard", "none")
+    DEEPEP = "deepep"
+    @classmethod
+    def _missing_(cls, value):
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -29,6 +29,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35; `python/sglang/srt/layers/moe/utils.py` added +43/-0; `python/sglang/srt/models/deepseek_v2.py` modified +10/-15; `python/sglang/srt/models/glm4_moe.py` modified +10/-15; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +23/-0; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` renamed +8/-15
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/srt/test_deepep_large.py`, `test/srt/test_deepep_small.py`, `test/srt/test_eplb.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #8751 - [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead

- 链接: https://github.com/sgl-project/sglang/pull/8751
- 状态/时间: merged / 2025-08-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `89588179cfe4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+26/-6，可读 patch 65 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: - 1/3: https://github.com/sgl-project/sglang/pull/8751 （23s => 22s) - 2/3: https://github.com/sgl-project/sglang/pull/8753 (22s => 12s) - 3/3: https://github.com/sgl-project/sgl...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunks: -766,7 +766,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -805,11 +808,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location，涉及 `load_weights, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunks: -766,7 +766,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -805,11 +808,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -766,7 +766,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        params_dict = dict(self.named_parameters())
+        # Cache params_dict to avoid repeated expensive traversal of model parameters
+        if not hasattr(self, "_cached_params_dict"):
+            self._cached_params_dict = dict(self.named_parameters())
+        params_dict = self._cached_params_dict
@@ -805,11 +808,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +26/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8753 - [2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights

- 链接: https://github.com/sgl-project/sglang/pull/8753
- 状态/时间: merged / 2025-08-06
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 36 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/eplb/expert_location.py`；PR 正文摘要: - 1/3: https://github.com/sgl-project/sglang/pull/8751 （23s => 22s) - 2/3: https://github.com/sgl-project/sglang/pull/8753 (22s => 12s) - 3/3: https://github.com/sgl-project/sgl...。
- 实现要点: `python/sglang/srt/eplb/expert_location.py` modified +5/-1 (6 lines); hunks: -35,6 +35,7 @@ class ExpertLocationMetadata:; -221,6 +222,7 @@ def _init_raw(; symbols: ExpertLocationMetadata, _init_raw, update, logical_to_all_physical，涉及 `ExpertLocationMetadata, _init_raw, update`。
- 代码 diff 细节:
  - `python/sglang/srt/eplb/expert_location.py` modified +5/-1 (6 lines); hunks: -35,6 +35,7 @@ class ExpertLocationMetadata:; -221,6 +222,7 @@ def _init_raw(; symbols: ExpertLocationMetadata, _init_raw, update, logical_to_all_physical
- 关键代码摘录:

```diff
diff -- python/sglang/srt/eplb/expert_location.py
@@ -35,6 +35,7 @@ class ExpertLocationMetadata:
+    logical_to_all_physical_map_cpu: torch.Tensor  # CPU copy for performance
@@ -221,6 +222,7 @@ def _init_raw(
+            logical_to_all_physical_map_cpu=logical_to_all_physical_map_padded.cpu(),
@@ -251,6 +253,7 @@ def update(
+            "logical_to_all_physical_map_cpu",
@@ -270,9 +273,10 @@ def update(
```

- 已读文件:
  - runtime: `python/sglang/srt/eplb/expert_location.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/eplb/expert_location.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8987 - Fix incorrect default get_hidden_dim logic

- 链接: https://github.com/sgl-project/sglang/pull/8987
- 状态/时间: merged / 2025-08-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+36/-143，可读 patch 236 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix incorrect default get_hidden_dim logic」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/granite.py`, `python/sglang/srt/models/llama.py`；PR 正文摘要: See #8939 1. Correct previously wrong calculation logic. 2. Clean-up unnecessary model-wise overrides Benchmark & Profiling。
- 实现要点: `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunks: -432,40 +432,6 @@ def forward_split_prefill(; symbols: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size，涉及 `forward_split_prefill, get_hidden_dim, get_module_name`；`python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunks: -363,31 +363,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name，涉及 `forward, get_hidden_dim, get_module_name`；`python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunks: -532,31 +532,6 @@ def end_layer(self):; symbols: end_layer, get_input_embeddings, get_hidden_dim, get_module_name，涉及 `end_layer, get_input_embeddings, get_hidden_dim`；`python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunks: -330,30 +330,6 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward，涉及 `__init__, get_input_embeddings, get_hidden_dim`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunks: -432,40 +432,6 @@ def forward_split_prefill(; symbols: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size
  - `python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunks: -363,31 +363,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
  - `python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunks: -532,31 +532,6 @@ def end_layer(self):; symbols: end_layer, get_input_embeddings, get_hidden_dim, get_module_name
  - `python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunks: -330,30 +330,6 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
  - `python/sglang/srt/models/torch_native_llama.py` modified +0/-24 (24 lines); hunks: -416,30 +416,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma2.py
@@ -432,40 +432,6 @@ def forward_split_prefill(
-    def get_hidden_dim(self, module_name):
-        # return input_dim, output_dim
-        if module_name in ["q_proj", "qkv_proj"]:
-            return (
-                self.config.hidden_size,
-                self.config.head_dim * self.config.num_attention_heads,
diff -- python/sglang/srt/models/granite.py
@@ -363,31 +363,6 @@ def forward(
-    def get_hidden_dim(self, module_name):
-        # return input_dim, output_dim
-        if module_name in ["q_proj", "o_proj", "qkv_proj"]:
-            return self.config.hidden_size, self.config.hidden_size
-        elif module_name in ["kv_proj"]:
-            return self.config.hidden_size, self.config.hidden_size // (
diff -- python/sglang/srt/models/llama.py
@@ -532,31 +532,6 @@ def end_layer(self):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma2.py` modified +0/-34; `python/sglang/srt/models/granite.py` modified +0/-25; `python/sglang/srt/models/llama.py` modified +0/-25; `python/sglang/srt/models/qwen3.py` modified +0/-24; `python/sglang/srt/models/torch_native_llama.py` modified +0/-24; `python/sglang/srt/models/gemma3n_mm.py` modified +12/-6
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma3n_mm.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9014 - Fuse writing KV buffer into rope kernel (part 2: srt)

- 链接: https://github.com/sgl-project/sglang/pull/9014
- 状态/时间: merged / 2025-08-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+65/-6，可读 patch 147 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fuse writing KV buffer into rope kernel (part 2: srt)」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py`；PR 正文摘要: Fuse set_kv_buffer to sgl-kernel rope function, only for trtllm_mha attention (below is from @fzyzcjy) speed may be suboptimal (I have not done any ncu profile or thorough optim...。
- 实现要点: `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg，涉及 `GptOssConfig, forward_normal, _enable_fused_set_kv_buffer`；`python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda，涉及 `forward_cuda`；`python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config，涉及 `_set_envs_and_config`；`.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:。
- 代码 diff 细节:
  - `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg
  - `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
  - `.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:
  - `docker/Dockerfile.gb200` modified +1/-1 (2 lines); hunks: -64,7 +64,7 @@ RUN python3 -m pip install --no-cache-dir --upgrade pip setupt...
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gpt_oss.py
@@ -66,10 +66,15 @@
+_is_cuda = is_cuda()
+if _is_cuda:
+    from sgl_kernel import FusedSetKVBufferArg
@@ -196,6 +201,32 @@ def forward_normal(
+def _enable_fused_set_kv_buffer():
+    return _is_cuda
diff -- python/sglang/srt/layers/rotary_embedding.py
@@ -222,6 +222,7 @@ def forward_cuda(
+        fused_set_kv_buffer_arg=None,  # Optional[FusedSetKVBufferArg]
@@ -231,8 +232,17 @@ def forward_cuda(
+                # Compatible with old sgl-kernel
+                **(
+                    dict(fused_set_kv_buffer_arg=fused_set_kv_buffer_arg)
+                    if fused_set_kv_buffer_arg is not None
diff -- python/sglang/srt/entrypoints/engine.py
@@ -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +51/-2; `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/pyproject.toml` modified +1/-1
  - ci: `.github/workflows/pr-test-pd-router.yml` modified +1/-1
  - other: `docker/Dockerfile.gb200` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9147 - support Qwen3-MoE-w4afp8

- 链接: https://github.com/sgl-project/sglang/pull/9147
- 状态/时间: open / 2025-08-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 636 个文件，+14735/-62339，可读 patch 94998 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support Qwen3-MoE-w4afp8」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py`；PR 正文摘要: Follow https://github.com/sgl-project/sglang/pull/8118. Base on https://github.com/sgl-project/sglang/pull/7762. This PR primarily implements the adaptation of SGLang for Qwen3-...。
- 实现要点: `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunks: -1,1917 +0,0; symbols: BlockBase, __init__, get_activation, adaptive_enc_mask，涉及 `BlockBase, __init__, get_activation`；`python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunks: -1,1700 +0,0; symbols: DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config，涉及 `DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__`；`python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273 (1273 lines); hunks: -1,1273 +0,0; symbols: OpenAIServingResponses, __init__, _request_id_prefix, create_responses，涉及 `OpenAIServingResponses, __init__, _request_id_prefix`；`python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260 (1260 lines); hunks: -1,1260 +0,0; symbols: ConformerEncoderLayer, __init__, forward, TransformerEncoderBase，涉及 `ConformerEncoderLayer, __init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunks: -1,1917 +0,0; symbols: BlockBase, __init__, get_activation, adaptive_enc_mask
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunks: -1,1700 +0,0; symbols: DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273 (1273 lines); hunks: -1,1273 +0,0; symbols: OpenAIServingResponses, __init__, _request_id_prefix, create_responses
  - `python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260 (1260 lines); hunks: -1,1260 +0,0; symbols: ConformerEncoderLayer, __init__, forward, TransformerEncoderBase
  - `python/sglang/srt/models/gpt_oss.py` removed +0/-1134 (1134 lines); hunks: -1,1134 +0,0; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/phi4mm_utils.py
@@ -1,1917 +0,0 @@
-# Copyright 2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py
@@ -1,1700 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-"""Attention layer with Dual chunk flash attention and sparse attention.
-"""
-import functools
-import logging
-import math
diff -- python/sglang/srt/entrypoints/openai/serving_responses.py
@@ -1,1273 +0,0 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700; `python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273; `python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260; `python/sglang/srt/models/gpt_oss.py` removed +0/-1134; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +856/-275
- 验证与风险: diff 自带测试面 `python/sglang/test/attention/test_trtllm_mla_backend.py`, `python/sglang/test/few_shot_gsm8k.py`, `python/sglang/test/few_shot_gsm8k_engine.py`, `python/sglang/test/run_eval.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- 链接: https://github.com/sgl-project/sglang/pull/9101
- 状态/时间: merged / 2025-08-14
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+68/-16，可读 patch 210 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Feature: support qwen and llama4 reducescatter for dp attention padding」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`；PR 正文摘要: Similar to #8280 and #8539 , this PR adds support for using reduce-scatter instead of all-reduce after MoE/MLP layers in Qwen2 MoE, Qwen3 MoE, and Llama4 when DP attention uses...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunks: -144,11 +144,14 @@ def __init__(; -159,15 +162,19 @@ def get_moe_weights(self):; symbols: __init__, forward, get_moe_weights, forward_normal，涉及 `__init__, forward, get_moe_weights`；`python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunks: -107,10 +107,14 @@ def __init__(; -175,7 +179,10 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer，涉及 `__init__, forward, _is_moe_layer`；`python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunks: -91,10 +91,18 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunks: -144,11 +144,14 @@ def __init__(; -159,15 +162,19 @@ def get_moe_weights(self):; symbols: __init__, forward, get_moe_weights, forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunks: -107,10 +107,14 @@ def __init__(; -175,7 +179,10 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer
  - `python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunks: -91,10 +91,18 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/lora/layers.py` modified +6/-2 (8 lines); hunks: -253,7 +253,7 @@ def apply_lora(self, base_output: torch.Tensor, x: torch.Ten...; -270,7 +270,11 @@ def forward(self, input_: torch.Tensor):; symbols: apply_lora, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -144,11 +144,14 @@ def __init__(
-        self, hidden_states: torch.Tensor, forward_batch: Optional[ForwardBatch] = None
+        self,
+        hidden_states: torch.Tensor,
+        forward_batch: Optional[ForwardBatch] = None,
+        use_reduce_scatter: bool = False,
-            return self.forward_normal(hidden_states)
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -107,10 +107,14 @@ def __init__(
-    def forward(self, x):
+    def forward(
+        self,
+        x,
+        use_reduce_scatter: bool = False,
+    ):
diff -- python/sglang/srt/models/llama4.py
@@ -131,14 +131,19 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +18/-5; `python/sglang/srt/models/qwen2_moe.py` modified +18/-4; `python/sglang/srt/models/llama4.py` modified +16/-3; `python/sglang/srt/models/llama.py` modified +10/-2; `python/sglang/srt/lora/layers.py` modified +6/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/lora/layers.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama4.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #8118 - [feat] Support tp mode for DeepSeek-R1-W4AFP8

- 链接: https://github.com/sgl-project/sglang/pull/8118
- 状态/时间: merged / 2025-09-02
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+291/-120，可读 patch 710 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[feat] Support tp mode for DeepSeek-R1-W4AFP8」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；PR 正文摘要: Support tp mode for DeepSeek w4a8 model, which has a better performace than ep mode. 1. Add W4AFp8MoEMethod and associated `create_weights`, `process_weights_after_loading` func...。
- 实现要点: `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunks: -1,12 +1,14; -91,21 +93,38 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales，涉及 `get_quant_method, get_scaled_act_names, W4AFp8MoEMethod`；`python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9 (10 lines); hunks: -91,18 +91,10 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe，涉及 `cutlass_w4a8_moe`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2 (7 lines); hunks: -175,6 +175,8 @@ def __init__(; -593,8 +595,9 @@ def _weight_loader_impl(; symbols: __init__, _weight_loader_impl，涉及 `__init__, _weight_loader_impl`；`python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights，涉及 `determine_num_fused_shared_experts, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunks: -1,12 +1,14; -91,21 +93,38 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9 (10 lines); hunks: -91,18 +91,10 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2 (7 lines); hunks: -175,6 +175,8 @@ def __init__(; -593,8 +595,9 @@ def _weight_loader_impl(; symbols: __init__, _weight_loader_impl
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/configs/model_config.py` modified +2/-1 (3 lines); hunks: -393,9 +393,10 @@ def _parse_quant_hf_config(self):; symbols: _parse_quant_hf_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/w4afp8.py
@@ -1,12 +1,14 @@
-from typing import TYPE_CHECKING, Any, Dict, List, Optional
+from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
+from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
+from sglang.srt.layers.linear import LinearBase, UnquantizedLinearMethod
@@ -91,21 +93,38 @@ def get_quant_method(
+        from sglang.srt.managers.schedule_batch import global_server_args_dict
diff -- python/sglang/srt/layers/moe/cutlass_w4a8_moe.py
@@ -91,18 +91,10 @@ def cutlass_w4a8_moe(
-    assert (
-        w1_scale.shape[1] == w1_q.shape[2] * 2 / 512
-        and w1_scale.shape[2] == w1_q.shape[1] * 4
-    ), "W1 scale shape mismatch"
-    assert (
-        w2_scale.shape[1] == w2_q.shape[2] * 2 / 512
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -175,6 +175,8 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2; `python/sglang/srt/models/deepseek_v2.py` modified +5/-0; `python/sglang/srt/configs/model_config.py` modified +2/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +206/-60
  - tests: `python/sglang/test/test_cutlass_w4a8_moe.py` modified +24/-9
- 验证与风险: diff 自带测试面 `python/sglang/test/test_cutlass_w4a8_moe.py`, `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #7912 - Qwen FP8/NVFP4 ModelOPT Quantization support

- 链接: https://github.com/sgl-project/sglang/pull/7912
- 状态/时间: merged / 2025-09-03
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+43/-4，可读 patch 82 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Qwen FP8/NVFP4 ModelOPT Quantization support」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`；PR 正文摘要: 1. Add the quantization support for Qwen models from modelopt. so user can use it by one line of simple code: `llm = sgl.Engine(model_path="Qwen3-30B-A3B", modelopt_quant="model...。
- 实现要点: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunks: -517,6 +517,39 @@ def get_min_capability(cls) -> int:; -549,7 +582,7 @@ def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4C...; symbols: get_min_capability, get_config_filenames, common_group_size, from_config，涉及 `get_min_capability, get_config_filenames, common_group_size`；`python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunks: -24,7 +24,10; -458,7 +461,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunks: -517,6 +517,39 @@ def get_min_capability(cls) -> int:; -549,7 +582,7 @@ def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4C...; symbols: get_min_capability, get_config_filenames, common_group_size, from_config
  - `python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunks: -24,7 +24,10; -458,7 +461,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/modelopt_quant.py
@@ -517,6 +517,39 @@ def get_min_capability(cls) -> int:
+    @staticmethod
+    def common_group_size(cfg: dict) -> int:
+        """Return the unique group_size across the config; raise if missing/mismatched."""
+        sizes = set()
+        # Top-level and 'quantization' block
+        v = cfg.get("group_size")
diff -- python/sglang/srt/models/qwen3.py
@@ -24,7 +24,10 @@
-from sglang.srt.model_loader.weight_utils import default_weight_loader
+from sglang.srt.model_loader.weight_utils import (
+    default_weight_loader,
+    maybe_remap_kv_scale_name,
+)
@@ -458,7 +461,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2; `python/sglang/srt/models/qwen3.py` modified +8/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9973 - Optimize Qwen3-moe model by using flashinfer fused allreduce

- 链接: https://github.com/sgl-project/sglang/pull/9973
- 状态/时间: merged / 2025-09-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `ec15c8360e73`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+52/-12，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize Qwen3-moe model by using flashinfer fused allreduce」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: This PR is to make qwen moe model leverage FlashInfer fused_allreduce to fuse allreduce+rmsnorm+residual_add. The E2E input throughput improved 2.2%. Currently use a small MoE m...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunks: -42,7 +42,10; -57,10 +60,17; symbols: forward, get_moe_weights, forward_normal, __init__，涉及 `forward, get_moe_weights, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunks: -42,7 +42,10; -57,10 +60,17; symbols: forward, get_moe_weights, forward_normal, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -42,7 +42,10 @@
-from sglang.srt.layers.moe import get_moe_a2a_backend
+from sglang.srt.layers.moe import (
+    get_moe_a2a_backend,
+    should_use_flashinfer_cutlass_moe_fp4_allgather,
+)
@@ -57,10 +60,17 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +39/-8
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #9338 - Refactor TopK to ensure readability and extensibility

- 链接: https://github.com/sgl-project/sglang/pull/9338
- 状态/时间: merged / 2025-09-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 14 个文件，+52/-47，可读 patch 296 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Refactor TopK to ensure readability and extensibility」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；PR 正文摘要: Some recent fixes and optimizations are hardcoded in `deepseek_v2.py`. This PR slightly adjust the code structure and rename some variables to ensure readability and extensibility.。
- 实现要点: `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native，涉及 `TopKConfig, __init__, forward_native`；`python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim，涉及 `_is_fp4_quantization_enabled, selection, _get_tile_tokens_dim`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class，涉及 `_forward_ll, get_moe_impl_class`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunks: -260,7 +260,7 @@ def __init__(; -853,7 +853,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/topk.py
@@ -19,6 +19,7 @@
+    TYPE_CHECKING,
@@ -51,6 +52,9 @@
+if TYPE_CHECKING:
+    from sglang.srt.layers.quantization import QuantizationConfig
@@ -94,6 +98,7 @@ class TopKConfig:
+    output_format: Optional[TopKOutputFormat] = None
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -65,14 +65,10 @@
-    should_use_flashinfer_trtllm_moe,
-from sglang.srt.layers.moe.fused_moe_triton.layer import (
-    FusedMoE,
-    _is_fp4_quantization_enabled,
-)
-from sglang.srt.layers.moe.topk import TopK
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -74,16 +74,6 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +30/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; `python/sglang/srt/models/longcat_flash.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10574 - [Ascend]optimize Qwen3 on Ascend

- 链接: https://github.com/sgl-project/sglang/pull/10574
- 状态/时间: merged / 2025-09-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `e22f3a5ec914`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+81/-2，可读 patch 170 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend]optimize Qwen3 on Ascend」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: related to #10337 1. ~~use high-performance Attention Ops named `torch_npu._npu_paged_attention` in ACLGraph~~. The internal testing is ready. However, the relevant software pac...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunks: -30,12 +30,19; -235,9 +242,18 @@ def forward(; symbols: Qwen3Attention, forward，涉及 `Qwen3Attention, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunks: -30,12 +30,19; -235,9 +242,18 @@ def forward(; symbols: Qwen3Attention, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -30,12 +30,19 @@
-from sglang.srt.utils import add_prefix, is_cuda
+from sglang.srt.utils import (
+    add_prefix,
+    get_cmo_stream,
+    is_cuda,
+    is_npu,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +18/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/model_executor/model_runner.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- 链接: https://github.com/sgl-project/sglang/pull/10749
- 状态/时间: merged / 2025-09-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `a5095d62623f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+105/-34，可读 patch 207 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fuse write kv buffer into rope for qwen3 moe & bailing moe」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Fused write kv buffer into rope for qwen3 moe and bailing moe models. Got minor e2e speedup. Inspired by https://github.com/sgl-project/sglang/pull/9014 gsm8k result:。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core，涉及 `forward_prepare, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -60,6 +60,10 @@
+from sglang.srt.models.utils import (
+    create_fused_set_kv_buffer_arg,
+    enable_fused_set_kv_buffer,
+)
@@ -412,15 +416,31 @@ def forward_prepare(
-        q, k = self.rotary_emb(positions, q, k)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10975 - Use more general heuristics to set the default value of --mem-fraction-static

- 链接: https://github.com/sgl-project/sglang/pull/10975
- 状态/时间: merged / 2025-09-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+157/-141，可读 patch 568 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Use more general heuristics to set the default value of --mem-fraction-static」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`；PR 正文摘要: - Clean up the settings of chunked prefill size, mem fraction static in server_args.py. Make the heuristics more general. - We only want to check the `gpu_mem` once with a singl...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +0/-1 (1 lines); hunks: -1,6 +1,5；`python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunks: -523,108 +523,142 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes，涉及 `_handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes`；`python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunks: -35,6 +35,7; -84,8 +85,6 @@ class GenerateReqInput:; symbols: SessionParams, GenerateReqInput, contains_mm_input, __getitem__，涉及 `SessionParams, GenerateReqInput, contains_mm_input`；`.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunks: -99,8 +99,6 @@ jobs:; -189,8 +187,6 @@ jobs:。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +0/-1 (1 lines); hunks: -1,6 +1,5
  - `python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunks: -523,108 +523,142 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes
  - `python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunks: -35,6 +35,7; -84,8 +85,6 @@ class GenerateReqInput:; symbols: SessionParams, GenerateReqInput, contains_mm_input, __getitem__
  - `.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunks: -99,8 +99,6 @@ jobs:; -189,8 +187,6 @@ jobs:
  - `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10 (20 lines); hunks: -242,11 +242,8 @@ def find_local_hf_snapshot_dir(; -347,11 +344,14 @@ def download_weights_from_hf(; symbols: find_local_hf_snapshot_dir, download_weights_from_hf
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -1,6 +1,5 @@
-from functools import partial
diff -- python/sglang/srt/server_args.py
@@ -523,108 +523,142 @@ def _handle_missing_default_values(self):
-        Configure GPU memory-dependent settings including mem_fraction_static,
-        chunked_prefill_size, cuda_graph_max_bs, and cuda_graph_bs.
-        """
-        # Set mem fraction static
-        if self.mem_fraction_static is None:
-            if gpu_mem is not None:
diff -- python/sglang/srt/managers/io_struct.py
@@ -35,6 +35,7 @@
+# Parameters for a session
@@ -84,8 +85,6 @@ class GenerateReqInput:
-    # Extra key for classifying the request (e.g. cache_salt)
-    extra_key: Optional[Union[List[str], str]] = None
@@ -134,18 +133,23 @@ class GenerateReqInput:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +0/-1; `python/sglang/srt/server_args.py` modified +116/-82; `python/sglang/srt/managers/io_struct.py` modified +22/-13; `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10
  - ci: `.github/workflows/pr-test.yml` modified +0/-26
  - tests: `test/srt/test_multi_instance_release_memory_occupation.py` modified +5/-2; `test/srt/run_suite.py` modified +1/-5; `test/srt/test_mla_deepseek_v3.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/srt/lora/test_lora_llama4.py`, `test/srt/run_suite.py`, `test/srt/test_mla_deepseek_v3.py`, `test/srt/test_multi_instance_release_memory_occupation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- 链接: https://github.com/sgl-project/sglang/pull/10985
- 状态/时间: merged / 2025-10-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `61305291430a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+14/-2，可读 patch 58 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: https://github.com/sgl-project/sglang/pull/10749 This PR fixes an issue in Qwen3-MOE where `fused_set_kv_buffer_arg` was passed to `q, k = self.rotary_emb`, but only `RotaryEmbe...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core，涉及 `__init__, forward_prepare, forward_core`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -51,7 +51,7 @@
-from sglang.srt.layers.rotary_embedding import get_rope
+from sglang.srt.layers.rotary_embedding import MRotaryEmbedding, get_rope
@@ -358,6 +358,10 @@ def __init__(
+        self.compatible_with_fused_kv_buffer = (
+            False if isinstance(self.rotary_emb, MRotaryEmbedding) else True
+        )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #10911 - model: qwen3-omni (thinker-only)

- 链接: https://github.com/sgl-project/sglang/pull/10911
- 状态/时间: merged / 2025-10-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `86b04d25b3f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+1947/-328，可读 patch 2837 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「model: qwen3-omni (thinker-only)」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: solve #11343。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +2/-1 (3 lines); hunks: -661,13 +661,14 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-1 (3 lines); hunks: -661,13 +661,14 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -661,13 +661,14 @@ def __init__(
+        decoder_layer_type=Qwen3MoeDecoderLayer,
-            decoder_layer_type=Qwen3MoeDecoderLayer,
+            decoder_layer_type=decoder_layer_type,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-1
- 验证与风险: diff 自带测试面 `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_b.py`, `test/srt/test_vision_openai_server_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12002 - Eagle3 DP attention for Qwen3 MoE

- 链接: https://github.com/sgl-project/sglang/pull/12002
- 状态/时间: merged / 2025-10-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `750940ae3660`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+219/-27，可读 patch 372 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Eagle3 DP attention for Qwen3 MoE」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Add eagle3 dp attention for Qwen3 MoE model for large scale EP deployment. We test DP attention for Eagle3 in non-PD and PD scenarios. The results are all good. Single node of 8...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunks: -539,10 +539,16 @@ def forward(; -774,13 +780,15 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: forward, set_eagle3_layers_to_capture, load_weights，涉及 `forward, set_eagle3_layers_to_capture, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunks: -539,10 +539,16 @@ def forward(; -774,13 +780,15 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: forward, set_eagle3_layers_to_capture, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -539,10 +539,16 @@ def forward(
+        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +16/-8
- 验证与风险: diff 自带测试面 `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_eagle_dp_attention.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #12543 - Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell

- 链接: https://github.com/sgl-project/sglang/pull/12543
- 状态/时间: merged / 2025-11-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+107/-9，可读 patch 226 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；PR 正文摘要: Dependency **Require flashinfer-python >= 0.5.0** Usage Triton |Tasks|Version| Filter |n-shot| Metric | |Value | |Stderr| |-----|------:|----------------|-----:|-----------|---|...。
- 实现要点: `python/sglang/srt/layers/moe/utils.py` modified +20/-1 (21 lines); hunks: -2,7 +2,7; -248,3 +248,22 @@ def speculative_moe_backend_context():; symbols: speculative_moe_backend_context, RoutingMethodType，涉及 `speculative_moe_backend_context, RoutingMethodType`；`python/sglang/srt/layers/quantization/fp8.py` modified +12/-7 (19 lines); hunks: -1193,6 +1193,7 @@ def apply_with_router_logits(; -1204,26 +1205,30 @@ def apply_with_router_logits(; symbols: apply_with_router_logits，涉及 `apply_with_router_logits`；`python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1 (6 lines); hunks: -36,6 +36,7; -56,7 +57,7; symbols: __init__, _load_per_tensor_weight_scale，涉及 `__init__, _load_per_tensor_weight_scale`；`python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0 (2 lines); hunks: -68,6 +68,7 @@ def __init__(; -81,6 +82,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/utils.py` modified +20/-1 (21 lines); hunks: -2,7 +2,7; -248,3 +248,22 @@ def speculative_moe_backend_context():; symbols: speculative_moe_backend_context, RoutingMethodType
  - `python/sglang/srt/layers/quantization/fp8.py` modified +12/-7 (19 lines); hunks: -1193,6 +1193,7 @@ def apply_with_router_logits(; -1204,26 +1205,30 @@ def apply_with_router_logits(; symbols: apply_with_router_logits
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1 (6 lines); hunks: -36,6 +36,7; -56,7 +57,7; symbols: __init__, _load_per_tensor_weight_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0 (2 lines); hunks: -68,6 +68,7 @@ def __init__(; -81,6 +82,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +2/-0 (2 lines); hunks: -57,6 +57,7; -162,6 +163,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -2,7 +2,7 @@
-from enum import Enum
+from enum import Enum, IntEnum
@@ -248,3 +248,22 @@ def speculative_moe_backend_context():
+# The type of method in top-K routing, for use in torch custom op
+# Please keep this in sync with the counterpart defined in https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/trtllm/fused_moe/runner.h
+class RoutingMethodType(IntEnum):
diff -- python/sglang/srt/layers/quantization/fp8.py
@@ -1193,6 +1193,7 @@ def apply_with_router_logits(
+        from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -1204,26 +1205,30 @@ def apply_with_router_logits(
-        assert (
-            topk_config.num_expert_group is not None
-            and topk_config.topk_group is not None
-        ), "Current trtllm_fp8_block_scale_moe kernel does not support these two arguments as None"
diff -- python/sglang/srt/layers/moe/fused_moe_triton/layer.py
@@ -36,6 +36,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +20/-1; `python/sglang/srt/layers/quantization/fp8.py` modified +12/-7; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0; `python/sglang/srt/models/qwen2_moe.py` modified +2/-0
  - tests: `test/srt/nightly/test_flashinfer_trtllm_gen_moe_backend.py` added +65/-0; `test/srt/run_suite.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/srt/nightly/test_flashinfer_trtllm_gen_moe_backend.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13489 - Flashinfer TRTLLM-GEN-MoE + Qwen3

- 链接: https://github.com/sgl-project/sglang/pull/13489
- 状态/时间: merged / 2025-11-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `92ad2ff9ce0e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+43/-1，可读 patch 72 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Flashinfer TRTLLM-GEN-MoE + Qwen3」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: This cmd needs to work, https://github.com/sgl-project/sglang/pull/12543 only make it work for Qwen2 MoE (qwen3 next) Also set better default, st: Uses `flashinfer_trtllm` autom...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -49,6 +49,7; -111,6 +112,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -49,6 +49,7; -111,6 +112,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -49,6 +49,7 @@
+from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -111,6 +112,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.Renormalize,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12078 - [Ascend] qwen optimization

- 链接: https://github.com/sgl-project/sglang/pull/12078
- 状态/时间: merged / 2025-11-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+561/-108，可读 patch 998 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend] qwen optimization」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`；PR 正文摘要: related to #10337 -bugfix: 1.memory bugfix(w8a8_int8.py): in previous code, both layer.w13_weight and layer.w2_weight occupied double memory. now we solve it. 2.Cache Management...。
- 实现要点: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunks: -45,6 +45,10; -411,9 +415,142 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__，涉及 `DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE`；`python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunks: -625,53 +625,93 @@ def forward_decode_graph(; symbols: forward_decode_graph，涉及 `forward_decode_graph`；`python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: FuseEPDispatchOutput, format, FuseEPCombineInput, NpuFuseEPDispatcher，涉及 `FuseEPDispatchOutput, format, FuseEPCombineInput`；`python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunks: -70,6 +70,7; -78,6 +79,10; symbols: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare，涉及 `Qwen3MoeSparseMoeBlock, forward, op_core`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunks: -45,6 +45,10; -411,9 +415,142 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunks: -625,53 +625,93 @@ def forward_decode_graph(; symbols: forward_decode_graph
  - `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: FuseEPDispatchOutput, format, FuseEPCombineInput, NpuFuseEPDispatcher
  - `python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunks: -70,6 +70,7; -78,6 +79,10; symbols: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare
  - `python/sglang/srt/models/qwen3.py` modified +40/-4 (44 lines); hunks: -44,6 +44,9; -161,6 +164,33 @@ def _apply_qk_norm(; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward_prepare_native
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/ep_moe/layer.py
@@ -45,6 +45,10 @@
+if _is_npu:
+    import torch_npu
@@ -411,9 +415,142 @@ def npu_fused_moe_without_routing_weights_bf16(
+class NpuFuseEPMoE(DeepEPMoE):
+    def __init__(
+        self,
diff -- python/sglang/srt/layers/attention/ascend_backend.py
@@ -625,53 +625,93 @@ def forward_decode_graph(
-            k_cache = forward_batch.token_to_kv_pool.get_key_buffer(
-                layer.layer_id
-            ).view(-1, self.page_size, layer.tp_k_head_num * layer.qk_head_dim)
-            v_cache = forward_batch.token_to_kv_pool.get_value_buffer(
-                layer.layer_id
-            ).view(-1, self.page_size, layer.tp_v_head_num * layer.v_head_dim)
diff -- python/sglang/srt/layers/moe/token_dispatcher/fuseep.py
@@ -0,0 +1,97 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0; `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45; `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0; `python/sglang/srt/models/qwen3_moe.py` modified +56/-4; `python/sglang/srt/models/qwen3.py` modified +40/-4; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +29/-11
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #12330 - [CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next

- 链接: https://github.com/sgl-project/sglang/pull/12330
- 状态/时间: merged / 2025-12-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `test/srt/cpu/test_qwen3.py`；关联提交 `974c562a254e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+218/-0，可读 patch 241 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `test/srt/cpu/test_qwen3.py`；PR 正文摘要: This pr adds `fused_qkvzba_split_reshape_cat` kernel for Qwen3-next on CPU. Reference: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_next.py#L37...。
- 实现要点: `test/srt/cpu/test_qwen3.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat，涉及 `fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat`。
- 代码 diff 细节:
  - `test/srt/cpu/test_qwen3.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat
- 关键代码摘录:

```diff
diff -- test/srt/cpu/test_qwen3.py
@@ -0,0 +1,87 @@
+import unittest
+import torch
+from utils import precision
+from sglang.test.test_utils import CustomTestCase
+torch.manual_seed(1234)
+def fix_query_key_value_ordering_reshape_cat(
```

- 已读文件:
  - tests: `test/srt/cpu/test_qwen3.py` added +87/-0
- 验证与风险: diff 自带测试面 `test/srt/cpu/test_qwen3.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #14093 - Add fused FP8 KV cache write kernel for TRTLLM MHA backend

- 链接: https://github.com/sgl-project/sglang/pull/14093
- 状态/时间: merged / 2025-12-05
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+854/-7，可读 patch 925 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add fused FP8 KV cache write kernel for TRTLLM MHA backend」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: This PR introduces a fused FP8 KV cache write kernel for the TRTLLM MHA backend. It combines FP8 quantization and paged KV cache writes into a single Triton kernel (plus a naive...。
- 实现要点: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunks: -0,0 +1,467; symbols: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer，涉及 `_process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer`；`python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunks: -5,6 +5,7; -14,9 +15,12; symbols: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata，涉及 `get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer`；`python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunks: -422,6 +422,7 @@ def forward_prepare_npu(; -449,6 +450,7 @@ def forward_prepare_native(; symbols: forward_prepare_npu, forward_prepare_native, forward_core，涉及 `forward_prepare_npu, forward_prepare_native, forward_core`；`test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache，涉及 `TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunks: -0,0 +1,467; symbols: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer
  - `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunks: -5,6 +5,7; -14,9 +15,12; symbols: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunks: -422,6 +422,7 @@ def forward_prepare_npu(; -449,6 +450,7 @@ def forward_prepare_native(; symbols: forward_prepare_npu, forward_prepare_native, forward_core
  - `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py
@@ -0,0 +1,467 @@
+"""
+Fused FP8 quantization + paged KV cache write kernel for TRTLLM MHA backend.
+This kernel fuses the following operations:
+1. FP8 quantization of K and V tensors (from BF16/FP16 to FP8)
+2. Per-token or per-page scale computation
+3. Writing quantized K/V to paged KV cache layout
diff -- python/sglang/srt/layers/attention/trtllm_mha_backend.py
@@ -5,6 +5,7 @@
+import logging
@@ -14,9 +15,12 @@
+from sglang.srt.layers.attention.trtllm_fp8_kv_kernel import fused_fp8_set_kv_buffer
+logger = logging.getLogger(__name__)
@@ -411,6 +415,36 @@ def get_cuda_graph_seq_len_fill_value(self) -> int:
+    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -422,6 +422,7 @@ def forward_prepare_npu(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0; `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6; `python/sglang/srt/models/qwen3_moe.py` modified +9/-1
  - tests: `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0
- 验证与风险: diff 自带测试面 `test/manual/test_trtllm_fp8_kv_kernel.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #13998 - [apply][2/2] Fused qk_norm_rope for Qwen3-MoE

- 链接: https://github.com/sgl-project/sglang/pull/13998
- 状态/时间: merged / 2025-12-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `26d95008b65b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+199/-22，可读 patch 317 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[apply][2/2] Fused qk_norm_rope for Qwen3-MoE」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Qwen3-MoE qk_norm and rope ops account for a non-trivial proportion in model inference. See below: Take Qwen3-30B-A3B for example, it has 48 Decode Layers. Each layer runs qk_no...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunks: -18,10 +18,12; -73,6 +75,13; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range，涉及 `compute_yarn_parameters, get_mscale, find_correction_dim`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunks: -18,10 +18,12; -73,6 +75,13; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -18,10 +18,12 @@
-from typing import Any, Dict, Iterable, List, Optional, Tuple
+import math
+from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar
+from transformers import PretrainedConfig
@@ -73,6 +75,13 @@
+_is_cuda = is_cuda()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +193/-22
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #11984 - [Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix

- 链接: https://github.com/sgl-project/sglang/pull/11984
- 状态/时间: closed / 2025-12-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 17 个文件，+834/-95，可读 patch 1206 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/quantization/w4a4_int4.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`；PR 正文摘要: The latest version of CANN and Torch-NPU added support for int4 matmul-s and quantization, switching to lower bit quantization will reduce memory consumption by up to 2-x times...。
- 实现要点: `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: W4A4Int4Config, for, __init__, get_supported_act_dtypes，涉及 `W4A4Int4Config, for, __init__`；`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68 (159 lines); hunks: -7,6 +7,8; -17,8 +19,9; symbols: get_min_capability, process_weights_after_loading, create_weights，涉及 `get_min_capability, process_weights_after_loading, create_weights`；`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1 (124 lines); hunks: -11,6 +11,8; -26,7 +28,7; symbols: GPTQMarlinState, get_moe_method, apply, CompressedTensorsW8A8Int8MoEMethod，涉及 `GPTQMarlinState, get_moe_method, apply`；`python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8 (33 lines); hunks: -64,6 +64,7; -730,8 +731,11 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod，涉及 `process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: W4A4Int4Config, for, __init__, get_supported_act_dtypes
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68 (159 lines); hunks: -7,6 +7,8; -17,8 +19,9; symbols: get_min_capability, process_weights_after_loading, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1 (124 lines); hunks: -11,6 +11,8; -26,7 +28,7; symbols: GPTQMarlinState, get_moe_method, apply, CompressedTensorsW8A8Int8MoEMethod
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8 (33 lines); hunks: -64,6 +64,7; -730,8 +731,11 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -66,6 +66,7; -947,14 +948,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/w4a4_int4.py
@@ -0,0 +1,284 @@
+from __future__ import annotations
+from types import MappingProxyType
+from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, cast
+import torch
+from sglang.srt.layers.parameter import PerTensorScaleParameter
+from sglang.srt.layers.quantization.base_config import (
diff -- python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
@@ -7,6 +7,8 @@
+# TODO this import is a hotfix to avoid circular import error. revert after quantization refactor
+import sglang.srt.layers.quantization.w8a8_int8 as w8a8_int8_quant
@@ -17,8 +19,9 @@
-from sglang.srt.utils import is_cuda
+from sglang.srt.utils import is_cuda, is_npu
+_is_npu = is_npu()
diff -- python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -11,6 +11,8 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8; `python/sglang/srt/models/qwen3_moe.py` modified +10/-7; `python/sglang/srt/layers/quantization/unquant.py` modified +14/-0
- 验证与风险: diff 自带测试面 `test/srt/ascend/test_ascend_memory_consumption.py`, `test/srt/ascend/test_ascend_w4a4_quantization.py`, `test/srt/ascend/test_ascend_w8a8_quantization.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15223 - [bug fix][pp] fix qwen3 model load

- 链接: https://github.com/sgl-project/sglang/pull/15223
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `e9abb52576ea`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-3，可读 patch 17 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[bug fix][pp] fix qwen3 model load」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Now when launch Qwen3-0.6b with tp 2 pp4, the startup failed with below error: > [2025-12-16 10:15:15 PP0 TP1] Scheduler hit an exception: Traceback (most recent call last): Fil...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunks: -392,13 +392,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunks: -392,13 +392,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -392,13 +392,13 @@ def __init__(
-                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
+                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
-                    size=(config.vocab_size, config.hidden_size),
+                    size=self.lm_head.weight.shape,
-                    src=self.pp_group.first_rank,
+                    src=0,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15390 - [NPU]qwen3 pp bugfix

- 链接: https://github.com/sgl-project/sglang/pull/15390
- 状态/时间: merged / 2025-12-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `8bf7f240b654`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-3，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]qwen3 pp bugfix」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Fix bug for qwen3 model when pp-size > 1 qwen32b, pp-size=4。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunks: -167,10 +167,10 @@ def forward_prepare_native(self, positions, hidden_states):; -205,6 +205,7 @@ def forward(; symbols: forward_prepare_native, forward_prepare_npu, forward，涉及 `forward_prepare_native, forward_prepare_npu, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -542,7 +542,7 @@ def forward_prepare_npu(; symbols: forward_prepare_npu，涉及 `forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunks: -167,10 +167,10 @@ def forward_prepare_native(self, positions, hidden_states):; -205,6 +205,7 @@ def forward(; symbols: forward_prepare_native, forward_prepare_npu, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -542,7 +542,7 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -167,10 +167,10 @@ def forward_prepare_native(self, positions, hidden_states):
-    def forward_prepare_npu(self, positions, hidden_states):
+    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
-        if self.attn.layer_id == 0:
+        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
@@ -205,6 +205,7 @@ def forward(
+                forward_batch=forward_batch,
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -542,7 +542,7 @@ def forward_prepare_npu(
-        if self.attn.layer_id == 0:
+        if self.attn.layer_id == forward_batch.token_to_kv_pool.start_layer:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-2; `python/sglang/srt/models/qwen3_moe.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- 链接: https://github.com/sgl-project/sglang/pull/15835
- 状态/时间: merged / 2025-12-28
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 15 个文件，+827/-127，可读 patch 1151 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Feature] JIT Fused QK norm + qk norm clean up」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`；PR 正文摘要: 1. QK Norm kernel is not efficient enough. Around 2 months ago, flashinfer has released a newer version of QK-norm, which has not been integrated into sgl-kernel. This may lead...。
- 实现要点: `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm，涉及 `create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm`；`python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope，涉及 `__init__, _apply_qk_norm, op_prepare`；`python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native，涉及 `__init__, _apply_qk_norm, forward_prepare_native`；`python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward，涉及 `__init__, _apply_qk_norm, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope
  - `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -250,28 +251,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/utils.py
@@ -11,24 +11,27 @@
+from __future__ import annotations
-from typing import Any, Optional
+from typing import TYPE_CHECKING, Any, Optional, Tuple
+from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
+from sglang.jit_kernel.utils import register_jit_op
+from sglang.srt.environ import envs
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -57,12 +57,12 @@
-from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
+    apply_qk_norm,
@@ -498,31 +498,6 @@ def __init__(
-    def _apply_qk_norm(
-        self, q: torch.Tensor, k: torch.Tensor
-    ) -> Tuple[torch.Tensor, torch.Tensor]:
diff -- python/sglang/srt/models/qwen3.py
@@ -21,14 +21,14 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/utils.py` modified +80/-5; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27; `python/sglang/srt/models/qwen3.py` modified +9/-24; `python/sglang/srt/models/bailing_moe.py` modified +9/-23; `python/sglang/srt/models/glm4_moe.py` modified +9/-23; `python/sglang/srt/models/llada2.py` modified +9/-23
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_qknorm.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #16115 - [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head

- 链接: https://github.com/sgl-project/sglang/pull/16115
- 状态/时间: merged / 2026-01-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `7dd679cbb93a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+52/-16，可读 patch 166 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: 1、--enable-dp-lm-head causes Qwen3 to report an error. 2、split_qkv_rmsnorm_rope op parameters are not aligned. 1、When dp and enable-dp-lm-head are enabled, the group of lmhead d...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__，涉及 `forward_prepare_npu, __init__`；`python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu，涉及 `forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, forward_batch):
-            self.q_norm.weight,
-            self.k_norm.weight,
-            self.q_norm.variance_epsilon,
+            eps=self.q_norm.variance_epsilon,
+            q_weight=self.q_norm.weight,
+            k_weight=self.k_norm.weight,
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -523,12 +523,12 @@ def forward_prepare_npu(
-            self.q_norm.weight,
-            self.k_norm.weight,
-            self.q_norm.variance_epsilon,
+            eps=self.q_norm.variance_epsilon,
+            q_weight=self.q_norm.weight,
+            k_weight=self.k_norm.weight,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-3; `python/sglang/srt/models/qwen3_moe.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/llama.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- 链接: https://github.com/sgl-project/sglang/pull/13715
- 状态/时间: merged / 2026-01-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+49/-3，可读 patch 157 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix EPLB + FP4 Quantization Compatibility Issue」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`；PR 正文摘要: Note: The following description was generated by AI and may not be accurate. EPLB + FP4 Quantization Compatibility Issue Analysis and Fix Problem Description When using EPLB (Ex...。
- 实现要点: `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather，涉及 `get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather`；`python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward，涉及 `get_moe_weights, forward`；`python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts，涉及 `get_moe_weights, _forward_shared_experts`；`python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal，涉及 `get_moe_weights, forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -63,6 +63,7; -324,6 +325,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/moe/utils.py
@@ -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:
+def filter_moe_weight_param_global_expert(name, x, num_local_experts):
+    """
+    Filter out for MoE expert parameters that requires global expert.
+    """
+    return (
+        not getattr(x, "_sglang_require_global_experts", False)
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -103,7 +103,10 @@
-from sglang.srt.layers.moe.utils import RoutingMethodType
+from sglang.srt.layers.moe.utils import (
+    RoutingMethodType,
+    filter_moe_weight_param_global_expert,
+)
@@ -587,6 +590,9 @@ def get_moe_weights(self):
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -58,7 +58,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +12/-0; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15890 - [PP] fix wrong weight logic for tie_word_embeddings model

- 链接: https://github.com/sgl-project/sglang/pull/15890
- 状态/时间: merged / 2026-01-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+19/-48，可读 patch 108 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[PP] fix wrong weight logic for tie_word_embeddings model」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`；PR 正文摘要: see #15843 This PR fixes a bug where models with tie_word_embeddings=True (e.g., Qwen3-4B) output garbage text (random characters) when running with Pipeline Parallelism (e.g.,...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunks: -378,20 +378,6 @@ def __init__(; -500,6 +486,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunks: -457,20 +457,6 @@ def __init__(; -589,22 +575,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: __init__, load_weights，涉及 `__init__, load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunks: -378,20 +378,6 @@ def __init__(; -500,6 +486,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunks: -457,20 +457,6 @@ def __init__(; -589,22 +575,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: __init__, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -378,20 +378,6 @@ def __init__(
-        # perform weight tying for PP
-        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
-            if self.pp_group.is_first_rank:
-                self.pp_group.send(
-                    self.model.embed_tokens.weight, dst=self.pp_group.world_size - 1
-                )
diff -- python/sglang/srt/models/qwen2.py
@@ -457,20 +457,6 @@ def __init__(
-        # perform weight tying for PP
-        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
-            if self.pp_group.is_first_rank:
-                self.pp_group.send(
-                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
-                )
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-24; `python/sglang/srt/models/qwen2.py` modified +9/-24
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15904 - [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix

- 链接: https://github.com/sgl-project/sglang/pull/15904
- 状态/时间: merged / 2026-01-28
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `b77b0ffd6021`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+111/-49，可读 patch 296 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: The part of closed PR https://github.com/sgl-project/sglang/pull/11984. Adding weight conversion from ND to FRACTAL_NZ speeds up the GroupedMatmul kernel 1. Add NZ conversion fo...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -71,6 +71,7; -1119,14 +1120,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, get_model_config_for_expert_location，涉及 `load_weights, get_model_config_for_expert_location`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -71,6 +71,7; -1119,14 +1120,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, get_model_config_for_expert_location
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -71,6 +71,7 @@
+    LazyValue,
@@ -1119,14 +1120,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-        # TODO mimic deepseek
-        # Lazy initialization of expert weights cache to avoid slowing down load_weights
-            self.routed_experts_weights_of_layer = {
-                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-7
- 验证与风险: diff 自带测试面 `test/registered/ascend/test_ascend_memory_consumption.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #15203 - [NPU] support GPTQ quantization on npu

- 链接: https://github.com/sgl-project/sglang/pull/15203
- 状态/时间: merged / 2026-01-29
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+259/-6，可读 patch 361 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] support GPTQ quantization on npu」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/linear.py`；PR 正文摘要: This PR follows https://github.com/sgl-project/sglang/issues/15202 and Roadmap of NPU support https://github.com/sgl-project/sglang/issues/13664. 1. Add `GPTQLinearAscendMethod`...。
- 实现要点: `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunks: -49,7 +49,7; -63,6 +63,10; symbols: __init__, __repr__, get_scaled_act_names, get_name，涉及 `__init__, __repr__, get_scaled_act_names`；`python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunks: -299,7 +299,12 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7；`test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestAscendGPTQInt8, setUpClass, test_a_gsm8k，涉及 `TestAscendGPTQInt8, setUpClass, test_a_gsm8k`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunks: -49,7 +49,7; -63,6 +63,10; symbols: __init__, __repr__, get_scaled_act_names, get_name
  - `python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunks: -299,7 +299,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7
  - `test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestAscendGPTQInt8, setUpClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -133,6 +133,7
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/quantization/gptq.py
@@ -49,7 +49,7 @@
-from sglang.srt.utils import is_cuda
+from sglang.srt.utils import is_cuda, is_npu, set_weight_attrs
@@ -63,6 +63,10 @@
+_is_npu = is_npu()
+if _is_npu:
+    import torch_npu
diff -- python/sglang/srt/models/qwen3.py
@@ -299,7 +299,12 @@ def forward(
-                if _is_npu and not get_global_server_args().enable_piecewise_cuda_graph
+                if _is_npu
+                and not get_global_server_args().enable_piecewise_cuda_graph
+                and (
+                    hasattr(self.mlp.gate_up_proj, "weight")
+                    and hasattr(self.mlp.down_proj, "weight")
diff -- python/sglang/srt/layers/linear.py
@@ -61,6 +61,7 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5; `python/sglang/srt/models/qwen3.py` modified +6/-1; `python/sglang/srt/layers/linear.py` modified +1/-0
  - tests: `test/srt/ascend/test_ascend_gptq.py` added +73/-0; `test/srt/run_suite.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/srt/ascend/test_ascend_gptq.py`, `test/srt/run_suite.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17535 - Update weight rename check for Qwen3 Embeddings

- 链接: https://github.com/sgl-project/sglang/pull/17535
- 状态/时间: merged / 2026-02-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `793bf9fc0649`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-1，可读 patch 13 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Update weight rename check for Qwen3 Embeddings」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Update weight rename check for Qwen3 Embeddings Fine-tuned models may not have "Embedding" in their name causing loading issues Since SGLang always expects the "model" prefix an...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -490,7 +490,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -490,7 +490,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -490,7 +490,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
-            if "Embedding" in self.config.name_or_path:
+            if not name.startswith("model.") and (
+                name.startswith("layers.")
+                or name.startswith("embed_tokens.")
+                or name.startswith("norm.")
+            ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18189 - [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch

- 链接: https://github.com/sgl-project/sglang/pull/18189
- 状态/时间: merged / 2026-02-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `ca36d88fa640`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-0，可读 patch 15 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Support https://huggingface.co/nvidia/Qwen3-235B-A22B-Instruct-2507-NVFP4 Previously it failed to launch on SGLang. However, the 30B NVFP4 always worked. https://huggingface.co/...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunks: -890,6 +890,14 @@ def __init__(; symbols: __init__, Qwen3MoeForCausalLM，涉及 `__init__, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunks: -890,6 +890,14 @@ def __init__(; symbols: __init__, Qwen3MoeForCausalLM
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -890,6 +890,14 @@ def __init__(
+    # Mapping from fused module names to their component weight names.
+    # Required for quantization configs (e.g., ModelOpt FP4) to correctly identify
+    # which layers should be skipped based on the exclude_modules/ignore list.
+    packed_modules_mapping = {
+        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
+        "gate_up_proj": ["gate_proj", "up_proj"],
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +8/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19532 - [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe

- 链接: https://github.com/sgl-project/sglang/pull/19532
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；关联提交 `365ca1edb5af`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+8/-2，可读 patch 24 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: The previous PR 17511 added a condition ( forward_mode.is_extend() ) to avoid using fused Triton kernel (split_qkv_rmsnorm_rope) during prefill stage. However, when using specul...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunks: -181,7 +181,10 @@ def forward(; symbols: forward，涉及 `forward`；`python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunks: -620,7 +620,10 @@ def forward_prepare(; symbols: forward_prepare，涉及 `forward_prepare`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunks: -181,7 +181,10 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunks: -620,7 +620,10 @@ def forward_prepare(; symbols: forward_prepare
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -181,7 +181,10 @@ def forward(
-        if not _is_npu or forward_batch.forward_mode.is_extend():
+        if (
+            not _is_npu
+            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
+        ):
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -620,7 +620,10 @@ def forward_prepare(
-        if not _is_npu or forward_batch.forward_mode.is_extend():
+        if (
+            not _is_npu
+            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
+        ):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-1; `python/sglang/srt/models/qwen3_moe.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20127 - [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next

- 链接: https://github.com/sgl-project/sglang/pull/20127
- 状态/时间: open / 2026-03-08
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+66/-25，可读 patch 148 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`；PR 正文摘要: Fixes #19908 Qwen3MoE, Qwen2MoE, and Qwen3Next always create a separate `ParallelLMHead`, ignoring `config.tie_word_embeddings`. When `True`, the checkpoint has no `lm_head.weig...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunks: -62,7 +62,7; -947,13 +947,20 @@ def __init__(; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunks: -743,13 +743,20 @@ def __init__(; -850,6 +857,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights，涉及 `__init__, load_weights`；`python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunks: -888,14 +888,17 @@ def __init__(; -936,9 +939,11 @@ def get_embed_and_head(self):; symbols: __init__, get_embed_and_head, set_embed_and_head, load_weights，涉及 `__init__, get_embed_and_head, set_embed_and_head`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunks: -62,7 +62,7; -947,13 +947,20 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunks: -743,13 +743,20 @@ def __init__(; -850,6 +857,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunks: -888,14 +888,17 @@ def __init__(; -936,9 +939,11 @@ def get_embed_and_head(self):; symbols: __init__, get_embed_and_head, set_embed_and_head, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -62,7 +62,7 @@
-from sglang.srt.layers.utils import get_layer_id
+from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
@@ -947,13 +947,20 @@ def __init__(
-        self.lm_head = ParallelLMHead(
-            config.vocab_size,
-            config.hidden_size,
diff -- python/sglang/srt/models/qwen2_moe.py
@@ -743,13 +743,20 @@ def __init__(
-        self.lm_head = ParallelLMHead(
-            config.vocab_size,
-            config.hidden_size,
-            quant_config=quant_config,
-            prefix=add_prefix("lm_head", prefix),
-            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
diff -- python/sglang/srt/models/qwen3_next.py
@@ -888,14 +888,17 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +25/-8; `python/sglang/srt/models/qwen2_moe.py` modified +24/-7; `python/sglang/srt/models/qwen3_next.py` modified +17/-10
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3_next.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20474 - Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3

- 链接: https://github.com/sgl-project/sglang/pull/20474
- 状态/时间: open / 2026-03-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+159/-7，可读 patch 221 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `test/srt/xpu/test_qwen3.py`；PR 正文摘要: This PR integrates Qwen3 support on Intel XPU. It includes: 1. **Merged changes from contributor_fork/qwen3** (commit by Yang Wang): Fixes to run Qwen3 on XPU by adding XPU-spec...。
- 实现要点: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunks: -243,6 +243,15 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index，涉及 `forward_npu, forward_xpu, get_rope_index`；`python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunks: -21,11 +21,13; -172,6 +174,8 @@ def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count，涉及 `_layer_norm_fwd_1pass_kernel, _get_sm_count`；`test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: TestQwen3, setUpClass, tearDownClass, get_request_json，涉及 `TestQwen3, setUpClass, tearDownClass`；`docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunks: -20,6 +20,17 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +49,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...。
- 代码 diff 细节:
  - `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunks: -243,6 +243,15 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunks: -21,11 +21,13; -172,6 +174,8 @@ def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count
  - `test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: TestQwen3, setUpClass, tearDownClass, get_request_json
  - `docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunks: -20,6 +20,17 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +49,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...
  - `.github/workflows/pr-test-xpu.yml` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ jobs:
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/rotary_embedding/mrope.py
@@ -243,6 +243,15 @@ def forward_npu(
+    def forward_xpu(
+        self,
+        positions: torch.Tensor,
+        query: torch.Tensor,
+        key: torch.Tensor,
+        fused_set_kv_buffer_arg=None,
diff -- python/sglang/srt/layers/attention/fla/layernorm_gated.py
@@ -21,11 +21,13 @@
+    is_xpu,
+_is_xpu = is_xpu()
@@ -172,6 +174,8 @@ def _layer_norm_fwd_1pass_kernel(
+    if _is_xpu:
+        return torch.xpu.get_device_properties(device).gpu_eu_count
diff -- test/srt/xpu/test_qwen3.py
@@ -0,0 +1,133 @@
+"""
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0; `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0
  - tests: `test/srt/xpu/test_qwen3.py` added +133/-0; `test/srt/run_suite.py` modified +1/-0
  - other: `docker/xpu.Dockerfile` modified +11/-6
  - ci: `.github/workflows/pr-test-xpu.yml` modified +1/-1
- 验证与风险: diff 自带测试面 `test/srt/run_suite.py`, `test/srt/xpu/test_qwen3.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- 链接: https://github.com/sgl-project/sglang/pull/20520
- 状态/时间: open / 2026-03-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+172/-10，可读 patch 319 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]TP Communications compression For Qwen3 models for NPU」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/layers/linear.py`, `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2.py`；PR 正文摘要: Implemented INT8 TP communications compression on prefill for Qwen3 models. Compression achieves average 5% performance improvement on prefill intense benchmarks (see Benchmarki...。
- 实现要点: `python/sglang/srt/layers/linear.py` modified +11/-2 (13 lines); hunks: -19,6 +19,7; -38,6 +39,7; symbols: weight_loader_v2, forward，涉及 `weight_loader_v2, forward`；`python/sglang/srt/layers/communicator.py` modified +8/-3 (11 lines); hunks: -27,6 +27,7; -974,9 +975,13 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual，涉及 `_gather_hidden_states_and_residual`；`python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward，涉及 `forward`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/linear.py` modified +11/-2 (13 lines); hunks: -19,6 +19,7; -38,6 +39,7; symbols: weight_loader_v2, forward
  - `python/sglang/srt/layers/communicator.py` modified +8/-3 (11 lines); hunks: -27,6 +27,7; -974,9 +975,13 @@ def _gather_hidden_states_and_residual(; symbols: _gather_hidden_states_and_residual
  - `python/sglang/srt/models/qwen2.py` modified +6/-2 (8 lines); hunks: -91,13 +91,17 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward
  - `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0 (37 lines); hunks: -0,0 +1,37; symbols: TestLlama
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/linear.py
@@ -19,6 +19,7 @@
+    tensor_model_parallel_quant_all_reduce,
@@ -38,6 +39,7 @@
+from sglang.srt.server_args import get_global_server_args
@@ -1489,7 +1491,7 @@ def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor
-    def forward(self, input_, skip_all_reduce=False):
+    def forward(self, input_, skip_all_reduce=False, forward_batch=None):
diff -- python/sglang/srt/layers/communicator.py
@@ -27,6 +27,7 @@
+    attention_tensor_model_parallel_quant_all_reduce,
@@ -974,9 +975,13 @@ def _gather_hidden_states_and_residual(
-                hidden_states = attention_tensor_model_parallel_all_reduce(
-                    hidden_states
-                )
+                quantize_communications = \
diff -- python/sglang/srt/models/qwen2.py
@@ -91,13 +91,17 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/linear.py` modified +11/-2; `python/sglang/srt/layers/communicator.py` modified +8/-3; `python/sglang/srt/models/qwen2.py` modified +6/-2; `python/sglang/srt/models/qwen3.py` modified +1/-1; `python/sglang/srt/distributed/device_communicators/npu_communicator.py` modified +29/-1; `python/sglang/srt/server_args.py` modified +21/-0
  - tests: `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py` added +37/-0; `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py` added +37/-0
- 验证与风险: diff 自带测试面 `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #17784 - Upgrade transformers==5.3.0

- 链接: https://github.com/sgl-project/sglang/pull/17784
- 状态/时间: merged / 2026-03-18
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 95 个文件，+1136/-343，可读 patch 2752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Upgrade transformers==5.3.0」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`；PR 正文摘要: Address #17779 — Upgrade `transformers` to `5.3.0`. Changes - Bump `transformers>=5.2.0`, `huggingface_hub>=1.0.0`; remove `hf_transfer` - `get_rope_config()` utility for backwa...。
- 实现要点: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update，涉及 `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`；`python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope，涉及 `_get_rope_param, get_rope`；`python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes，涉及 `ModelImpl, is_deepseek_nsa, _derive_model_shapes`；`python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__，涉及 `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`。
- 代码 diff 细节:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/gemma3_causal.py
@@ -166,18 +166,36 @@ def __init__(
+        # In transformers v5, rope_parameters is nested per layer type:
+        #   {"sliding_attention": {"rope_theta": 10000}, "full_attention": {"rope_theta": 1000000}}
+        # In v4 it was flat: {"rope_type": "default", "rope_theta": ...}
+        rope_params = config.rope_parameters
+        is_nested = isinstance(rope_params, dict) and "full_attention" in rope_params
-            self.rope_theta = config.rope_local_base_freq
diff -- python/sglang/srt/layers/rotary_embedding/factory.py
@@ -2,6 +2,7 @@
+import logging
@@ -26,6 +27,29 @@
+logger = logging.getLogger(__name__)
+def _get_rope_param(rope_scaling, key, default, scaling_type):
+    """Get a parameter from rope_scaling dict, warn if missing.
+    In transformers v5, config.rope_scaling is an alias for rope_parameters
diff -- python/sglang/srt/configs/model_config.py
@@ -51,10 +51,20 @@ class ModelImpl(str, Enum):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- 验证与风险: diff 自带测试面 `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- 链接: https://github.com/sgl-project/sglang/pull/20931
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `46a76af97bec`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-3，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugifx] qwen3 rope parameter compatibility」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: To fix #20932 SGLang fails to load some Qwen3 MoE checkpoints whose HF config uses top-level `rope_theta` (v4-style) but does not define `rope_parameters` (v5-style). This PR up...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__，涉及 `forward_prepare_native, apply_qk_norm_rope, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -78,6 +78,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -566,7 +567,7 @@ def forward_prepare_native(
-            theta = self.config.rope_parameters["rope_theta"]
+            theta = self.rope_theta
@@ -691,8 +692,8 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_moe.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18233 - Support Qwen3 MoE context parallel

- 链接: https://github.com/sgl-project/sglang/pull/18233
- 状态/时间: merged / 2026-03-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `bb737d7a829b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+968/-73，可读 patch 1552 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Qwen3 MoE context parallel」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Context parallelism is essential in long context LLM inference. It splits a long input sequence across multiple GPUs so attention can be computed in parallel, drastically reduci...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +35/-5 (40 lines); hunks: -26,11 +26,14; -59,6 +62,11; symbols: __init__, forward_normal, get_input_embeddings, forward，涉及 `__init__, forward_normal, get_input_embeddings`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +35/-5 (40 lines); hunks: -26,11 +26,14; -59,6 +62,11; symbols: __init__, forward_normal, get_input_embeddings, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -26,11 +26,14 @@
+    get_attn_context_model_parallel_rank,
+    get_attn_context_model_parallel_world_size,
+    get_moe_data_parallel_world_size,
+    get_moe_tensor_parallel_world_size,
-    get_tensor_model_parallel_world_size,
-    tensor_model_parallel_all_reduce,
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +35/-5
- 验证与风险: diff 自带测试面 `python/sglang/test/attention/test_flashattn_backend.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`, `test/registered/unit/managers/test_prefill_adder.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21195 - Enable the qwen3 test

- 链接: https://github.com/sgl-project/sglang/pull/21195
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3_moe.py`；关联提交 `dac148167c80`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-5，可读 patch 32 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable the qwen3 test」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `python/sglang/srt/models/qwen3_moe.py`；PR 正文未提供可用摘要。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunks: -33,6 +33,7; -321,6 +322,9 @@ def forward_normal(; symbols: forward_normal，涉及 `forward_normal`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunks: -33,6 +33,7; -321,6 +322,9 @@ def forward_normal(; symbols: forward_normal
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -33,6 +33,7 @@
+    moe_expert_parallel_all_reduce,
@@ -321,6 +322,9 @@ def forward_normal(
+        if self.ep_size > 1 and not should_allreduce_fusion:
+            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-0
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen3_30b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21412 - [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints

- 链接: https://github.com/sgl-project/sglang/pull/21412
- 状态/时间: open / 2026-03-25
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-2，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Fixes the non-MoE counterpart of #20932, which fixed the same issue in `qwen3_moe.py`. SGLang fails to load Qwen3 (non-MoE) checkpoints whose HF config uses top-level `rope_thet...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunks: -31,6 +31,7; -216,8 +217,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunks: -31,6 +31,7; -216,8 +217,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -31,6 +31,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -216,8 +217,7 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
+        rope_theta, rope_scaling = get_rope_config(config)
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #19059 - [jit_kernel] Add fused_qknorm_rope JIT kernel

- 链接: https://github.com/sgl-project/sglang/pull/19059
- 状态/时间: merged / 2026-03-27
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+1127/-3，可读 patch 1152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[jit_kernel] Add fused_qknorm_rope JIT kernel」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`；PR 正文摘要: Part of tracking issue #17865 — migrate sgl-kernel AOT kernels to the lightweight python/sglang/jit_kernel/ system. This PR ports sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunks: -91,7 +91,10; -503,12 +506,18 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave，涉及 `_compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads`；`python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunks: -0,0 +1,307；`python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: bench_fused_qknorm_rope, calculate_diff，涉及 `bench_fused_qknorm_rope, calculate_diff`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunks: -91,7 +91,10; -503,12 +506,18 @@ def __init__(; symbols: __init__
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunks: -0,0 +1,307
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: bench_fused_qknorm_rope, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0 (181 lines); hunks: -0,0 +1,181; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, can_use_fused_qk_norm_rope, fused_qk_norm_rope
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -91,7 +91,10 @@
-    from sgl_kernel import fused_qk_norm_rope
+    from sglang.jit_kernel.fused_qknorm_rope import (
+        can_use_fused_qk_norm_rope,
+        fused_qk_norm_rope,
+    )
@@ -503,12 +506,18 @@ def __init__(
diff -- python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py
@@ -0,0 +1,444 @@
+"""
+Correctness tests for the fused_qknorm_rope JIT kernel.
+Validates fused_qk_norm_rope against a pure-PyTorch reference and (when
+available) the sgl_kernel AOT implementation.
+"""
+import pytest
diff -- python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh
@@ -0,0 +1,307 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +12/-3; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0; `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0
  - tests: `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21770 - [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon

- 链接: https://github.com/sgl-project/sglang/pull/21770
- 状态/时间: open / 2026-03-31
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+159/-0，可读 patch 161 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`；PR 正文摘要: This PR adds initial correctness coverage for the Apple Silicon MLX path using `Qwen/Qwen3-0.6B`. What’s included - `test/registered/models/test_qwen3_mlx_correctness.py` - a sm...。
- 实现要点: `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat，涉及 `TestQwen3MlxCorrectness, setUpClass, tearDownClass`；`test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy，涉及 `TestQwen3MlxAccuracy, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat
  - `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy
- 关键代码摘录:

```diff
diff -- test/registered/models/test_qwen3_mlx_correctness.py
@@ -0,0 +1,89 @@
+import os
+import unittest
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.test_utils import (
+    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
diff -- test/registered/models/test_qwen3_mlx_accuracy.py
@@ -0,0 +1,70 @@
+import os
+import unittest
+from types import SimpleNamespace
+from sglang.srt.utils import kill_process_tree
+from sglang.test.few_shot_gsm8k import run_eval
+from sglang.test.test_utils import (
```

- 已读文件:
  - tests: `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0; `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0
- 验证与风险: diff 自带测试面 `test/registered/models/test_qwen3_mlx_accuracy.py`, `test/registered/models/test_qwen3_mlx_correctness.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21654 - [jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE

- 链接: https://github.com/sgl-project/sglang/pull/21654
- 状态/时间: merged / 2026-04-01
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+208/-77，可读 patch 545 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[jit_kernel] Optimize fused_qknorm_rope: deduplicate sincosf for interleave RoPE」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`；PR 正文摘要: **Five optimizations to fusedQKNormRopeKernel:** **1. __sincosf deduplication (interleave RoPE)** In interleave (GPT-J) style RoPE, element pairs (2k, 2k+1) share the same half_...。
- 实现要点: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -513,6 +513,7 @@ def __init__(; -521,6 +522,7 @@ def __init__(; symbols: __init__，涉及 `__init__`；`python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunks: -39,11 +39,11 @@ namespace {; -68,11 +68,14 @@ compute_freq_yarn(float base, int rotary_dim, int half_dim,...；`python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunks: -1,8 +1,8; -39,7 +39,7; symbols: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff，涉及 `bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff`；`python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunks: -13,17 +13,20; -55,24 +58,25 @@ def fused_qk_norm_rope_out(; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out，涉及 `_jit_fused_qknorm_rope_module, fused_qk_norm_rope_out`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -513,6 +513,7 @@ def __init__(; -521,6 +522,7 @@ def __init__(; symbols: __init__
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunks: -39,11 +39,11 @@ namespace {; -68,11 +68,14 @@ compute_freq_yarn(float base, int rotary_dim, int half_dim,...
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunks: -1,8 +1,8; -39,7 +39,7; symbols: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunks: -13,17 +13,20; -55,24 +58,25 @@ def fused_qk_norm_rope_out(; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2 (4 lines); hunks: -122,7 +122,7 @@ def apply_interleave(x):; -231,7 +231,7 @@ def test_fused_qknorm_rope_partial_rotary(head_dim, is_neox):; symbols: apply_interleave, apply_neox, test_fused_qknorm_rope_partial_rotary
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -513,6 +513,7 @@ def __init__(
+        _yarn_factor, _, _, _ = compute_yarn_parameters(config)
@@ -521,6 +522,7 @@ def __init__(
+                _yarn_factor != 1.0,
diff -- python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh
@@ -39,11 +39,11 @@ namespace {
-__device__ inline float
-compute_freq_yarn(float base, int rotary_dim, int half_dim, float factor, float low, float high) {
+template <bool yarn>
+__device__ inline float compute_freq(float base, int rotary_dim, int half_dim, float factor, float low, float high) {
-  if (factor != 1.0f) {
+  if constexpr (yarn) {
diff -- python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py
@@ -1,8 +1,8 @@
-Measures throughput (us) for fused_qk_norm_rope across typical
-LLM configurations (head_dim x num_heads x num_tokens).
+Measures throughput (µs) for fused_qk_norm_rope across typical
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4; `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16
  - tests: `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2
- 验证与风险: diff 自带测试面 `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- 链接: https://github.com/sgl-project/sglang/pull/21458
- 状态/时间: merged / 2026-04-01
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `912494f59665`, `a188208e9a03`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+101/-3，可读 patch 152 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Use aiter's fused_qk_norm_mrope_3d_cache_pts_quant_shuffle kernel to replace 4 separate kernel launches (QKV split, QK RMSNorm, 3D mRoPE, KV cache write) with a single HIP kerne...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope，涉及 `__init__, forward_prepare_native, forward_prepare_npu`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -19,6 +19,7 @@
+from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
@@ -30,13 +31,25 @@
-from sglang.srt.utils import add_prefix, is_cuda, is_npu
+from sglang.srt.utils import add_prefix, get_bool_env_var, is_cuda, is_hip, is_npu
+_is_hip = is_hip()
+_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22429 - [NPU]add Qwen3-32b and Qwen3-8b low latency md

- 链接: https://github.com/sgl-project/sglang/pull/22429
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+296/-0，可读 patch 310 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]add Qwen3-32b and Qwen3-8b low latency md」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `docs/platforms/ascend/ascend_npu_best_practice.md`；PR 正文摘要: [NPU]add Qwen3-32b and Qwen3-8b low latency md [NPU]add Qwen3-32b and Qwen3-8b low latency md None None。
- 实现要点: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunks: -37,6 +37,10 @@ you encounter issues or have any questions, please [open an i...; -2345,6 +2349,298 @@ We tested it based on the `RANDOM` dataset.。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunks: -37,6 +37,10 @@ you encounter issues or have any questions, please [open an i...; -2345,6 +2349,298 @@ We tested it based on the `RANDOM` dataset.
- 关键代码摘录:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -37,6 +37,10 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Qwen3-32B       | Atlas 800I A3 | 2     | PD Mixed    | 1K+0.3K | 12ms | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-1k-0_3k-12ms-on-a3-2-cards-mixed-mode)      |
+| Qwen3-32B       | Atlas 800I A3 | 2     | PD Mixed    | 6K+1.5K | 17ms | W8A8 INT8    | [Optimal Configuration](#qwen3-32b-6k-1_5k-17ms-on-a3-2-cards-mixed-mode)      |
+| Qwen3-8B        | Atlas 800I A3 | 1     | PD Mixed    | 1K+0.3K | 7ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-8b-1k-0_3k-7ms-on-a3-1-cards-mixed-mode)        |
+| Qwen3-8B        | Atlas 800I A3 | 1     | PD Mixed    | 6K+1.5K | 9ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-8b-6k-1_5k-9ms-on-a3-1-cards-mixed-mode)        |
@@ -2345,6 +2349,298 @@ We tested it based on the `RANDOM` dataset.
+### Qwen3-32B 1K-0_3K 12ms on A3 2 Cards Mixed Mode
```

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_best_practice.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22450 - [NPU] Add Qwen3-14B low latency doc

- 链接: https://github.com/sgl-project/sglang/pull/22450
- 状态/时间: open / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+323/-0，可读 patch 344 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add Qwen3-14B low latency doc」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `docs/platforms/ascend/ascend_npu_best_practice.md`；PR 正文摘要: Add Qwen3-14B low latency example. docs/platforms/ascend/ascend_npu_best_practice.md。
- 实现要点: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunks: -39,6 +39,9 @@ you encounter issues or have any questions, please [open an is...; -53,6 +56,7 @@ you encounter issues or have any questions, please [open an is...。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunks: -39,6 +39,9 @@ you encounter issues or have any questions, please [open an is...; -53,6 +56,7 @@ you encounter issues or have any questions, please [open an is...
- 关键代码摘录:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -39,6 +39,9 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Qwen3-14B       | Atlas 800I A3 | 1     | PD Mixed    | 1K+0.3K | 9ms | W8A8 INT8  | [Optimal Configuration](#qwen3-14b-1k-0_3k-9ms-on-a3-1-card-mixed-mode)      |
+| Qwen3-14B       | Atlas 800I A3 | 1     | PD Mixed    | 6K+1.5K | 11ms | W8A8 INT8 | [Optimal Configuration](#qwen3-14b-6k-1_5k-11ms-on-a3-1-card-mixed-mode)     |
+| Qwen3-14B       | Atlas 800I A3 | 1     | PD Mixed    | 3.5K+1.5K | 8ms | W8A8 INT8 | [Optimal Configuration](#qwen3-14b-3_5k-1_5k-8ms-on-a3-1-card-mixed-mode)    |
@@ -53,6 +56,7 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Qwen3-14B                      | Atlas 800I A3 | 1     | PD Mixed          | 3.5K+1.5K | 50ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-14b-3_5k-1_5k-50ms-on-a3-1-card-mi
@@ -1864,6 +1868,325 @@ We tested it based on the `RANDOM` dataset.
```

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_best_practice.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22358 - Enable DFLASH support for additional model backends

- 链接: https://github.com/sgl-project/sglang/pull/22358
- 状态/时间: merged / 2026-04-09
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+152/-5，可读 patch 299 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable DFLASH support for additional model backends」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`；PR 正文摘要: Enable DFLASH for additional supported models from the z-lab collection: https://huggingface.co/collections/z-lab/dflash Based on #20547, landing this early to enable support fo...。
- 实现要点: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture，涉及 `forward, get_layer, get_input_embeddings`；`python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`；`python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head，涉及 `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`；`python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture，涉及 `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_5.py
@@ -574,8 +574,15 @@ def forward(
-        hidden_states, residual = self.layer_communicator.prepare_attn(
-            hidden_states, residual, forward_batch
+        hidden_states, residual = (
+            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
+                hidden_states,
+                residual,
diff -- python/sglang/srt/models/kimi_k25.py
@@ -849,6 +849,30 @@ def set_eagle3_layers_to_capture(
+    def set_dflash_layers_to_capture(self, layer_ids: List[int]) -> None:
+        """Set the layers to capture for DFLASH draft model training."""
+        if not hasattr(self.language_model, "set_dflash_layers_to_capture"):
+            raise AttributeError(
+                "language_model does not support DFLASH layer capture."
+            )
diff -- python/sglang/srt/models/qwen3_next.py
@@ -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22529 - [Model] Support sliding window attention for Qwen3

- 链接: https://github.com/sgl-project/sglang/pull/22529
- 状态/时间: open / 2026-04-10
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+29/-0，可读 patch 71 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support sliding window attention for Qwen3」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: - Add interleaved sliding window attention (SWA) support for Qwen3 model - Driven by HF config fields `layer_types` (primary) and `max_window_layers` (fallback), consistent with...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunks: -36,6 +36,17; -71,6 +82,7 @@ def __init__(; symbols: get_attention_sliding_window_size, __init__，涉及 `get_attention_sliding_window_size, __init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunks: -36,6 +36,17; -71,6 +82,7 @@ def __init__(; symbols: get_attention_sliding_window_size, __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -36,6 +36,17 @@
+# Aligned with HF's implementation, using sliding window inclusive with the last token
+# SGLang assumes exclusive
+def get_attention_sliding_window_size(config):
+    if getattr(config, "sliding_window", None) is not None:
+        return config.sliding_window - 1
+    else:
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +29/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22446 - [NPU] add qwen3-30b-a3b low latency example

- 链接: https://github.com/sgl-project/sglang/pull/22446
- 状态/时间: merged / 2026-04-11
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+130/-0，可读 patch 141 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] add qwen3-30b-a3b low latency example」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `docs/platforms/ascend/ascend_npu_best_practice.md`；PR 正文摘要: add qwen3-30b-a3b low latency example docs/platforms/ascend/ascend_npu_best_practice.md。
- 实现要点: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunks: -41,6 +41,8 @@ you encounter issues or have any questions, please [open an is...; -2779,3 +2781,131 @@ We tested it based on the `RANDOM` dataset.。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunks: -41,6 +41,8 @@ you encounter issues or have any questions, please [open an is...; -2779,3 +2781,131 @@ We tested it based on the `RANDOM` dataset.
- 关键代码摘录:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -41,6 +41,8 @@ you encounter issues or have any questions, please [open an issue](https://githu
+| Qwen3-30B-A3B   | Atlas 800I A3 | 1     | PD Mixed    | 6K+1.5K | 10ms | W8A8 INT8    | [Optimal Configuration](#qwen3-30b-a3b-6k-1_5k-10ms-on-a3-1-cards-mixed-mode)  |
+| Qwen3-30B-A3B   | Atlas 800I A3 | 1     | PD Mixed    | 1K+0.3K | 8ms  | W8A8 INT8    | [Optimal Configuration](#qwen3-30b-a3b-1k-0_3k-8ms-on-a3-1-cards-mixed-mode)   |
@@ -2779,3 +2781,131 @@ We tested it based on the `RANDOM` dataset.
+### Qwen3-30B-A3B 6K-1_5K 10ms on A3 1 Cards Mixed Mode
+Model: Qwen3-30B-A3B
+Hardware: Atlas 800I A3 1Card
```

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_best_practice.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22674 - [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization

- 链接: https://github.com/sgl-project/sglang/pull/22674
- 状态/时间: open / 2026-04-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+2/-0，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Support Qwen3.5-MoE and Qwen3-Next quantization」；模型线: Qwen3 Core；类别: 性能/后端优化；主要 diff: `python/sglang/srt/model_loader/loader.py`；PR 正文摘要: Qwen3.5 and Qwen3Next introduces GDN (Gated Dual Attention) linear attention layers that fuse separate projections into combined modules at load time: in_proj_qkv + in_proj_z →...。
- 实现要点: `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def _get_quantization_config(; symbols: _get_quantization_config，涉及 `_get_quantization_config`。
- 代码 diff 细节:
  - `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def _get_quantization_config(; symbols: _get_quantization_config
- 关键代码摘录:

```diff
diff -- python/sglang/srt/model_loader/loader.py
@@ -215,6 +215,8 @@ def _get_quantization_config(
+                    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
+                    "in_proj_ba": ["in_proj_b", "in_proj_a"],
```

- 已读文件:
  - runtime: `python/sglang/srt/model_loader/loader.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/model_loader/loader.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22687 - [NPU]qwen3-8b and 32b md bugfix

- 链接: https://github.com/sgl-project/sglang/pull/22687
- 状态/时间: merged / 2026-04-13
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-8，可读 patch 68 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU]qwen3-8b and 32b md bugfix」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `docs/platforms/ascend/ascend_npu_best_practice.md`；PR 正文摘要: [NPU]qwen3-8b and 32b md bugfix [NPU]qwen3-8b and 32b md bugfix None None。
- 实现要点: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunks: -2397,7 +2397,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`; -2410,7 +2409,7 @@ python -m sglang.launch_server --model-path $MODEL_PATH \。
- 代码 diff 细节:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunks: -2397,7 +2397,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`; -2410,7 +2409,7 @@ python -m sglang.launch_server --model-path $MODEL_PATH \
- 关键代码摘录:

```diff
diff -- docs/platforms/ascend/ascend_npu_best_practice.md
@@ -2397,7 +2397,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
-export HCCL_BUFFSIZE=400
@@ -2410,7 +2409,7 @@ python -m sglang.launch_server --model-path $MODEL_PATH \
-    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx --speculative-draft-model-quantization unquant \
+    --speculative-algorithm EAGLE3 --speculative-draft-model-path xxx \
@@ -2470,7 +2469,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
-export HCCL_BUFFSIZE=400
```

- 已读文件:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8
- 验证与风险: 该 PR 主要落在文档/示例 `docs/platforms/ascend/ascend_npu_best_practice.md`；验证重点是文档命令仍能映射到当前 CLI 参数和模型仓库名。

### PR #22739 - Restore Qwen3 rope config fallback

- 链接: https://github.com/sgl-project/sglang/pull/22739
- 状态/时间: merged / 2026-04-14
- 反查来源: `git log --name-only -- <model-files>` 反查到 `python/sglang/srt/models/qwen3.py`；关联提交 `520ce526b919`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+10/-2，可读 patch 19 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Restore Qwen3 rope config fallback」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3.py`；PR 正文摘要: Repro This happens when Qwen3 is launched with a JSON model override that provides `rope_scaling`. That makes `config.rope_parameters` exist, but it may not contain `rope_theta`...。
- 实现要点: `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -316,8 +316,16 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
+        if (
+            hasattr(config, "rope_parameters")
+            and config.rope_parameters
+            and "rope_theta" in config.rope_parameters
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-2
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22837 - [Bug] Qwen3 reasoning detector silently swallows tool_call when is missing

- 链接: https://github.com/sgl-project/sglang/pull/22837
- 状态/时间: open / 2026-04-15
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+43/-0，可读 patch 57 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bug] Qwen3 reasoning detector silently swallows tool_call when is missing」；模型线: Qwen3 Core；类别: 缺陷修复；主要 diff: `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`；PR 正文摘要: `Qwen3Detector` forwards every constructor argument to the base class **except** `tool_start_token`. The base `BaseReasoningFormatDetector` already contains a fallback that, whe...。
- 实现要点: `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunks: -269,6 +269,48 @@ def test_streaming_qwen3_forced_reasoning_format(self):; symbols: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector，涉及 `test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close`；`python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunks: -269,6 +269,48 @@ def test_streaming_qwen3_forced_reasoning_format(self):; symbols: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- test/registered/unit/parser/test_reasoning_parser.py
@@ -269,6 +269,48 @@ def test_streaming_qwen3_forced_reasoning_format(self):
+    def test_detect_and_parse_tool_call_without_think_close(self):
+        """
+        Regression test: when force_reasoning=True and the model emits <tool_call>
+        without first closing </think>, the tool_call must be split into normal_text
+        so the downstream tool-call parser can still see it. Otherwise the entire
+        output is silently swallowed into reasoning_content and the function call
diff -- python/sglang/srt/parser/reasoning_parser.py
@@ -242,6 +242,7 @@ def __init__(
+            tool_start_token="<tool_call>",
```

- 已读文件:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0
- 验证与风险: diff 自带测试面 `test/registered/unit/parser/test_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- 链接: https://github.com/sgl-project/sglang/pull/22003
- 状态/时间: merged / 2026-04-20
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+276/-25，可读 patch 485 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support moe_dp_size = 1 for various attention_cp_size」；模型线: Qwen3 Core；类别: 模型支持/运行时入口；主要 diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py`；PR 正文摘要: Previously, we can only support attention_cp_size == moe_dp_size which is too restricted. In the real world case, we should let the MoE part unchanged and only apply the context...。
- 实现要点: `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode，涉及 `ScatterMode, model_input_output, _compute_layer_input_mode`；`python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size，涉及 `attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank`；`python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings，涉及 `__init__, get_input_embeddings`；`python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split，涉及 `is_prefill_cp_in_seq_split, can_cp_split`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-1 (6 lines); hunks: -33,6 +33,9; -709,6 +712,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/communicator.py
@@ -50,8 +50,12 @@
+    get_moe_cp_rank,
+    get_moe_cp_size,
+    is_enable_moe_cp_allgather,
+    moe_cp_all_gather_into_tensor,
@@ -188,11 +192,13 @@ class ScatterMode(Enum):
+    MOE_FULL: full within the MoE group (cp_per_moe CP chunks), used when moe_dp_size < attn_cp_size
diff -- python/sglang/srt/layers/dp_attention.py
@@ -18,6 +18,9 @@
+)
+from sglang.srt.distributed import get_moe_dp_group as _get_moe_dp_group
+from sglang.srt.distributed import (
@@ -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
+def get_moe_cp_group() -> GroupCoordinator:
+    """Returns the MOE_DP group, which includes CP partners when attn_cp_size > moe_dp_size."""
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -968,9 +968,10 @@ def __init__(
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +164/-10; `python/sglang/srt/layers/dp_attention.py` modified +28/-0; `python/sglang/srt/models/qwen3_moe.py` modified +4/-3; `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4; `python/sglang/srt/models/qwen2_moe.py` modified +5/-1; `python/sglang/srt/distributed/parallel_state.py` modified +13/-7
  - tests: `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0
- 验证与风险: diff 自带测试面 `test/registered/4-gpu-models/test_qwen3_30b.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23372 - [NPU] Add CI tests for Speculative Decoding

- 链接: https://github.com/sgl-project/sglang/pull/23372
- 状态/时间: open / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+729/-14，可读 patch 799 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NPU] Add CI tests for Speculative Decoding」；模型线: Qwen3 Core；类别: 文档/测试/CI；主要 diff: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`；PR 正文摘要: This PR adds a comprehensive suite of NPU-specific CI tests for speculative decoding and expert parallelism features on Ascend hardware. The goal is to validate the correctness,...。
- 实现要点: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode，涉及 `TestAscendSpeculativeAttentionMode, setUpClass, start_prefill`；`test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info，涉及 `TestNpuSpeculativeDraftParams, setUpClass, tearDownClass`；`test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k，涉及 `TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k`；`test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k，涉及 `TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass`。
- 代码 diff 细节:
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +185/-0 (185 lines); hunks: -0,0 +1,185; symbols: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +159/-0 (159 lines); hunks: -0,0 +1,159; symbols: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +156/-0 (156 lines); hunks: -0,0 +1,156; symbols: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +105/-0 (105 lines); hunks: -0,0 +1,105; symbols: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: TestAscendSpeculativeMoeA2ABackend, setUpClass, test_a_gsm8k
- 关键代码摘录:

```diff
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py
@@ -0,0 +1,185 @@
+import os
+import unittest
+from types import SimpleNamespace
+from urllib.parse import urlparse
+from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
+from sglang.test.ascend.test_ascend_utils import (
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py
@@ -0,0 +1,159 @@
+import os
+import unittest
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.test_ascend_utils import (
+    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py
@@ -0,0 +1,156 @@
```

- 已读文件:
  - tests: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +185/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +159/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +156/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +105/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +97/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +14/-12
- 验证与风险: diff 自带测试面 `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23397 - [alignment-sglang] PR3: Dense Deterministic Math

- 链接: https://github.com/sgl-project/sglang/pull/23397
- 状态/时间: open / 2026-04-21
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 16 个文件，+2285/-50，可读 patch 2602 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[alignment-sglang] PR3: Dense Deterministic Math」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/layers/on_policy_utils.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/models/qwen3.py`；PR 正文摘要: - Make dense Qwen3 rollout numerically compatible with Megatron true-on-policy scoring - Stacked on PR2: #23350 Changes - `layernorm.py`: RMSNorm `forward_native` with fp32 prec...。
- 实现要点: `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target，涉及 `_get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled`；`python/sglang/srt/model_executor/cuda_graph_runner.py` modified +33/-12 (45 lines); hunks: -53,6 +53,9; -65,6 +68,7; symbols: _capture_one_stream，涉及 `_capture_one_stream`；`python/sglang/srt/models/qwen3.py` modified +13/-17 (30 lines); hunks: -15,6 +15,10; -102,13 +106,8 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`；`python/sglang/srt/layers/communicator.py` modified +18/-3 (21 lines); hunks: -27,6 +27,7; -57,6 +58,11; symbols: postprocess_layer, should_use_reduce_scatter, should_fuse_mlp_allreduce_with_next_layer, _gather_hidden_states_and_residual，涉及 `postprocess_layer, should_use_reduce_scatter, should_fuse_mlp_allreduce_with_next_layer`。
- 代码 diff 细节:
  - `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +33/-12 (45 lines); hunks: -53,6 +53,9; -65,6 +68,7; symbols: _capture_one_stream
  - `python/sglang/srt/models/qwen3.py` modified +13/-17 (30 lines); hunks: -15,6 +15,10; -102,13 +106,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/communicator.py` modified +18/-3 (21 lines); hunks: -27,6 +27,7; -57,6 +58,11; symbols: postprocess_layer, should_use_reduce_scatter, should_fuse_mlp_allreduce_with_next_layer, _gather_hidden_states_and_residual
  - `python/sglang/srt/layers/linear.py` modified +16/-2 (18 lines); hunks: -19,6 +19,7; -27,6 +28,10; symbols: forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/layers/on_policy_utils.py
@@ -0,0 +1,222 @@
+from __future__ import annotations
+import contextlib
+import os
+from typing import Any, Iterator, Optional
+import torch
+ROW_LINEAR_INV_BLOCK_K = 128
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -53,6 +53,9 @@
+from sglang.srt.layers.on_policy_utils import (
+    patch_prefill_only_deterministic_inference_for_cuda_graph,
+)
@@ -65,6 +68,7 @@
+from sglang.srt.server_args import get_global_server_args
@@ -802,19 +806,36 @@ def _capture_one_stream(stream_idx: Optional[int] = None):
diff -- python/sglang/srt/models/qwen3.py
@@ -15,6 +15,10 @@
```

- 已读文件:
  - runtime: `python/sglang/srt/layers/on_policy_utils.py` added +222/-0; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +33/-12; `python/sglang/srt/models/qwen3.py` modified +13/-17; `python/sglang/srt/layers/communicator.py` modified +18/-3; `python/sglang/srt/layers/linear.py` modified +16/-2; `python/sglang/srt/models/qwen2.py` modified +7/-9
- 验证与风险: diff 自带测试面 `test/registered/core/test_dense_deterministic_math.py`, `test/registered/core/test_on_policy_wiring.py`, `test/registered/core/test_tp_invariant_ops.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #23434 - [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model

- 链接: https://github.com/sgl-project/sglang/pull/23434
- 状态/时间: open / 2026-04-22
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-0，可读 patch 10 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model」；模型线: Qwen3 Core；类别: 模型实现调整；主要 diff: `python/sglang/srt/models/qwen3_classification.py`；PR 正文摘要: `Qwen3ForCausalLM` forwards `get_input_embeddings()` to its inner `Qwen3Model`, but `Qwen3ForPooledOutput` — the shared base for `Qwen3ForSequenceClassification` and the reward...。
- 实现要点: `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunks: -50,6 +50,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward，涉及 `__init__, get_input_embeddings, forward`。
- 代码 diff 细节:
  - `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunks: -50,6 +50,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward
- 关键代码摘录:

```diff
diff -- python/sglang/srt/models/qwen3_classification.py
@@ -50,6 +50,9 @@ def __init__(
+    def get_input_embeddings(self) -> nn.Embedding:
+        return self.model.get_input_embeddings()
```

- 已读文件:
  - runtime: `python/sglang/srt/models/qwen3_classification.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `python/sglang/srt/models/qwen3_classification.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
