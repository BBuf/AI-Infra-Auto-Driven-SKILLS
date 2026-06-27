# sglang Qwen3 Core Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` | [#25971](https://github.com/sgl-project/sglang/pull/25971) |
| `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx` | [#25971](https://github.com/sgl-project/sglang/pull/25971) |
| `python/sglang/srt/models/qwen3.py` | [#4693](https://github.com/sgl-project/sglang/pull/4693), [#6250](https://github.com/sgl-project/sglang/pull/6250), [#6990](https://github.com/sgl-project/sglang/pull/6990), [#7312](https://github.com/sgl-project/sglang/pull/7312), [#7681](https://github.com/sgl-project/sglang/pull/7681), [#7740](https://github.com/sgl-project/sglang/pull/7740), [#10574](https://github.com/sgl-project/sglang/pull/10574), [#15223](https://github.com/sgl-project/sglang/pull/15223), [#15390](https://github.com/sgl-project/sglang/pull/15390), [#16115](https://github.com/sgl-project/sglang/pull/16115), [#17535](https://github.com/sgl-project/sglang/pull/17535), [#19532](https://github.com/sgl-project/sglang/pull/19532), ... (16 total) |
| `python/sglang/srt/models/qwen3_moe.py` | [#4693](https://github.com/sgl-project/sglang/pull/4693), [#5917](https://github.com/sgl-project/sglang/pull/5917), [#6120](https://github.com/sgl-project/sglang/pull/6120), [#6250](https://github.com/sgl-project/sglang/pull/6250), [#6533](https://github.com/sgl-project/sglang/pull/6533), [#6598](https://github.com/sgl-project/sglang/pull/6598), [#6652](https://github.com/sgl-project/sglang/pull/6652), [#6709](https://github.com/sgl-project/sglang/pull/6709), [#6820](https://github.com/sgl-project/sglang/pull/6820), [#7740](https://github.com/sgl-project/sglang/pull/7740), [#8751](https://github.com/sgl-project/sglang/pull/8751), [#9973](https://github.com/sgl-project/sglang/pull/9973), ... (30 total) |
| `python/sglang/srt/models/qwen3_moe_mtp.py` | [#26468](https://github.com/sgl-project/sglang/pull/26468) |
| `test/registered/cpu/test_qwen3.py` | no direct PR-number commit |
| `test/registered/lora/test_lora_qwen3.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 40
- Extra PRs preserved from existing docs: 66
- Total PRs in this document: 106
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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
| 2025-06-05 | [#6820](https://github.com/sgl-project/sglang/pull/6820) | merged | Fix Qwen3MoE missing token padding optimization | `python/sglang/srt/models/qwen3_moe.py` |
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
| 2025-12-03 | [#12330](https://github.com/sgl-project/sglang/pull/12330) | merged | [CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next | `sgl-kernel/csrc/cpu/model/qwen3.cpp`, `test/srt/cpu/test_qwen3.py`, `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` |
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
| 2026-03-18 | [#17784](https://github.com/sgl-project/sglang/pull/17784) | merged | Upgrade transformers==5.3.0 | `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py` |
| 2026-03-20 | [#20931](https://github.com/sgl-project/sglang/pull/20931) | merged | [Bugifx] qwen3 rope parameter compatibility | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-22 | [#18233](https://github.com/sgl-project/sglang/pull/18233) | merged | Support Qwen3 MoE context parallel | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-24 | [#21195](https://github.com/sgl-project/sglang/pull/21195) | merged | Enable the qwen3 test | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-03-25 | [#21412](https://github.com/sgl-project/sglang/pull/21412) | open | [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints | `python/sglang/srt/models/qwen3.py` |
| 2026-03-27 | [#19059](https://github.com/sgl-project/sglang/pull/19059) | merged | [jit_kernel] Add fused_qknorm_rope JIT kernel | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` |
| 2026-03-31 | [#21770](https://github.com/sgl-project/sglang/pull/21770) | open | [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon | `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py` |
| 2026-04-01 | [#21654](https://github.com/sgl-project/sglang/pull/21654) | merged | Fused_qknorm_rope kernel optimization: up to 2.4× faster | `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` |
| 2026-04-01 | [#21458](https://github.com/sgl-project/sglang/pull/21458) | merged | [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write | `python/sglang/srt/models/qwen3.py` |
| 2026-04-09 | [#22429](https://github.com/sgl-project/sglang/pull/22429) | merged | [NPU]add Qwen3-32b and Qwen3-8b low latency md | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22450](https://github.com/sgl-project/sglang/pull/22450) | open | [NPU] Add Qwen3-14B low latency doc | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-09 | [#22358](https://github.com/sgl-project/sglang/pull/22358) | merged | Enable DFLASH support for additional model backends | `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py` |
| 2026-04-10 | [#22529](https://github.com/sgl-project/sglang/pull/22529) | open | [Model] Support sliding window attention for Qwen3 | `python/sglang/srt/models/qwen3.py` |
| 2026-04-11 | [#22446](https://github.com/sgl-project/sglang/pull/22446) | merged | [NPU] add qwen3-30b-a3b low latency example | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-13 | [#22687](https://github.com/sgl-project/sglang/pull/22687) | merged | [NPU]qwen3-8b and 32b md bugfix | `docs/platforms/ascend/ascend_npu_best_practice.md` |
| 2026-04-14 | [#22739](https://github.com/sgl-project/sglang/pull/22739) | merged | Restore Qwen3 rope config fallback | `python/sglang/srt/models/qwen3.py` |
| 2026-04-20 | [#22003](https://github.com/sgl-project/sglang/pull/22003) | merged | Support moe_dp_size = 1 for various attention_cp_size | `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-04-21 | [#23372](https://github.com/sgl-project/sglang/pull/23372) | open | [NPU] Add CI tests for Speculative Decoding | `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` |
| 2026-04-21 | [#23397](https://github.com/sgl-project/sglang/pull/23397) | open | [alignment-sglang] PR3: Dense Deterministic Math | `test/srt/models/test_params_mapping.py`, `python/sglang/srt/layers/on_policy_utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-04-25 | [#23731](https://github.com/sgl-project/sglang/pull/23731) | merged | Fix Qwen3 MoE double-reduce when DP attention + EP + reduce_scatterv (#23729) | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-04-26 | [#23734](https://github.com/sgl-project/sglang/pull/23734) | merged | Fix Qwen3 MoE: also guard EP all-reduce with not use_reduce_scatter (follow-up to #23731) | `python/sglang/srt/models/qwen3_moe.py` |
| 2026-04-26 | [#19484](https://github.com/sgl-project/sglang/pull/19484) | merged | [CPU] Add Qwen3.5 model optimization for CPU | `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2026-04-29 | [#23434](https://github.com/sgl-project/sglang/pull/23434) | merged | [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model | `python/sglang/srt/models/qwen3_classification.py` |
| 2026-05-02 | [#20520](https://github.com/sgl-project/sglang/pull/20520) | merged | [NPU]TP Communications compression For Qwen3 models for NPU | `python/sglang/srt/models/qwen3.py` |
| 2026-05-07 | [#22674](https://github.com/sgl-project/sglang/pull/22674) | closed | [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization | `python/sglang/srt/model_loader/loader.py` |
| 2026-05-20 | [#25825](https://github.com/sgl-project/sglang/pull/25825) | merged | [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool | `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py` |
| 2026-05-21 | [#25983](https://github.com/sgl-project/sglang/pull/25983) | merged | feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext | `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` |
| 2026-05-23 | [#23292](https://github.com/sgl-project/sglang/pull/23292) | merged | [CP] 1/N: Support MLA Prefill Context Parallel | `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py` |
| 2026-05-26 | [#25971](https://github.com/sgl-project/sglang/pull/25971) | merged | [CPU Doc]Add Xeon CPU info in Qwen3 Cookbook | `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` |
| 2026-05-27 | [#23269](https://github.com/sgl-project/sglang/pull/23269) | merged | Support batch size > 1 when enable CP | `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py` |
| 2026-05-29 | [#26673](https://github.com/sgl-project/sglang/pull/26673) | merged | [refactor] remove unused op_mlp | `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py` |
| 2026-05-29 | [#26468](https://github.com/sgl-project/sglang/pull/26468) | merged | [Model] Add Qwen3-MoE MTP | `python/sglang/srt/models/qwen3_moe_mtp.py`, `python/sglang/srt/models/qwen3_moe.py` |
| 2026-05-31 | [#26798](https://github.com/sgl-project/sglang/pull/26798) | merged | Make qwen3's set_embed_and_head idempotent | `python/sglang/srt/models/qwen3.py` |
| 2026-06-02 | [#25813](https://github.com/sgl-project/sglang/pull/25813) | merged | docs(cookbook): port popular model usage guides into cookbook pages | `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` |
| 2026-06-10 | [#23906](https://github.com/sgl-project/sglang/pull/23906) | merged | [Refactor] Cuda Graph Runner/Backend Refactor | `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` |
| 2026-06-10 | [#9147](https://github.com/sgl-project/sglang/pull/9147) | closed | support Qwen3-MoE-w4afp8 | `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py` |
| 2026-06-10 | [#22837](https://github.com/sgl-project/sglang/pull/22837) | closed | [Bug] Qwen3 reasoning detector silently swallows tool_call when is missing | `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py` |
| 2026-06-18 | [#28567](https://github.com/sgl-project/sglang/pull/28567) | merged | Add get_parallel(): a structured accessor for parallel-topology state | `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py` |
| 2026-06-18 | [#28421](https://github.com/sgl-project/sglang/pull/28421) | merged | [3/N][CP] Implement zigzag CP strategy | `python/sglang/srt/layers/cp/zigzag.py`, `python/sglang/srt/layers/cp/utils.py`, `python/sglang/srt/model_executor/model_runner.py` |
| 2026-06-19 | [#28697](https://github.com/sgl-project/sglang/pull/28697) | merged | [docs] Add B300 cookbook deployment options | `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` |
| 2026-06-20 | [#28810](https://github.com/sgl-project/sglang/pull/28810) | merged | [CI] Remove deprecated test/srt legacy CI setup | `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py` |

## Per-PR Diff Audit Cards

### PR #4693 - [Model] Adding Qwen3 and Qwen3MoE

- Link: https://github.com/sgl-project/sglang/pull/4693
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `4db463b1ad6e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +780/-14, 840 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Adding Qwen3 and Qwen3MoE"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Model] Adding Qwen3 and Qwen3MoE"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention, touching `Qwen3MoeSparseMoeBlock, __init__, forward`; `python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward, touching `Qwen3Attention, __init__, _apply_qk_norm`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` added +423/-0 (423 lines); hunks: -0,0 +1,423; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, Qwen3MoeAttention
  - `python/sglang/srt/models/qwen3.py` added +335/-0 (335 lines); hunks: -0,0 +1,335; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` added +423/-0; `python/sglang/srt/models/qwen3.py` added +335/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/flashinfer_backend.py`, `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #5917 - [qwen3] support qwen3 ep moe

- Link: https://github.com/sgl-project/sglang/pull/5917
- Status/date: merged / 2025-04-30
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `e330f2b86cd2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-6, 86 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[qwen3] support qwen3 ep moe"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[qwen3] support qwen3 ep moe"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunks: -40,6 +40,7; -48,6 +49,7; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-3 (11 lines); hunks: -40,6 +40,7; -48,6 +49,7; symbols: __init__, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +8/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6121 - feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088

- Link: https://github.com/sgl-project/sglang/pull/6121
- Status/date: merged / 2025-05-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +449/-70, 756 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`; technical summary: Covers "feat: add dp attention support for Qwen 2/3 MoE models, fixes #6088"; the main implementation surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/layers/dp_attention.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunks: -16,6 +16,8; -28,14 +30,23; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunks: -17,12 +17,15; -32,14 +35,23; symbols: __init__, forward, _FFNInputMode, touching `__init__, forward, _FFNInputMode`; `python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunks: -142,16 +142,6 @@ def get_local_attention_dp_size():; symbols: get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size, touching `get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size`; `python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunks: -269,6 +269,7 @@ def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model...; symbols: _maybe_prepare_dp_attn_batch, touching `_maybe_prepare_dp_attn_batch`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +227/-32 (259 lines); hunks: -16,6 +16,8; -28,14 +30,23; symbols: __init__, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +221/-28 (249 lines); hunks: -17,12 +17,15; -32,14 +35,23; symbols: __init__, forward, _FFNInputMode
  - `python/sglang/srt/layers/dp_attention.py` modified +0/-10 (10 lines); hunks: -142,16 +142,6 @@ def get_local_attention_dp_size():; symbols: get_local_attention_dp_size, get_local_attention_dp_rank, disable_dp_size
  - `python/sglang/bench_one_batch.py` modified +1/-0 (1 lines); hunks: -269,6 +269,7 @@ def _maybe_prepare_dp_attn_batch(batch: ScheduleBatch, model...; symbols: _maybe_prepare_dp_attn_batch
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +227/-32; `python/sglang/srt/models/qwen3_moe.py` modified +221/-28; `python/sglang/srt/layers/dp_attention.py` modified +0/-10; `python/sglang/bench_one_batch.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/bench_one_batch.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen2_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6250 - Add pipeline parallelism for Qwen2 and Qwen3 Model

- Link: https://github.com/sgl-project/sglang/pull/6250
- Status/date: merged / 2025-05-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `11553c1a3727`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +340/-73, 736 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add pipeline parallelism for Qwen2 and Qwen3 Model"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Add pipeline parallelism for Qwen2 and Qwen3 Model"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunks: -1,12 +1,14; -19,15 +21,18; symbols: Qwen3Attention, __init__, forward, start_layer, touching `Qwen3Attention, __init__, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunks: -17,6 +17,7; -28,6 +29,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, start_layer, touching `Qwen3MoeSparseMoeBlock, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +52/-10 (62 lines); hunks: -1,12 +1,14; -19,15 +21,18; symbols: Qwen3Attention, __init__, forward, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +49/-10 (59 lines); hunks: -17,6 +17,7; -28,6 +29,7; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, start_layer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +52/-10; `python/sglang/srt/models/qwen3_moe.py` modified +49/-10
- Risk and verification: The diff ships test coverage in `test/srt/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6120 - Support qwen3 deepep

- Link: https://github.com/sgl-project/sglang/pull/6120
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `fc0e3b91744b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +125/-8, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support qwen3 deepep"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Support qwen3 deepep"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunks: -32,6 +32,7; -54,8 +55,10; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +121/-7 (128 lines); hunks: -32,6 +32,7; -54,8 +55,10; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +121/-7
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6533 - support eplb for qwen3

- Link: https://github.com/sgl-project/sglang/pull/6533
- Status/date: merged / 2025-05-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `e6f113569e51`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +46/-25, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support eplb for qwen3"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "support eplb for qwen3"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunks: -55,7 +55,7; -67,6 +67,8; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, forward_normal, touching `Qwen3MoeSparseMoeBlock, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-22 (61 lines); hunks: -55,7 +55,7; -67,6 +67,8; symbols: Qwen3MoeSparseMoeBlock, __init__, forward, forward_normal
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +39/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6546 - added support for tied weights in qwen pipeline parallelism

- Link: https://github.com/sgl-project/sglang/pull/6546
- Status/date: merged / 2025-05-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +134/-20, 205 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "added support for tied weights in qwen pipeline parallelism"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`, `test/srt/test_pp_single_node.py`; technical summary: Covers "added support for tied weights in qwen pipeline parallelism"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`, `test/srt/test_pp_single_node.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunks: -21,7 +21,7; -249,15 +249,36 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`; `python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunks: -386,15 +386,36 @@ def __init__(; -470,7 +491,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights, touching `__init__, load_weights`; `test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunks: -116,6 +116,62 @@ def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test, touching `test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass`; `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +39/-10 (49 lines); hunks: -21,7 +21,7; -249,15 +249,36 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +38/-9 (47 lines); hunks: -386,15 +386,36 @@ def __init__(; -470,7 +491,15 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `test/srt/test_pp_single_node.py` modified +56/-0 (56 lines); hunks: -116,6 +116,62 @@ def test_pp_consistency(self):; symbols: test_pp_consistency, TestQwenPPTieWeightsAccuracy, setUpClass, run_gsm8k_test
  - `.github/workflows/pr-test.yml` modified +1/-1 (2 lines); hunks: -84,7 +84,7 @@ jobs:
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +39/-10; `python/sglang/srt/models/qwen2.py` modified +38/-9
  - tests: `test/srt/test_pp_single_node.py` modified +56/-0
  - ci: `.github/workflows/pr-test.yml` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6598 - qwen3moe support two batch overlap

- Link: https://github.com/sgl-project/sglang/pull/6598
- Status/date: merged / 2025-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `f9bab3d59100`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +351/-28, 515 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "qwen3moe support two batch overlap"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "qwen3moe support two batch overlap"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunks: -68,6 +68,9; -79,6 +82,7; symbols: __init__, forward_deepep, op_gate, op_select_experts, touching `__init__, forward_deepep, op_gate`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +200/-11 (211 lines); hunks: -68,6 +68,9; -79,6 +82,7; symbols: __init__, forward_deepep, op_gate, op_select_experts
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +200/-11
- Risk and verification: The diff ships test coverage in `test/srt/test_two_batch_overlap.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6652 - Fix qwen3 tbo/dp-lm-head

- Link: https://github.com/sgl-project/sglang/pull/6652
- Status/date: merged / 2025-05-27
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `b18416fbf869`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +3/-1, 25 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix qwen3 tbo/dp-lm-head"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fix qwen3 tbo/dp-lm-head"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -688,6 +688,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -688,6 +688,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -688,6 +688,7 @@ def __init__(
+            use_attn_tp_group=global_server_args_dict["enable_dp_lm_head"],
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/two_batch_overlap.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6709 - Fix PP for Qwen3 MoE

- Link: https://github.com/sgl-project/sglang/pull/6709
- Status/date: merged / 2025-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `e06b07610597`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +60/-4, 85 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix PP for Qwen3 MoE"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fix PP for Qwen3 MoE"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -812,9 +812,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -812,9 +812,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +3/-3
- Risk and verification: The diff ships test coverage in `test/srt/test_pp_single_node.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6818 - Fix wrong weight reference in dynamic EPLB

- Link: https://github.com/sgl-project/sglang/pull/6818
- Status/date: merged / 2025-06-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +27/-13, 83 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix wrong weight reference in dynamic EPLB"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py`; technical summary: Covers "Fix wrong weight reference in dynamic EPLB"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunks: -91,6 +91,7; -1661,6 +1662,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights, touching `__init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion`; `python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -18,15 +18,10; -811,6 +806,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`; `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -2257,3 +2257,16 @@ def support_triton(backend: str) -> bool:; symbols: support_triton, cpu_has_amx_support, LazyValue, __init__, touching `support_triton, cpu_has_amx_support, LazyValue`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-8 (21 lines); hunks: -91,6 +91,7; -1661,6 +1662,18 @@ def __init__(; symbols: __init__, routed_experts_weights_of_layer, determine_n_share_experts_fusion, post_load_weights
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-5 (6 lines); hunks: -18,15 +18,10; -811,6 +806,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
  - `python/sglang/srt/utils.py` modified +13/-0 (13 lines); hunks: -2257,3 +2257,16 @@ def support_triton(backend: str) -> bool:; symbols: support_triton, cpu_has_amx_support, LazyValue, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +13/-8; `python/sglang/srt/models/qwen3_moe.py` modified +1/-5; `python/sglang/srt/utils.py` modified +13/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6820 - Fix Qwen3MoE missing token padding optimization

- Link: https://github.com/sgl-project/sglang/pull/6820
- Status/date: merged / 2025-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `5aff1e9392d0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-3, 49 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Qwen3MoE missing token padding optimization"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fix Qwen3MoE missing token padding optimization"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -193,6 +193,7 @@ def forward_deepep(; -260,6 +261,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, touching `forward_deepep, op_select_experts`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -193,6 +193,7 @@ def forward_deepep(; -260,6 +261,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -193,6 +193,7 @@ def forward_deepep(
+                num_token_non_padded=forward_batch.num_token_non_padded,
@@ -260,6 +261,7 @@ def op_select_experts(self, state):
+                num_token_non_padded=state.forward_batch.num_token_non_padded,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #6990 - support qwen3 emebedding

- Link: https://github.com/sgl-project/sglang/pull/6990
- Status/date: merged / 2025-06-09
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `451ffe74d907`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +3/-0, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support qwen3 emebedding"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "support qwen3 emebedding"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunks: -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +2/-0 (2 lines); hunks: -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -333,6 +333,8 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if "Embedding" in self.config.name_or_path:
+                name = add_prefix(name, "model")
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +2/-0
- Risk and verification: The diff ships test coverage in `test/srt/models/test_embedding_models.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #6964 - Support both approximate and exact expert distribution collection

- Link: https://github.com/sgl-project/sglang/pull/6964
- Status/date: merged / 2025-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +101/-71, 257 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support both approximate and exact expert distribution collection"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/managers/expert_distribution.py`; technical summary: Covers "Support both approximate and exact expert distribution collection"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/managers/expert_distribution.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunks: -456,22 +456,25 @@ def op_select_experts(self, state):; symbols: op_select_experts, touching `op_select_experts`; `python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunks: -255,17 +255,20 @@ def op_select_experts(self, state):; symbols: op_select_experts, touching `op_select_experts`; `python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunks: -264,15 +264,23 @@ def init_new(; -347,7 +355,9 @@ def on_forward_pass_start(self, forward_batch: ForwardBatch):; symbols: init_new, __init__, on_forward_pass_start, on_select_experts, touching `init_new, __init__, on_forward_pass_start`; `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -182,7 +182,7 @@ class ServerArgs:; symbols: ServerArgs, touching `ServerArgs`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +19/-16 (35 lines); hunks: -456,22 +456,25 @@ def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-11 (25 lines); hunks: -255,17 +255,20 @@ def op_select_experts(self, state):; symbols: op_select_experts
  - `python/sglang/srt/managers/expert_distribution.py` modified +67/-43 (110 lines); hunks: -264,15 +264,23 @@ def init_new(; -347,7 +355,9 @@ def on_forward_pass_start(self, forward_batch: ForwardBatch):; symbols: init_new, __init__, on_forward_pass_start, on_select_experts
  - `python/sglang/srt/server_args.py` modified +1/-1 (2 lines); hunks: -182,7 +182,7 @@ class ServerArgs:; symbols: ServerArgs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +19/-16; `python/sglang/srt/models/qwen3_moe.py` modified +14/-11; `python/sglang/srt/managers/expert_distribution.py` modified +67/-43; `python/sglang/srt/server_args.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/expert_distribution.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7580 - Move files related to EPLB

- Link: https://github.com/sgl-project/sglang/pull/7580
- Status/date: merged / 2025-06-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 22 files, +42/-54, 289 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Move files related to EPLB"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "Move files related to EPLB"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunks: -39,6 +39,19; -54,18 +67,6; `python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunks: -31,6 +31,11; -64,11 +69,6; `python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -77,11 +80,6; `python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -63,11 +66,6.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +13/-13 (26 lines); hunks: -39,6 +39,19; -54,18 +67,6
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-5 (10 lines); hunks: -31,6 +31,11; -64,11 +69,6
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -77,11 +80,6
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-5 (8 lines); hunks: -32,6 +32,9; -63,11 +66,6
  - `python/sglang/srt/layers/moe/topk.py` modified +3/-3 (6 lines); hunks: -18,12 +18,12
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +13/-13; `python/sglang/srt/models/qwen2_moe.py` modified +5/-5; `python/sglang/srt/models/deepseek_v2.py` modified +3/-5; `python/sglang/srt/models/qwen3_moe.py` modified +3/-5; `python/sglang/srt/layers/moe/topk.py` modified +3/-3; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/srt/test_expert_location_updater.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7681 - support qwen3 dense model dp attention

- Link: https://github.com/sgl-project/sglang/pull/7681
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `646cef2e2ea5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +49/-17, 139 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support qwen3 dense model dp attention"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "support qwen3 dense model dp attention"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunks: -14,6 +14,8; -54,18 +56,21 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +42/-16 (58 lines); hunks: -14,6 +14,8; -54,18 +56,21 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +42/-16
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7740 - [optimize] add two stream norm for qwen3

- Link: https://github.com/sgl-project/sglang/pull/7740
- Status/date: merged / 2025-07-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `264dc6e74462`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +54/-10, 229 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[optimize] add two stream norm for qwen3"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[optimize] add two stream norm for qwen3"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunks: -23,15 +23,17; -49,6 +51,7 @@ def __init__(; symbols: Qwen3Attention, __init__, _apply_qk_norm, touching `Qwen3Attention, __init__, _apply_qk_norm`; `python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunks: -67,6 +67,7; -76,11 +77,12; symbols: Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm, touching `Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +24/-5 (29 lines); hunks: -23,15 +23,17; -49,6 +51,7 @@ def __init__(; symbols: Qwen3Attention, __init__, _apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +24/-5 (29 lines); hunks: -67,6 +67,7; -76,11 +77,12; symbols: Qwen3MoeSparseMoeBlock, __init__, _apply_qk_norm
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +24/-5; `python/sglang/srt/models/qwen3_moe.py` modified +24/-5
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7723 - [Bug] add flashinfer bool check for fusedmoe in Qwen moe models

- Link: https://github.com/sgl-project/sglang/pull/7723
- Status/date: merged / 2025-07-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +18/-0, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] add flashinfer bool check for fusedmoe in Qwen moe models"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[Bug] add flashinfer bool check for fusedmoe in Qwen moe models"; the main implementation surface is `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -143,6 +143,15 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunks: -115,6 +115,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -143,6 +143,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-0 (9 lines); hunks: -115,6 +115,15 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3_moe.py` modified +9/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7745 - [feat] Support EAGLE3 for Qwen

- Link: https://github.com/sgl-project/sglang/pull/7745
- Status/date: merged / 2025-07-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +81/-6, 197 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feat] Support EAGLE3 for Qwen"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`; technical summary: Covers "[feat] Support EAGLE3 for Qwen"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunks: -2,7 +2,7; -325,6 +325,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, set_embed_and_head, touching `__init__, get_input_embeddings, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunks: -18,7 +18,7; -717,6 +717,7 @@ def __init__(; symbols: __init__, forward, start_layer, end_layer, touching `__init__, forward, start_layer`; `python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunks: -440,6 +440,9 @@ def __init__(; -459,6 +462,7 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunks: -293,6 +293,9 @@ def __init__(; -321,7 +324,12 @@ def forward(; symbols: __init__, get_input_embedding, forward, touching `__init__, get_input_embedding, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +28/-2 (30 lines); hunks: -2,7 +2,7; -325,6 +325,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, set_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-2 (27 lines); hunks: -18,7 +18,7; -717,6 +717,7 @@ def __init__(; symbols: __init__, forward, start_layer, end_layer
  - `python/sglang/srt/models/qwen2_moe.py` modified +15/-1 (16 lines); hunks: -440,6 +440,9 @@ def __init__(; -459,6 +462,7 @@ def forward(; symbols: __init__, forward
  - `python/sglang/srt/models/qwen2.py` modified +13/-1 (14 lines); hunks: -293,6 +293,9 @@ def __init__(; -321,7 +324,12 @@ def forward(; symbols: __init__, get_input_embedding, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +28/-2; `python/sglang/srt/models/qwen3_moe.py` modified +25/-2; `python/sglang/srt/models/qwen2_moe.py` modified +15/-1; `python/sglang/srt/models/qwen2.py` modified +13/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7222 - DP Attention with Auto DeepEP Dispatch

- Link: https://github.com/sgl-project/sglang/pull/7222
- Status/date: merged / 2025-07-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +136/-90, 638 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DP Attention with Auto DeepEP Dispatch"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "DP Attention with Auto DeepEP Dispatch"; the main implementation surface is `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunks: -34,7 +34,7; -686,21 +686,21 @@ def dispatch_a(; symbols: dispatch_a, dispatch_b, combine, combine_a, touching `dispatch_a, dispatch_b, combine`; `python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunks: -229,7 +229,7 @@ def forward_deepep(; -240,14 +240,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a, touching `forward_deepep, op_dispatch_a, op_experts`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunks: -558,7 +558,7 @@ def forward_deepep(; -569,14 +569,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a, touching `forward_deepep, op_dispatch_a, op_experts`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3 (8 lines); hunks: -42,7 +42,7; -1178,12 +1178,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13 (28 lines); hunks: -34,7 +34,7; -686,21 +686,21 @@ def dispatch_a(; symbols: dispatch_a, dispatch_b, combine, combine_a
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-9 (16 lines); hunks: -229,7 +229,7 @@ def forward_deepep(; -240,14 +240,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-7 (14 lines); hunks: -558,7 +558,7 @@ def forward_deepep(; -569,14 +569,14 @@ def forward_deepep(; symbols: forward_deepep, op_dispatch_a, op_experts, op_combine_a
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3 (8 lines); hunks: -42,7 +42,7; -1178,12 +1178,14 @@ def forward(; symbols: forward
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +2/-0 (2 lines); hunks: -254,6 +254,7 @@ class ForwardBatch:; -299,6 +300,7 @@ def init_new(; symbols: ForwardBatch, init_new
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +15/-13; `python/sglang/srt/models/qwen3_moe.py` modified +7/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-7; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +5/-3; `python/sglang/srt/model_executor/forward_batch_info.py` modified +2/-0; `python/sglang/srt/two_batch_overlap.py` modified +7/-3
  - tests: `test/srt/test_hybrid_dp_ep_tp_mtp.py` modified +80/-40
- Risk and verification: The diff ships test coverage in `test/srt/test_hybrid_dp_ep_tp_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7762 - feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode

- Link: https://github.com/sgl-project/sglang/pull/7762
- Status/date: merged / 2025-07-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +1006/-9, 1203 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`; technical summary: Covers "feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode"; the main implementation surface is `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: W4AFp8Config, for, __init__, get_name, touching `W4AFp8Config, for, __init__`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunks: -0,0 +1,215; symbols: cutlass_w4a8_moe, touching `cutlass_w4a8_moe`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunks: -12,6 +12,7; -20,6 +21,8; symbols: __init__, internal, touching `__init__, internal`; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunks: -146,6 +146,7 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_i...; -158,9 +159,66 @@ def run_moe_ep_preproess(topk_ids: torch.Tensor, num_expert...; symbols: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess, pre_reorder_triton_kernel_for_cutlass_moe, touching `compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0 (264 lines); hunks: -0,0 +1,264; symbols: W4AFp8Config, for, __init__, get_name
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0 (215 lines); hunks: -0,0 +1,215; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2 (142 lines); hunks: -12,6 +12,7; -20,6 +21,8; symbols: __init__, internal
  - `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0 (58 lines); hunks: -146,6 +146,7 @@ def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_i...; -158,9 +159,66 @@ def run_moe_ep_preproess(topk_ids: torch.Tensor, num_expert...; symbols: compute_seg_indptr_triton_kernel, run_moe_ep_preproess, run_cutlass_moe_ep_preproess, pre_reorder_triton_kernel_for_cutlass_moe
  - `python/sglang/srt/layers/quantization/fp8.py` modified +27/-6 (33 lines); hunks: -1,7 +1,7; -200,7 +200,7 @@ class Fp8LinearMethod(LinearMethodBase):; symbols: Fp8LinearMethod, __init__, create_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` added +264/-0; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` added +215/-0; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +140/-2; `python/sglang/srt/layers/moe/ep_moe/kernels.py` modified +58/-0; `python/sglang/srt/layers/quantization/fp8.py` modified +27/-6; `python/sglang/srt/configs/model_config.py` modified +12/-1
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_cutlass_w4a8_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7634 - [Feature] Layer-wise Prefill

- Link: https://github.com/sgl-project/sglang/pull/7634
- Status/date: merged / 2025-07-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +464/-2, 616 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Layer-wise Prefill"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`; technical summary: Covers "[Feature] Layer-wise Prefill"; the main implementation surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunks: -647,6 +647,69 @@ def forward(; symbols: forward, forward_split_prefill, load_weights, touching `forward, forward_split_prefill, load_weights`; `python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunks: -381,6 +381,57 @@ def forward(; symbols: forward, forward_split_prefill, get_hidden_dim, touching `forward, forward_split_prefill, get_hidden_dim`; `python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunks: -318,6 +318,54 @@ def forward(; symbols: forward, forward_split_prefill, load_weights, touching `forward, forward_split_prefill, load_weights`; `python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunks: -406,6 +406,7 @@ def __init__(; -554,6 +555,49 @@ def forward(; symbols: __init__, forward, forward_split_prefill, start_layer, touching `__init__, forward, forward_split_prefill`.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +63/-0 (63 lines); hunks: -647,6 +647,69 @@ def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/gemma2.py` modified +51/-0 (51 lines); hunks: -381,6 +381,57 @@ def forward(; symbols: forward, forward_split_prefill, get_hidden_dim
  - `python/sglang/srt/models/gemma.py` modified +48/-0 (48 lines); hunks: -318,6 +318,54 @@ def forward(; symbols: forward, forward_split_prefill, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +44/-0 (44 lines); hunks: -406,6 +406,7 @@ def __init__(; -554,6 +555,49 @@ def forward(; symbols: __init__, forward, forward_split_prefill, start_layer
  - `python/sglang/srt/models/qwen3_moe.py` modified +43/-0 (43 lines); hunks: -745,6 +745,49 @@ def forward(; symbols: forward, forward_split_prefill, start_layer
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +63/-0; `python/sglang/srt/models/gemma2.py` modified +51/-0; `python/sglang/srt/models/gemma.py` modified +48/-0; `python/sglang/srt/models/qwen2_moe.py` modified +44/-0; `python/sglang/srt/models/qwen3_moe.py` modified +43/-0; `python/sglang/srt/models/qwen3.py` modified +41/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/managers/schedule_batch.py`, `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7966 - [1/N] MoE Refactor: refactor `select_experts`

- Link: https://github.com/sgl-project/sglang/pull/7966
- Status/date: merged / 2025-07-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 39 files, +557/-872, 2848 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/N] MoE Refactor: refactor `select_experts`"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; technical summary: Covers "[1/N] MoE Refactor: refactor `select_experts`"; the main implementation surface is `python/sglang/srt/layers/quantization/unquant.py`, `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunks: -1,5 +1,7; -21,6 +23,9; symbols: __init__, create_weights, apply, forward_cuda, touching `__init__, create_weights, apply`; `python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunks: -12,12 +12,15; -52,6 +55,168; symbols: TopKOutput, TopK, __init__, forward_native, touching `TopKOutput, TopK, __init__`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunks: -1,15 +1,17; -20,6 +22,12; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method, touching `GPTQMarlinState, CompressedTensorsMoEMethod, __new__`; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunks: -3,7 +3,7; -37,6 +37,9; symbols: get_quant_method, apply, create_weights, touching `get_quant_method, apply, create_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152 (207 lines); hunks: -1,5 +1,7; -21,6 +23,9; symbols: __init__, create_weights, apply, forward_cuda
  - `python/sglang/srt/layers/moe/topk.py` modified +171/-5 (176 lines); hunks: -12,12 +12,15; -52,6 +55,168; symbols: TopKOutput, TopK, __init__, forward_native
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71 (92 lines); hunks: -1,15 +1,17; -20,6 +22,12; symbols: GPTQMarlinState, CompressedTensorsMoEMethod, __new__, get_moe_method
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75 (89 lines); hunks: -3,7 +3,7; -37,6 +37,9; symbols: get_quant_method, apply, create_weights
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74 (87 lines); hunks: -1,17 +1,13; -28,7 +24,7; symbols: __init__, determine_expert_map, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/unquant.py` modified +55/-152; `python/sglang/srt/layers/moe/topk.py` modified +171/-5; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +21/-71; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +14/-75; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +13/-74; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +11/-52
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_block_fp8.py`, `python/sglang/test/test_block_fp8_ep.py`, `python/sglang/test/test_cutlass_w4a8_moe.py`, `python/sglang/test/test_fp4_moe.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7312 - Add get_hidden_dim to qwen3.py for correct lora

- Link: https://github.com/sgl-project/sglang/pull/7312
- Status/date: merged / 2025-07-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `877e35d7754c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +240/-2, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_hidden_dim to qwen3.py for correct lora"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Add get_hidden_dim to qwen3.py for correct lora"; the main implementation surface is `test/srt/models/lora/test_lora_qwen3.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models, touching `TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models`; `python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunks: -330,6 +330,30 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward, touching `__init__, get_input_embeddings, get_hidden_dim`.
- Code diff details:
  - `test/srt/models/lora/test_lora_qwen3.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: TestLoRA, _run_lora_multiple_batch_on_model_cases, test_ci_lora_models, test_all_lora_models
  - `python/sglang/srt/models/qwen3.py` modified +24/-0 (24 lines); hunks: -330,6 +330,30 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/srt/models/lora/test_lora_qwen3.py` added +209/-0
  - runtime: `python/sglang/srt/models/qwen3.py` modified +24/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/srt/models/lora/test_lora.py`, `test/srt/models/lora/test_lora_qwen3.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8280 - DP Enhancement

- Link: https://github.com/sgl-project/sglang/pull/8280
- Status/date: merged / 2025-07-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +665/-1116, 3002 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "DP Enhancement"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`; technical summary: Covers "DP Enhancement"; the main implementation surface is `python/sglang/srt/model_executor/forward_batch_info.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunks: -38,6 +38,11; -48,6 +53,7; symbols: ForwardBatch, init_new, touching `ForwardBatch, init_new`; `python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunks: -3,7 +3,8; -30,6 +31,34; symbols: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode, touching `DPPaddingMode, is_max_len, is_sum_len`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunks: -29,9 +29,9; -167,8 +167,15 @@ def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: get_batch_sizes_to_capture, __init__, can_run, capture_one_batch_size, touching `get_batch_sizes_to_capture, __init__, can_run`; `python/sglang/srt/layers/logits_processor.py` modified +34/-24 (58 lines); hunks: -27,7 +27,9; -111,7 +113,8 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata, touching `LogitsMetadata, from_forward_batch, compute_dp_attention_metadata`.
- Code diff details:
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22 (215 lines); hunks: -38,6 +38,11; -48,6 +53,7; symbols: ForwardBatch, init_new
  - `python/sglang/srt/layers/dp_attention.py` modified +72/-24 (96 lines); hunks: -3,7 +3,8; -30,6 +31,34; symbols: DPPaddingMode, is_max_len, is_sum_len, get_dp_padding_mode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25 (86 lines); hunks: -29,9 +29,9; -167,8 +167,15 @@ def get_batch_sizes_to_capture(model_runner: ModelRunner):; symbols: get_batch_sizes_to_capture, __init__, can_run, capture_one_batch_size
  - `python/sglang/srt/layers/logits_processor.py` modified +34/-24 (58 lines); hunks: -27,7 +27,9; -111,7 +113,8 @@ class LogitsMetadata:; symbols: LogitsMetadata, from_forward_batch, compute_dp_attention_metadata
  - `python/sglang/srt/model_executor/model_runner.py` modified +21/-4 (25 lines); hunks: -1462,9 +1462,13 @@ def apply_torch_tp(self):; -1576,8 +1580,18 @@ def _forward_raw(; symbols: apply_torch_tp, forward_decode, _forward_raw, _preprocess_logits
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/forward_batch_info.py` modified +193/-22; `python/sglang/srt/layers/dp_attention.py` modified +72/-24; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +61/-25; `python/sglang/srt/layers/logits_processor.py` modified +34/-24; `python/sglang/srt/model_executor/model_runner.py` modified +21/-4; `python/sglang/srt/layers/communicator.py` modified +12/-12
- Risk and verification: The diff ships test coverage in `test/srt/test_deepep_small.py`, `test/srt/test_hybrid_dp_ep_tp_mtp.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8036 - [NVIDIA] Add Flashinfer MoE blockscale fp8 backend

- Link: https://github.com/sgl-project/sglang/pull/8036
- Status/date: merged / 2025-07-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +179/-47, 439 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Add Flashinfer MoE blockscale fp8 backend"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "[NVIDIA] Add Flashinfer MoE blockscale fp8 backend"; the main implementation surface is `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunks: -47,12 +47,17; -64,6 +69,13; symbols: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical, touching `forward, _get_tile_tokens_dim, EPMoE`; `python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunks: -56,7 +56,11; -302,15 +306,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal, touching `__init__, forward_normal_dual_stream, forward_normal`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunks: -75,7 +75,7 @@ def __init__(; -92,16 +92,16 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunks: -711,7 +711,7 @@ def __init__(self, quant_config: ModelOptFp4Config):; -865,7 +865,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: __init__, create_weights, process_weights_after_loading, touching `__init__, create_weights, process_weights_after_loading`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7 (109 lines); hunks: -47,12 +47,17; -64,6 +69,13; symbols: forward, _get_tile_tokens_dim, EPMoE, _weight_loader_physical
  - `python/sglang/srt/models/deepseek_v2.py` modified +44/-20 (64 lines); hunks: -56,7 +56,11; -302,15 +306,19 @@ def __init__(; symbols: __init__, forward_normal_dual_stream, forward_normal
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7 (16 lines); hunks: -75,7 +75,7 @@ def __init__(; -92,16 +92,16 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5 (10 lines); hunks: -711,7 +711,7 @@ def __init__(self, quant_config: ModelOptFp4Config):; -865,7 +865,7 @@ def process_weights_after_loading(self, layer: torch.nn.Modu...; symbols: __init__, create_weights, process_weights_after_loading
  - `python/sglang/srt/models/qwen2_moe.py` modified +2/-2 (4 lines); hunks: -147,10 +147,10 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +102/-7; `python/sglang/srt/models/deepseek_v2.py` modified +44/-20; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +9/-7; `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +5/-5; `python/sglang/srt/models/qwen2_moe.py` modified +2/-2; `python/sglang/srt/models/qwen3_moe.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/modelopt_quant.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8421 - [3/N] MoE Refactor: Simplify DeepEP Output

- Link: https://github.com/sgl-project/sglang/pull/8421
- Status/date: merged / 2025-07-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +319/-276, 862 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[3/N] MoE Refactor: Simplify DeepEP Output"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[3/N] MoE Refactor: Simplify DeepEP Output"; the main implementation surface is `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunks: -1,7 +1,27; -24,7 +44,6; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, DeepEPDispatchMode, touching `DeepEPNormalOutput, format, DeepEPLLOutput`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunks: -1,5 +1,7; -50,6 +52,13; symbols: __init__, forward, dispatch, moe_impl, touching `__init__, forward, dispatch`; `python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunks: -144,19 +144,6 @@ def __init__(; -207,41 +194,12 @@ def forward_deepep(; symbols: __init__, forward, forward_deepep, op_gate, touching `__init__, forward, forward_deepep`; `python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunks: -594,41 +594,13 @@ def forward_deepep(; -689,8 +661,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b, touching `forward_deepep, op_select_experts, op_dispatch_a`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118 (187 lines); hunks: -1,7 +1,27; -24,7 +44,6; symbols: DeepEPNormalOutput, format, DeepEPLLOutput, DeepEPDispatchMode
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30 (180 lines); hunks: -1,5 +1,7; -50,6 +52,13; symbols: __init__, forward, dispatch, moe_impl
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-69 (81 lines); hunks: -144,19 +144,6 @@ def __init__(; -207,41 +194,12 @@ def forward_deepep(; symbols: __init__, forward, forward_deepep, op_gate
  - `python/sglang/srt/models/deepseek_v2.py` modified +13/-56 (69 lines); hunks: -594,41 +594,13 @@ def forward_deepep(; -689,8 +661,7 @@ def op_select_experts(self, state):; symbols: forward_deepep, op_select_experts, op_dispatch_a, op_dispatch_b
  - `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0 (48 lines); hunks: -0,0 +1,48; symbols: DispatchOutputFormat, is_standard, is_deepep_normal, is_deepep_ll
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py` modified +69/-118; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +150/-30; `python/sglang/srt/models/qwen3_moe.py` modified +12/-69; `python/sglang/srt/models/deepseek_v2.py` modified +13/-56; `python/sglang/srt/layers/moe/token_dispatcher/base_dispatcher.py` added +48/-0; `python/sglang/srt/layers/moe/token_dispatcher/standard.py` added +19/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/ep_moe/token_dispatcher.py`, `python/sglang/srt/layers/moe/token_dispatcher/__init__.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8448 - Support EPLB in FusedMoE

- Link: https://github.com/sgl-project/sglang/pull/8448
- Status/date: merged / 2025-07-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +107/-11, 407 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support EPLB in FusedMoE"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/glm4_moe.py`; technical summary: Covers "Support EPLB in FusedMoE"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/models/glm4_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunks: -11,6 +11,7; -62,8 +63,9 @@ def __init__(; symbols: __init__, weight_loader, _weight_loader_physical, touching `__init__, weight_loader, _weight_loader_physical`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunks: -183,6 +183,7 @@ def __init__(; -196,6 +197,7 @@ def __init__(; symbols: __init__, weight_loader, touching `__init__, weight_loader`; `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -434,6 +434,7 @@ def __init__(; -740,10 +741,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts, touching `__init__, determine_num_fused_shared_experts`; `python/sglang/srt/models/granitemoe.py` modified +3/-0 (3 lines); hunks: -43,6 +43,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1 (45 lines); hunks: -11,6 +11,7; -62,8 +63,9 @@ def __init__(; symbols: __init__, weight_loader, _weight_loader_physical
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3 (19 lines); hunks: -183,6 +183,7 @@ def __init__(; -196,6 +197,7 @@ def __init__(; symbols: __init__, weight_loader
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: -434,6 +434,7 @@ def __init__(; -740,10 +741,11 @@ def determine_num_fused_shared_experts(; symbols: __init__, determine_num_fused_shared_experts
  - `python/sglang/srt/models/granitemoe.py` modified +3/-0 (3 lines); hunks: -43,6 +43,7 @@ def __init__(; -71,6 +72,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/grok.py` modified +3/-0 (3 lines); hunks: -78,6 +78,7 @@ class Grok1MoE(nn.Module):; -128,6 +129,7 @@ def __init__(; symbols: Grok1MoE, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +44/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +16/-3; `python/sglang/srt/models/glm4_moe.py` modified +3/-1; `python/sglang/srt/models/granitemoe.py` modified +3/-0; `python/sglang/srt/models/grok.py` modified +3/-0; `python/sglang/srt/models/llama4.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/eplb/expert_distribution.py`, `python/sglang/srt/eplb/expert_location.py`, `python/sglang/srt/eplb/expert_location_dispatch.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8450 - [NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE

- Link: https://github.com/sgl-project/sglang/pull/8450
- Status/date: merged / 2025-08-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +131/-46, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; technical summary: Covers "[NVIDIA] Enable Flashinfer MoE blockscale fp8 backend for TP MoE"; the main implementation surface is `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunks: -1,10 +1,13; -33,6 +36,15; symbols: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping, touching `should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunks: -25,14 +25,22; -49,7 +57,6; symbols: _get_tile_tokens_dim, EPMoE, __init__, forward, touching `_get_tile_tokens_dim, EPMoE, __init__`; `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunks: -72,6 +72,7 @@ def dummy_func(*args, **kwargs):; -490,6 +491,16 @@ def apply(; symbols: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod, touching `dummy_func, apply, get_tile_tokens_dim`; `python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunks: -59,7 +59,7; -317,7 +317,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1 (55 lines); hunks: -1,10 +1,13; -33,6 +36,15; symbols: should_use_flashinfer_trtllm_moe, FusedMoeWeightScaleSupported, _weight_loader_impl, make_expert_input_scale_params_mapping
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34 (53 lines); hunks: -25,14 +25,22; -49,7 +57,6; symbols: _get_tile_tokens_dim, EPMoE, __init__, forward
  - `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0 (52 lines); hunks: -72,6 +72,7 @@ def dummy_func(*args, **kwargs):; -490,6 +491,16 @@ def apply(; symbols: dummy_func, apply, get_tile_tokens_dim, Fp8MoEMethod
  - `python/sglang/srt/models/deepseek_v2.py` modified +3/-4 (7 lines); hunks: -59,7 +59,7; -317,7 +317,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-3 (6 lines); hunks: -52,7 +52,7; -426,7 +426,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +54/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +19/-34; `python/sglang/srt/layers/quantization/fp8.py` modified +52/-0; `python/sglang/srt/models/deepseek_v2.py` modified +3/-4; `python/sglang/srt/models/glm4_moe.py` modified +3/-3; `python/sglang/srt/server_args.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/quantization/fp8.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8658 - [5/N] MoE Refactor: Update MoE parallelism arguments

- Link: https://github.com/sgl-project/sglang/pull/8658
- Status/date: merged / 2025-08-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +342/-299, 1748 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[5/N] MoE Refactor: Update MoE parallelism arguments"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`; technical summary: Covers "[5/N] MoE Refactor: Update MoE parallelism arguments"; the main implementation surface is `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunks: -1,28 +1,17; -31,11 +20,9; symbols: __init__, forward, get_moe_impl_class, touching `__init__, forward, get_moe_impl_class`; `python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: MoeA2ABackend, _missing_, is_deepep, is_standard, touching `MoeA2ABackend, _missing_, is_deepep`; `python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunks: -29,6 +29,7; -61,7 +62,6; symbols: __init__, get_moe_weights, touching `__init__, get_moe_weights`; `python/sglang/srt/models/glm4_moe.py` modified +10/-15 (25 lines); hunks: -23,6 +23,7; -50,7 +51,6; symbols: __init__, Glm4MoeDecoderLayer, touching `__init__, Glm4MoeDecoderLayer`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35 (44 lines); hunks: -1,28 +1,17; -31,11 +20,9; symbols: __init__, forward, get_moe_impl_class
  - `python/sglang/srt/layers/moe/utils.py` added +43/-0 (43 lines); hunks: -0,0 +1,43; symbols: MoeA2ABackend, _missing_, is_deepep, is_standard
  - `python/sglang/srt/models/deepseek_v2.py` modified +10/-15 (25 lines); hunks: -29,6 +29,7; -61,7 +62,6; symbols: __init__, get_moe_weights
  - `python/sglang/srt/models/glm4_moe.py` modified +10/-15 (25 lines); hunks: -23,6 +23,7; -50,7 +51,6; symbols: __init__, Glm4MoeDecoderLayer
  - `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +23/-0 (23 lines); hunks: -0,0 +1,23
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +9/-35; `python/sglang/srt/layers/moe/utils.py` added +43/-0; `python/sglang/srt/models/deepseek_v2.py` modified +10/-15; `python/sglang/srt/models/glm4_moe.py` modified +10/-15; `python/sglang/srt/layers/moe/token_dispatcher/__init__.py` modified +23/-0; `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` renamed +8/-15
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/srt/test_deepep_large.py`, `test/srt/test_deepep_small.py`, `test/srt/test_eplb.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #8751 - [1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead

- Link: https://github.com/sgl-project/sglang/pull/8751
- Status/date: merged / 2025-08-06
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `89588179cfe4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +26/-6, 65 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[1/3] Optimize Slime Update Weights: Remove QWen3MOE Load Weight Overhead"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunks: -766,7 +766,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -805,11 +808,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location, touching `load_weights, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +26/-6 (32 lines); hunks: -766,7 +766,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; -805,11 +808,22 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +26/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8753 - [2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights

- Link: https://github.com/sgl-project/sglang/pull/8753
- Status/date: merged / 2025-08-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/eplb/expert_location.py`; technical summary: Covers "[2/3] Optimize Slime Update Weights: Avoid GPU-to-CPU Device Sync when update expert weights"; the main implementation surface is `python/sglang/srt/eplb/expert_location.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/eplb/expert_location.py` modified +5/-1 (6 lines); hunks: -35,6 +35,7 @@ class ExpertLocationMetadata:; -221,6 +222,7 @@ def _init_raw(; symbols: ExpertLocationMetadata, _init_raw, update, logical_to_all_physical, touching `ExpertLocationMetadata, _init_raw, update`.
- Code diff details:
  - `python/sglang/srt/eplb/expert_location.py` modified +5/-1 (6 lines); hunks: -35,6 +35,7 @@ class ExpertLocationMetadata:; -221,6 +222,7 @@ def _init_raw(; symbols: ExpertLocationMetadata, _init_raw, update, logical_to_all_physical
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/eplb/expert_location.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/eplb/expert_location.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8987 - Fix incorrect default get_hidden_dim logic

- Link: https://github.com/sgl-project/sglang/pull/8987
- Status/date: merged / 2025-08-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +36/-143, 236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix incorrect default get_hidden_dim logic"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/granite.py`, `python/sglang/srt/models/llama.py`; technical summary: Covers "Fix incorrect default get_hidden_dim logic"; the main implementation surface is `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/granite.py`, `python/sglang/srt/models/llama.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunks: -432,40 +432,6 @@ def forward_split_prefill(; symbols: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size, touching `forward_split_prefill, get_hidden_dim, get_module_name`; `python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunks: -363,31 +363,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name, touching `forward, get_hidden_dim, get_module_name`; `python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunks: -532,31 +532,6 @@ def end_layer(self):; symbols: end_layer, get_input_embeddings, get_hidden_dim, get_module_name, touching `end_layer, get_input_embeddings, get_hidden_dim`; `python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunks: -330,30 +330,6 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward, touching `__init__, get_input_embeddings, get_hidden_dim`.
- Code diff details:
  - `python/sglang/srt/models/gemma2.py` modified +0/-34 (34 lines); hunks: -432,40 +432,6 @@ def forward_split_prefill(; symbols: forward_split_prefill, get_hidden_dim, get_module_name, get_attention_sliding_window_size
  - `python/sglang/srt/models/granite.py` modified +0/-25 (25 lines); hunks: -363,31 +363,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
  - `python/sglang/srt/models/llama.py` modified +0/-25 (25 lines); hunks: -532,31 +532,6 @@ def end_layer(self):; symbols: end_layer, get_input_embeddings, get_hidden_dim, get_module_name
  - `python/sglang/srt/models/qwen3.py` modified +0/-24 (24 lines); hunks: -330,30 +330,6 @@ def __init__(; symbols: __init__, get_input_embeddings, get_hidden_dim, forward
  - `python/sglang/srt/models/torch_native_llama.py` modified +0/-24 (24 lines); hunks: -416,30 +416,6 @@ def forward(; symbols: forward, get_hidden_dim, get_module_name, get_module_name_from_weight_name
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma2.py` modified +0/-34; `python/sglang/srt/models/granite.py` modified +0/-25; `python/sglang/srt/models/llama.py` modified +0/-25; `python/sglang/srt/models/qwen3.py` modified +0/-24; `python/sglang/srt/models/torch_native_llama.py` modified +0/-24; `python/sglang/srt/models/gemma3n_mm.py` modified +12/-6
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/lora/utils.py`, `python/sglang/srt/models/gemma2.py`, `python/sglang/srt/models/gemma3n_mm.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9014 - Fuse writing KV buffer into rope kernel (part 2: srt)

- Link: https://github.com/sgl-project/sglang/pull/9014
- Status/date: merged / 2025-08-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +65/-6, 147 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse writing KV buffer into rope kernel (part 2: srt)"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py`; technical summary: Covers "Fuse writing KV buffer into rope kernel (part 2: srt)"; the main implementation surface is `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/entrypoints/engine.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg, touching `GptOssConfig, forward_normal, _enable_fused_set_kv_buffer`; `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`; `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config, touching `_set_envs_and_config`; `.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/models/gpt_oss.py` modified +51/-2 (53 lines); hunks: -66,10 +66,15; -196,6 +201,32 @@ def forward_normal(; symbols: GptOssConfig, forward_normal, _enable_fused_set_kv_buffer, _create_fused_set_kv_buffer_arg
  - `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0 (10 lines); hunks: -222,6 +222,7 @@ def forward_cuda(; -231,8 +232,17 @@ def forward_cuda(; symbols: forward_cuda
  - `python/sglang/srt/entrypoints/engine.py` modified +1/-1 (2 lines); hunks: -655,7 +655,7 @@ def _set_envs_and_config(server_args: ServerArgs):; symbols: _set_envs_and_config
  - `.github/workflows/pr-test-pd-router.yml` modified +1/-1 (2 lines); hunks: -119,7 +119,7 @@ jobs:
  - `docker/Dockerfile.gb200` modified +1/-1 (2 lines); hunks: -64,7 +64,7 @@ RUN python3 -m pip install --no-cache-dir --upgrade pip setupt...
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/gpt_oss.py` modified +51/-2; `python/sglang/srt/layers/rotary_embedding.py` modified +10/-0; `python/sglang/srt/entrypoints/engine.py` modified +1/-1; `python/pyproject.toml` modified +1/-1
  - ci: `.github/workflows/pr-test-pd-router.yml` modified +1/-1
  - other: `docker/Dockerfile.gb200` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/pyproject.toml`, `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9101 - Feature: support qwen and llama4 reducescatter for dp attention padding

- Link: https://github.com/sgl-project/sglang/pull/9101
- Status/date: merged / 2025-08-14
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +68/-16, 210 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Feature: support qwen and llama4 reducescatter for dp attention padding"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`; technical summary: Covers "Feature: support qwen and llama4 reducescatter for dp attention padding"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunks: -144,11 +144,14 @@ def __init__(; -159,15 +162,19 @@ def get_moe_weights(self):; symbols: __init__, forward, get_moe_weights, forward_normal, touching `__init__, forward, get_moe_weights`; `python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunks: -107,10 +107,14 @@ def __init__(; -175,7 +179,10 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer, touching `__init__, forward, _is_moe_layer`; `python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunks: -91,10 +91,18 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +18/-5 (23 lines); hunks: -144,11 +144,14 @@ def __init__(; -159,15 +162,19 @@ def get_moe_weights(self):; symbols: __init__, forward, get_moe_weights, forward_normal
  - `python/sglang/srt/models/qwen2_moe.py` modified +18/-4 (22 lines); hunks: -107,10 +107,14 @@ def __init__(; -175,7 +179,10 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/models/llama4.py` modified +16/-3 (19 lines); hunks: -131,14 +131,19 @@ def __init__(; -412,6 +417,7 @@ def __init__(; symbols: __init__, forward, _is_moe_layer
  - `python/sglang/srt/models/llama.py` modified +10/-2 (12 lines); hunks: -91,10 +91,18 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/lora/layers.py` modified +6/-2 (8 lines); hunks: -253,7 +253,7 @@ def apply_lora(self, base_output: torch.Tensor, x: torch.Ten...; -270,7 +270,11 @@ def forward(self, input_: torch.Tensor):; symbols: apply_lora, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +18/-5; `python/sglang/srt/models/qwen2_moe.py` modified +18/-4; `python/sglang/srt/models/llama4.py` modified +16/-3; `python/sglang/srt/models/llama.py` modified +10/-2; `python/sglang/srt/lora/layers.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/lora/layers.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #8118 - [feat] Support tp mode for DeepSeek-R1-W4AFP8

- Link: https://github.com/sgl-project/sglang/pull/8118
- Status/date: merged / 2025-09-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +291/-120, 710 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[feat] Support tp mode for DeepSeek-R1-W4AFP8"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "[feat] Support tp mode for DeepSeek-R1-W4AFP8"; the main implementation surface is `python/sglang/srt/layers/quantization/w4afp8.py`, `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunks: -1,12 +1,14; -91,21 +93,38 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales, touching `get_quant_method, get_scaled_act_names, W4AFp8MoEMethod`; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9 (10 lines); hunks: -91,18 +91,10 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe, touching `cutlass_w4a8_moe`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2 (7 lines); hunks: -175,6 +175,8 @@ def __init__(; -593,8 +595,9 @@ def _weight_loader_impl(; symbols: __init__, _weight_loader_impl, touching `__init__, _weight_loader_impl`; `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights, touching `determine_num_fused_shared_experts, load_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25 (55 lines); hunks: -1,12 +1,14; -91,21 +93,38 @@ def get_quant_method(; symbols: get_quant_method, get_scaled_act_names, W4AFp8MoEMethod, interleave_scales
  - `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9 (10 lines); hunks: -91,18 +91,10 @@ def cutlass_w4a8_moe(; symbols: cutlass_w4a8_moe
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2 (7 lines); hunks: -175,6 +175,8 @@ def __init__(; -593,8 +595,9 @@ def _weight_loader_impl(; symbols: __init__, _weight_loader_impl
  - `python/sglang/srt/models/deepseek_v2.py` modified +5/-0 (5 lines); hunks: -2168,6 +2168,8 @@ def determine_num_fused_shared_experts(; -2471,6 +2473,9 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/configs/model_config.py` modified +2/-1 (3 lines); hunks: -393,9 +393,10 @@ def _parse_quant_hf_config(self):; symbols: _parse_quant_hf_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w4afp8.py` modified +30/-25; `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py` modified +1/-9; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-2; `python/sglang/srt/models/deepseek_v2.py` modified +5/-0; `python/sglang/srt/configs/model_config.py` modified +2/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +0/-3
  - other: `sgl-kernel/csrc/moe/cutlass_moe/w4a8/w4a8_grouped_mm_c3x.cu` modified +206/-60
  - tests: `python/sglang/test/test_cutlass_w4a8_moe.py` modified +24/-9
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_cutlass_w4a8_moe.py`, `sgl-kernel/tests/test_cutlass_w4a8_moe_mm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #7912 - Qwen FP8/NVFP4 ModelOPT Quantization support

- Link: https://github.com/sgl-project/sglang/pull/7912
- Status/date: merged / 2025-09-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +43/-4, 82 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Qwen FP8/NVFP4 ModelOPT Quantization support"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Qwen FP8/NVFP4 ModelOPT Quantization support"; the main implementation surface is `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunks: -517,6 +517,39 @@ def get_min_capability(cls) -> int:; -549,7 +582,7 @@ def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4C...; symbols: get_min_capability, get_config_filenames, common_group_size, from_config, touching `get_min_capability, get_config_filenames, common_group_size`; `python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunks: -24,7 +24,10; -458,7 +461,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2 (37 lines); hunks: -517,6 +517,39 @@ def get_min_capability(cls) -> int:; -549,7 +582,7 @@ def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4C...; symbols: get_min_capability, get_config_filenames, common_group_size, from_config
  - `python/sglang/srt/models/qwen3.py` modified +8/-2 (10 lines); hunks: -24,7 +24,10; -458,7 +461,10 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/modelopt_quant.py` modified +35/-2; `python/sglang/srt/models/qwen3.py` modified +8/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/quantization/modelopt_quant.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9973 - Optimize Qwen3-moe model by using flashinfer fused allreduce

- Link: https://github.com/sgl-project/sglang/pull/9973
- Status/date: merged / 2025-09-04
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `ec15c8360e73`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +52/-12, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize Qwen3-moe model by using flashinfer fused allreduce"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Optimize Qwen3-moe model by using flashinfer fused allreduce"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunks: -42,7 +42,10; -57,10 +60,17; symbols: forward, get_moe_weights, forward_normal, __init__, touching `forward, get_moe_weights, forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +39/-8 (47 lines); hunks: -42,7 +42,10; -57,10 +60,17; symbols: forward, get_moe_weights, forward_normal, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +39/-8
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #9338 - Refactor TopK to ensure readability and extensibility

- Link: https://github.com/sgl-project/sglang/pull/9338
- Status/date: merged / 2025-09-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 14 files, +52/-47, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Refactor TopK to ensure readability and extensibility"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "Refactor TopK to ensure readability and extensibility"; the main implementation surface is `python/sglang/srt/layers/moe/topk.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native, touching `TopKConfig, __init__, forward_native`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim, touching `_is_fp4_quantization_enabled, selection, _get_tile_tokens_dim`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class, touching `_forward_ll, get_moe_impl_class`.
- Code diff details:
  - `python/sglang/srt/layers/moe/topk.py` modified +30/-9 (39 lines); hunks: -19,6 +19,7; -51,6 +52,9; symbols: TopKConfig, __init__, forward_native
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-12 (19 lines); hunks: -65,14 +65,10; -375,21 +371,20 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10 (10 lines); hunks: -74,16 +74,6; symbols: _is_fp4_quantization_enabled, selection, _get_tile_tokens_dim
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4 (8 lines); hunks: -888,7 +888,7 @@ def _forward_ll(dispatch_output: DeepEPLLOutput):; -901,8 +901,7 @@ def get_moe_impl_class(quant_config: Optional[QuantizationCo...; symbols: _forward_ll, get_moe_impl_class
  - `python/sglang/srt/models/longcat_flash.py` modified +2/-2 (4 lines); hunks: -260,7 +260,7 @@ def __init__(; -853,7 +853,7 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Te...; symbols: __init__, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/topk.py` modified +30/-9; `python/sglang/srt/models/deepseek_v2.py` modified +7/-12; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +0/-10; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +4/-4; `python/sglang/srt/models/longcat_flash.py` modified +2/-2; `python/sglang/srt/models/qwen3_next.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`, `python/sglang/srt/layers/moe/topk.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10574 - [Ascend]optimize Qwen3 on Ascend

- Link: https://github.com/sgl-project/sglang/pull/10574
- Status/date: merged / 2025-09-23
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `e22f3a5ec914`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +81/-2, 170 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Ascend]optimize Qwen3 on Ascend"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Ascend]optimize Qwen3 on Ascend"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunks: -30,12 +30,19; -235,9 +242,18 @@ def forward(; symbols: Qwen3Attention, forward, touching `Qwen3Attention, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +18/-2 (20 lines); hunks: -30,12 +30,19; -235,9 +242,18 @@ def forward(; symbols: Qwen3Attention, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +18/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/quantization/w8a8_int8.py`, `python/sglang/srt/model_executor/model_runner.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10749 - Fuse write kv buffer into rope for qwen3 moe & bailing moe

- Link: https://github.com/sgl-project/sglang/pull/10749
- Status/date: merged / 2025-09-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `a5095d62623f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +105/-34, 207 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fuse write kv buffer into rope for qwen3 moe & bailing moe"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core, touching `forward_prepare, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +22/-2 (24 lines); hunks: -60,6 +60,10; -412,15 +416,31 @@ def forward_prepare(; symbols: forward_prepare, forward_core
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +22/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10975 - Use more general heuristics to set the default value of --mem-fraction-static

- Link: https://github.com/sgl-project/sglang/pull/10975
- Status/date: merged / 2025-09-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +157/-141, 568 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Use more general heuristics to set the default value of --mem-fraction-static"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`; technical summary: Covers "Use more general heuristics to set the default value of --mem-fraction-static"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/server_args.py`, `python/sglang/srt/managers/io_struct.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +0/-1 (1 lines); hunks: -1,6 +1,5; `python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunks: -523,108 +523,142 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes, touching `_handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes`; `python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunks: -35,6 +35,7; -84,8 +85,6 @@ class GenerateReqInput:; symbols: SessionParams, GenerateReqInput, contains_mm_input, __getitem__, touching `SessionParams, GenerateReqInput, contains_mm_input`; `.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunks: -99,8 +99,6 @@ jobs:; -189,8 +187,6 @@ jobs:.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +0/-1 (1 lines); hunks: -1,6 +1,5
  - `python/sglang/srt/server_args.py` modified +116/-82 (198 lines); hunks: -523,108 +523,142 @@ def _handle_missing_default_values(self):; symbols: _handle_missing_default_values, _handle_gpu_memory_settings, _generate_cuda_graph_batch_sizes
  - `python/sglang/srt/managers/io_struct.py` modified +22/-13 (35 lines); hunks: -35,6 +35,7; -84,8 +85,6 @@ class GenerateReqInput:; symbols: SessionParams, GenerateReqInput, contains_mm_input, __getitem__
  - `.github/workflows/pr-test.yml` modified +0/-26 (26 lines); hunks: -99,8 +99,6 @@ jobs:; -189,8 +187,6 @@ jobs:
  - `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10 (20 lines); hunks: -242,11 +242,8 @@ def find_local_hf_snapshot_dir(; -347,11 +344,14 @@ def download_weights_from_hf(; symbols: find_local_hf_snapshot_dir, download_weights_from_hf
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +0/-1; `python/sglang/srt/server_args.py` modified +116/-82; `python/sglang/srt/managers/io_struct.py` modified +22/-13; `python/sglang/srt/model_loader/weight_utils.py` modified +10/-10
  - ci: `.github/workflows/pr-test.yml` modified +0/-26
  - tests: `test/srt/test_multi_instance_release_memory_occupation.py` modified +5/-2; `test/srt/run_suite.py` modified +1/-5; `test/srt/test_mla_deepseek_v3.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/srt/lora/test_lora_llama4.py`, `test/srt/run_suite.py`, `test/srt/test_mla_deepseek_v3.py`, `test/srt/test_multi_instance_release_memory_occupation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #10985 - Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg

- Link: https://github.com/sgl-project/sglang/pull/10985
- Status/date: merged / 2025-10-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `61305291430a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +14/-2, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Quick Fix: fix Qwen3-VL launch failure caused by MRotaryEmbedding arg"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core, touching `__init__, forward_prepare, forward_core`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-2 (12 lines); hunks: -51,7 +51,7; -358,6 +358,10 @@ def __init__(; symbols: __init__, forward_prepare, forward_core
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #10911 - model: qwen3-omni (thinker-only)

- Link: https://github.com/sgl-project/sglang/pull/10911
- Status/date: merged / 2025-10-16
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `86b04d25b3f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +1947/-328, 2837 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "model: qwen3-omni (thinker-only)"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "model: qwen3-omni (thinker-only)"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +2/-1 (3 lines); hunks: -661,13 +661,14 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-1 (3 lines); hunks: -661,13 +661,14 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -661,13 +661,14 @@ def __init__(
+        decoder_layer_type=Qwen3MoeDecoderLayer,
-            decoder_layer_type=Qwen3MoeDecoderLayer,
+            decoder_layer_type=decoder_layer_type,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-1
- Risk and verification: The diff ships test coverage in `test/srt/test_vision_openai_server_a.py`, `test/srt/test_vision_openai_server_b.py`, `test/srt/test_vision_openai_server_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12002 - Eagle3 DP attention for Qwen3 MoE

- Link: https://github.com/sgl-project/sglang/pull/12002
- Status/date: merged / 2025-10-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `750940ae3660`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +219/-27, 372 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Eagle3 DP attention for Qwen3 MoE"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Eagle3 DP attention for Qwen3 MoE"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunks: -539,10 +539,16 @@ def forward(; -774,13 +780,15 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: forward, set_eagle3_layers_to_capture, load_weights, touching `forward, set_eagle3_layers_to_capture, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +16/-8 (24 lines); hunks: -539,10 +539,16 @@ def forward(; -774,13 +780,15 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optional...; symbols: forward, set_eagle3_layers_to_capture, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +16/-8
- Risk and verification: The diff ships test coverage in `python/sglang/test/test_utils.py`, `test/srt/run_suite.py`, `test/srt/test_eagle_dp_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #12543 - Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell

- Link: https://github.com/sgl-project/sglang/pull/12543
- Status/date: merged / 2025-11-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +107/-9, 226 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; technical summary: Covers "Enable Flashinfer TRTLLM-GEN-MoE FP8 blockwise kernel for Qwen3-Next on Blackwell"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/layers/quantization/fp8.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +20/-1 (21 lines); hunks: -2,7 +2,7; -248,3 +248,22 @@ def speculative_moe_backend_context():; symbols: speculative_moe_backend_context, RoutingMethodType, touching `speculative_moe_backend_context, RoutingMethodType`; `python/sglang/srt/layers/quantization/fp8.py` modified +12/-7 (19 lines); hunks: -1193,6 +1193,7 @@ def apply_with_router_logits(; -1204,26 +1205,30 @@ def apply_with_router_logits(; symbols: apply_with_router_logits, touching `apply_with_router_logits`; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1 (6 lines); hunks: -36,6 +36,7; -56,7 +57,7; symbols: __init__, _load_per_tensor_weight_scale, touching `__init__, _load_per_tensor_weight_scale`; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0 (2 lines); hunks: -68,6 +68,7 @@ def __init__(; -81,6 +82,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +20/-1 (21 lines); hunks: -2,7 +2,7; -248,3 +248,22 @@ def speculative_moe_backend_context():; symbols: speculative_moe_backend_context, RoutingMethodType
  - `python/sglang/srt/layers/quantization/fp8.py` modified +12/-7 (19 lines); hunks: -1193,6 +1193,7 @@ def apply_with_router_logits(; -1204,26 +1205,30 @@ def apply_with_router_logits(; symbols: apply_with_router_logits
  - `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1 (6 lines); hunks: -36,6 +36,7; -56,7 +57,7; symbols: __init__, _load_per_tensor_weight_scale
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0 (2 lines); hunks: -68,6 +68,7 @@ def __init__(; -81,6 +82,7 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +2/-0 (2 lines); hunks: -57,6 +57,7; -162,6 +163,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +20/-1; `python/sglang/srt/layers/quantization/fp8.py` modified +12/-7; `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` modified +5/-1; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +2/-0; `python/sglang/srt/models/qwen2_moe.py` modified +2/-0
  - tests: `test/srt/nightly/test_flashinfer_trtllm_gen_moe_backend.py` added +65/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/nightly/test_flashinfer_trtllm_gen_moe_backend.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13489 - Flashinfer TRTLLM-GEN-MoE + Qwen3

- Link: https://github.com/sgl-project/sglang/pull/13489
- Status/date: merged / 2025-11-18
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `92ad2ff9ce0e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +43/-1, 72 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Flashinfer TRTLLM-GEN-MoE + Qwen3"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Flashinfer TRTLLM-GEN-MoE + Qwen3"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -49,6 +49,7; -111,6 +112,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -49,6 +49,7; -111,6 +112,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -49,6 +49,7 @@
+from sglang.srt.layers.moe.utils import RoutingMethodType
@@ -111,6 +112,7 @@ def __init__(
+            routing_method_type=RoutingMethodType.Renormalize,
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12078 - [Ascend] qwen optimization

- Link: https://github.com/sgl-project/sglang/pull/12078
- Status/date: merged / 2025-11-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +561/-108, 998 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Ascend] qwen optimization"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`; technical summary: Covers "[Ascend] qwen optimization"; the main implementation surface is `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunks: -45,6 +45,10; -411,9 +415,142 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__, touching `DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE`; `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunks: -625,53 +625,93 @@ def forward_decode_graph(; symbols: forward_decode_graph, touching `forward_decode_graph`; `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: FuseEPDispatchOutput, format, FuseEPCombineInput, NpuFuseEPDispatcher, touching `FuseEPDispatchOutput, format, FuseEPCombineInput`; `python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunks: -70,6 +70,7; -78,6 +79,10; symbols: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare, touching `Qwen3MoeSparseMoeBlock, forward, op_core`.
- Code diff details:
  - `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0 (137 lines); hunks: -45,6 +45,10; -411,9 +415,142 @@ def npu_fused_moe_without_routing_weights_bf16(; symbols: DeepEPMoE, npu_fused_moe_without_routing_weights_bf16, NpuFuseEPMoE, __init__
  - `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45 (130 lines); hunks: -625,53 +625,93 @@ def forward_decode_graph(; symbols: forward_decode_graph
  - `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0 (97 lines); hunks: -0,0 +1,97; symbols: FuseEPDispatchOutput, format, FuseEPCombineInput, NpuFuseEPDispatcher
  - `python/sglang/srt/models/qwen3_moe.py` modified +56/-4 (60 lines); hunks: -70,6 +70,7; -78,6 +79,10; symbols: Qwen3MoeSparseMoeBlock, forward, op_core, forward_prepare
  - `python/sglang/srt/models/qwen3.py` modified +40/-4 (44 lines); hunks: -44,6 +44,9; -161,6 +164,33 @@ def _apply_qk_norm(; symbols: Qwen3Attention, __init__, _apply_qk_norm, forward_prepare_native
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +137/-0; `python/sglang/srt/layers/attention/ascend_backend.py` modified +85/-45; `python/sglang/srt/layers/moe/token_dispatcher/fuseep.py` added +97/-0; `python/sglang/srt/models/qwen3_moe.py` modified +56/-4; `python/sglang/srt/models/qwen3.py` modified +40/-4; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +29/-11
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/attention/ascend_backend.py`, `python/sglang/srt/layers/moe/ep_moe/layer.py`, `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #12330 - [CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next

- Link: https://github.com/sgl-project/sglang/pull/12330
- Status/date: merged / 2025-12-03
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +218/-0, 241 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `sgl-kernel/csrc/cpu/model/qwen3.cpp`, `test/srt/cpu/test_qwen3.py`, `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp`; technical summary: Covers "[CPU] add fused_qkvzba_split_reshape_cat kernel for Qwen3-next"; the main implementation surface is `sgl-kernel/csrc/cpu/model/qwen3.cpp`, `test/srt/cpu/test_qwen3.py`, `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `sgl-kernel/csrc/cpu/model/qwen3.cpp` added +115/-0 (115 lines); hunks: -0,0 +1,115; `test/srt/cpu/test_qwen3.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat, touching `fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat`; `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` modified +15/-0 (15 lines); hunks: -250,6 +250,15 @@ std::tuple rotary_embedding_cpu(; -389,6 +398,12 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {; `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -337,6 +337,7.
- Code diff details:
  - `sgl-kernel/csrc/cpu/model/qwen3.cpp` added +115/-0 (115 lines); hunks: -0,0 +1,115
  - `test/srt/cpu/test_qwen3.py` added +87/-0 (87 lines); hunks: -0,0 +1,87; symbols: fix_query_key_value_ordering_reshape_cat, TestQwen3, test_fused_qkvzba_split_reshape_cat
  - `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` modified +15/-0 (15 lines); hunks: -250,6 +250,15 @@ std::tuple rotary_embedding_cpu(; -389,6 +398,12 @@ TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -337,6 +337,7
- Key code excerpts:

```diff
diff -- sgl-kernel/csrc/cpu/model/qwen3.cpp
@@ -0,0 +1,115 @@
+#include "common.h"
+#include "vec.h"
+namespace {
+template <typename scalar_t>
+inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
+  using bVec = at::vec::Vectorized<scalar_t>;
diff -- test/srt/cpu/test_qwen3.py
@@ -0,0 +1,87 @@
+import unittest
+import torch
+from utils import precision
+from sglang.test.test_utils import CustomTestCase
+torch.manual_seed(1234)
+def fix_query_key_value_ordering_reshape_cat(
diff -- sgl-kernel/csrc/cpu/torch_extension_cpu.cpp
@@ -250,6 +250,15 @@ std::tuple<at::Tensor, at::Tensor> rotary_embedding_cpu(
```

- Reviewed files:
  - other: `sgl-kernel/csrc/cpu/model/qwen3.cpp` added +115/-0; `sgl-kernel/csrc/cpu/torch_extension_cpu.cpp` modified +15/-0
  - tests: `test/srt/cpu/test_qwen3.py` added +87/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_qwen3.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #14093 - Add fused FP8 KV cache write kernel for TRTLLM MHA backend

- Link: https://github.com/sgl-project/sglang/pull/14093
- Status/date: merged / 2025-12-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +854/-7, 925 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add fused FP8 KV cache write kernel for TRTLLM MHA backend"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Add fused FP8 KV cache write kernel for TRTLLM MHA backend"; the main implementation surface is `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py`, `python/sglang/srt/layers/attention/trtllm_mha_backend.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunks: -0,0 +1,467; symbols: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer, touching `_process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer`; `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunks: -5,6 +5,7; -14,9 +15,12; symbols: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata, touching `get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer`; `python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunks: -422,6 +422,7 @@ def forward_prepare_npu(; -449,6 +450,7 @@ def forward_prepare_native(; symbols: forward_prepare_npu, forward_prepare_native, forward_core, touching `forward_prepare_npu, forward_prepare_native, forward_core`; `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache, touching `TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness`.
- Code diff details:
  - `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0 (467 lines); hunks: -0,0 +1,467; symbols: _process_kv_tensor, _fused_fp8_set_kv_buffer_kernel, fused_fp8_set_kv_buffer, _naive_fp8_set_kv_buffer
  - `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6 (78 lines); hunks: -5,6 +5,7; -14,9 +15,12; symbols: get_cuda_graph_seq_len_fill_value, _should_use_fused_fp8_path, _fused_fp8_set_kv_buffer, init_forward_metadata
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-1 (10 lines); hunks: -422,6 +422,7 @@ def forward_prepare_npu(; -449,6 +450,7 @@ def forward_prepare_native(; symbols: forward_prepare_npu, forward_prepare_native, forward_core
  - `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0 (306 lines); hunks: -0,0 +1,306; symbols: TestTRTLLMFP8KVKernel, setUpClass, _test_kernel_correctness, test_basic_3d_input_3d_cache
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/trtllm_fp8_kv_kernel.py` added +467/-0; `python/sglang/srt/layers/attention/trtllm_mha_backend.py` modified +72/-6; `python/sglang/srt/models/qwen3_moe.py` modified +9/-1
  - tests: `test/manual/test_trtllm_fp8_kv_kernel.py` added +306/-0
- Risk and verification: The diff ships test coverage in `test/manual/test_trtllm_fp8_kv_kernel.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #13998 - [apply][2/2] Fused qk_norm_rope for Qwen3-MoE

- Link: https://github.com/sgl-project/sglang/pull/13998
- Status/date: merged / 2025-12-07
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `26d95008b65b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +199/-22, 317 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[apply][2/2] Fused qk_norm_rope for Qwen3-MoE"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[apply][2/2] Fused qk_norm_rope for Qwen3-MoE"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunks: -18,10 +18,12; -73,6 +75,13; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range, touching `compute_yarn_parameters, get_mscale, find_correction_dim`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +193/-22 (215 lines); hunks: -18,10 +18,12; -73,6 +75,13; symbols: compute_yarn_parameters, get_mscale, find_correction_dim, find_correction_range
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +193/-22
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/server_args.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #11984 - [Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix

- Link: https://github.com/sgl-project/sglang/pull/11984
- Status/date: closed / 2025-12-15
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 17 files, +834/-95, 1206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/layers/quantization/w4a4_int4.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`; technical summary: Covers "[Ascend]quantization: w4a4, compressed tensors, NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix"; the main implementation surface is `python/sglang/srt/layers/quantization/w4a4_int4.py`, `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: W4A4Int4Config, for, __init__, get_supported_act_dtypes, touching `W4A4Int4Config, for, __init__`; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68 (159 lines); hunks: -7,6 +7,8; -17,8 +19,9; symbols: get_min_capability, process_weights_after_loading, create_weights, touching `get_min_capability, process_weights_after_loading, create_weights`; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1 (124 lines); hunks: -11,6 +11,8; -26,7 +28,7; symbols: GPTQMarlinState, get_moe_method, apply, CompressedTensorsW8A8Int8MoEMethod, touching `GPTQMarlinState, get_moe_method, apply`; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8 (33 lines); hunks: -64,6 +64,7; -730,8 +731,11 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod, touching `process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0 (284 lines); hunks: -0,0 +1,284; symbols: W4A4Int4Config, for, __init__, get_supported_act_dtypes
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68 (159 lines); hunks: -7,6 +7,8; -17,8 +19,9; symbols: get_min_capability, process_weights_after_loading, create_weights
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1 (124 lines); hunks: -11,6 +11,8; -26,7 +28,7; symbols: GPTQMarlinState, get_moe_method, apply, CompressedTensorsW8A8Int8MoEMethod
  - `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8 (33 lines); hunks: -64,6 +64,7; -730,8 +731,11 @@ def process_weights_after_loading(self, layer):; symbols: process_weights_after_loading, NPU_W8A8LinearMethodMTImpl, NPU_W8A8DynamicLinearMethod
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -66,6 +66,7; -947,14 +948,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: load_weights, get_model_config_for_expert_location
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/w4a4_int4.py` added +284/-0; `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` modified +91/-68; `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +123/-1; `python/sglang/srt/layers/quantization/w8a8_int8.py` modified +25/-8; `python/sglang/srt/models/qwen3_moe.py` modified +10/-7; `python/sglang/srt/layers/quantization/unquant.py` modified +14/-0
- Risk and verification: The diff ships test coverage in `test/srt/ascend/test_ascend_memory_consumption.py`, `test/srt/ascend/test_ascend_w4a4_quantization.py`, `test/srt/ascend/test_ascend_w8a8_quantization.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15223 - [bug fix][pp] fix qwen3 model load

- Link: https://github.com/sgl-project/sglang/pull/15223
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `e9abb52576ea`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bug fix][pp] fix qwen3 model load"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[bug fix][pp] fix qwen3 model load"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunks: -392,13 +392,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +3/-3 (6 lines); hunks: -392,13 +392,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15390 - [NPU]qwen3 pp bugfix

- Link: https://github.com/sgl-project/sglang/pull/15390
- Status/date: merged / 2025-12-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `8bf7f240b654`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-3, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]qwen3 pp bugfix"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[NPU]qwen3 pp bugfix"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunks: -167,10 +167,10 @@ def forward_prepare_native(self, positions, hidden_states):; -205,6 +205,7 @@ def forward(; symbols: forward_prepare_native, forward_prepare_npu, forward, touching `forward_prepare_native, forward_prepare_npu, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -542,7 +542,7 @@ def forward_prepare_npu(; symbols: forward_prepare_npu, touching `forward_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +3/-2 (5 lines); hunks: -167,10 +167,10 @@ def forward_prepare_native(self, positions, hidden_states):; -205,6 +205,7 @@ def forward(; symbols: forward_prepare_native, forward_prepare_npu, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-1 (2 lines); hunks: -542,7 +542,7 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +3/-2; `python/sglang/srt/models/qwen3_moe.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15835 - [Feature] JIT Fused QK norm + qk norm clean up

- Link: https://github.com/sgl-project/sglang/pull/15835
- Status/date: merged / 2025-12-28
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 15 files, +827/-127, 1151 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] JIT Fused QK norm + qk norm clean up"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Feature] JIT Fused QK norm + qk norm clean up"; the main implementation surface is `python/sglang/srt/models/utils.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm, touching `create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm`; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope, touching `__init__, _apply_qk_norm, op_prepare`; `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native, touching `__init__, _apply_qk_norm, forward_prepare_native`; `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward, touching `__init__, _apply_qk_norm, forward`.
- Code diff details:
  - `python/sglang/srt/models/utils.py` modified +80/-5 (85 lines); hunks: -11,24 +11,27; -113,6 +116,8 @@ def create_fused_set_kv_buffer_arg(; symbols: create_fused_set_kv_buffer_arg, rot_pos_ids, apply_qk_norm
  - `python/sglang/srt/models/qwen3_moe.py` modified +9/-27 (36 lines); hunks: -57,12 +57,12; -498,31 +498,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, apply_qk_norm_rope
  - `python/sglang/srt/models/qwen3.py` modified +9/-24 (33 lines); hunks: -21,14 +21,14; -138,32 +138,17 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward_prepare_native
  - `python/sglang/srt/models/bailing_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -507,28 +508,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, forward
  - `python/sglang/srt/models/glm4_moe.py` modified +9/-23 (32 lines); hunks: -75,6 +75,7; -250,28 +251,6 @@ def __init__(; symbols: __init__, _apply_qk_norm, op_prepare, forward_prepare
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/utils.py` modified +80/-5; `python/sglang/srt/models/qwen3_moe.py` modified +9/-27; `python/sglang/srt/models/qwen3.py` modified +9/-24; `python/sglang/srt/models/bailing_moe.py` modified +9/-23; `python/sglang/srt/models/glm4_moe.py` modified +9/-23; `python/sglang/srt/models/llada2.py` modified +9/-23
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_qknorm.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16115 - [NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head

- Link: https://github.com/sgl-project/sglang/pull/16115
- Status/date: merged / 2026-01-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `7dd679cbb93a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +52/-16, 166 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[NPU][Bugfix] Fix qwen3 error when enable-dp-lm-head"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__, touching `forward_prepare_npu, __init__`; `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu, touching `forward_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +4/-3 (7 lines); hunks: -161,12 +161,12 @@ def forward_prepare_npu(self, positions, hidden_states, fo...; -372,6 +372,7 @@ def __init__(; symbols: forward_prepare_npu, __init__
  - `python/sglang/srt/models/qwen3_moe.py` modified +3/-3 (6 lines); hunks: -523,12 +523,12 @@ def forward_prepare_npu(; symbols: forward_prepare_npu
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-3; `python/sglang/srt/models/qwen3_moe.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/rotary_embedding.py`, `python/sglang/srt/layers/vocab_parallel_embedding.py`, `python/sglang/srt/models/llama.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #13715 - Fix EPLB + FP4 Quantization Compatibility Issue

- Link: https://github.com/sgl-project/sglang/pull/13715
- Status/date: merged / 2026-01-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +49/-3, 157 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix EPLB + FP4 Quantization Compatibility Issue"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`; technical summary: Covers "Fix EPLB + FP4 Quantization Compatibility Issue"; the main implementation surface is `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/qwen2_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather, touching `get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather`; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward, touching `get_moe_weights, forward`; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts, touching `get_moe_weights, _forward_shared_experts`; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal, touching `get_moe_weights, forward_normal`.
- Code diff details:
  - `python/sglang/srt/layers/moe/utils.py` modified +12/-0 (12 lines); hunks: -249,6 +249,18 @@ def get_tbo_token_distribution_threshold() -> float:; symbols: get_tbo_token_distribution_threshold, filter_moe_weight_param_global_expert, should_use_flashinfer_cutlass_moe_fp4_allgather
  - `python/sglang/srt/models/deepseek_v2.py` modified +7/-1 (8 lines); hunks: -103,7 +103,10; -587,6 +590,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward
  - `python/sglang/srt/models/qwen2_moe.py` modified +7/-1 (8 lines); hunks: -58,7 +58,10; -223,6 +226,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -51,7 +51,10; -281,6 +284,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, forward_normal
  - `python/sglang/srt/models/bailing_moe.py` modified +4/-0 (4 lines); hunks: -63,6 +63,7; -324,6 +325,9 @@ def get_moe_weights(self):; symbols: get_moe_weights, _forward_shared_experts
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/moe/utils.py` modified +12/-0; `python/sglang/srt/models/deepseek_v2.py` modified +7/-1; `python/sglang/srt/models/qwen2_moe.py` modified +7/-1; `python/sglang/srt/models/qwen3_moe.py` modified +7/-1; `python/sglang/srt/models/bailing_moe.py` modified +4/-0; `python/sglang/srt/models/glm4_moe.py` modified +4/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/layers/moe/utils.py`, `python/sglang/srt/models/bailing_moe.py`, `python/sglang/srt/models/deepseek_v2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15890 - [PP] fix wrong weight logic for tie_word_embeddings model

- Link: https://github.com/sgl-project/sglang/pull/15890
- Status/date: merged / 2026-01-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +19/-48, 108 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[PP] fix wrong weight logic for tie_word_embeddings model"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`; technical summary: Covers "[PP] fix wrong weight logic for tie_word_embeddings model"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunks: -378,20 +378,6 @@ def __init__(; -500,6 +486,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights, touching `__init__, load_weights`; `python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunks: -457,20 +457,6 @@ def __init__(; -589,22 +575,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +10/-24 (34 lines); hunks: -378,20 +378,6 @@ def __init__(; -500,6 +486,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2.py` modified +9/-24 (33 lines); hunks: -457,20 +457,6 @@ def __init__(; -589,22 +575,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch....; symbols: __init__, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-24; `python/sglang/srt/models/qwen2.py` modified +9/-24
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2.py`, `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15904 - [NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix

- Link: https://github.com/sgl-project/sglang/pull/15904
- Status/date: merged / 2026-01-28
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `b77b0ffd6021`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +111/-49, 296 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[NPU] NZ for non-quantized MOE, Qwen3 MOE double memory consumption fix"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -71,6 +71,7; -1119,14 +1120,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, get_model_config_for_expert_location, touching `load_weights, get_model_config_for_expert_location`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +10/-7 (17 lines); hunks: -71,6 +71,7; -1119,14 +1120,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torc...; symbols: load_weights, get_model_config_for_expert_location
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +10/-7
- Risk and verification: The diff ships test coverage in `test/registered/ascend/test_ascend_memory_consumption.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #15203 - [NPU] support GPTQ quantization on npu

- Link: https://github.com/sgl-project/sglang/pull/15203
- Status/date: merged / 2026-01-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +259/-6, 361 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] support GPTQ quantization on npu"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/linear.py`; technical summary: Covers "[NPU] support GPTQ quantization on npu"; the main implementation surface is `python/sglang/srt/layers/quantization/gptq.py`, `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/layers/linear.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunks: -49,7 +49,7; -63,6 +63,10; symbols: __init__, __repr__, get_scaled_act_names, get_name, touching `__init__, __repr__, get_scaled_act_names`; `python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunks: -299,7 +299,12 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7; `test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestAscendGPTQInt8, setUpClass, test_a_gsm8k, touching `TestAscendGPTQInt8, setUpClass, test_a_gsm8k`.
- Code diff details:
  - `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5 (183 lines); hunks: -49,7 +49,7; -63,6 +63,10; symbols: __init__, __repr__, get_scaled_act_names, get_name
  - `python/sglang/srt/models/qwen3.py` modified +6/-1 (7 lines); hunks: -299,7 +299,12 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/linear.py` modified +1/-0 (1 lines); hunks: -61,6 +61,7
  - `test/srt/ascend/test_ascend_gptq.py` added +73/-0 (73 lines); hunks: -0,0 +1,73; symbols: TestAscendGPTQInt8, setUpClass, test_a_gsm8k
  - `test/srt/run_suite.py` modified +1/-0 (1 lines); hunks: -133,6 +133,7
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/quantization/gptq.py` modified +178/-5; `python/sglang/srt/models/qwen3.py` modified +6/-1; `python/sglang/srt/layers/linear.py` modified +1/-0
  - tests: `test/srt/ascend/test_ascend_gptq.py` added +73/-0; `test/srt/run_suite.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/srt/ascend/test_ascend_gptq.py`, `test/srt/run_suite.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17535 - Update weight rename check for Qwen3 Embeddings

- Link: https://github.com/sgl-project/sglang/pull/17535
- Status/date: merged / 2026-02-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `793bf9fc0649`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-1, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Update weight rename check for Qwen3 Embeddings"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Update weight rename check for Qwen3 Embeddings"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -490,7 +490,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -490,7 +490,11 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18189 - [ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch

- Link: https://github.com/sgl-project/sglang/pull/18189
- Status/date: merged / 2026-02-08
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `ca36d88fa640`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-0, 15 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[ModelOpt] Fix broken Qwen3-235B-A22B-Instruct-2507-NVFP4 launch"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunks: -890,6 +890,14 @@ def __init__(; symbols: __init__, Qwen3MoeForCausalLM, touching `__init__, Qwen3MoeForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +8/-0 (8 lines); hunks: -890,6 +890,14 @@ def __init__(; symbols: __init__, Qwen3MoeForCausalLM
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +8/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19532 - [NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe

- Link: https://github.com/sgl-project/sglang/pull/19532
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; associated commits `365ca1edb5af`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +8/-2, 24 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[NPU] bugs fix: fix a condition bug when using speculative inference on Qwen3 and Qwen3 moe"; the main implementation surface is `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunks: -181,7 +181,10 @@ def forward(; symbols: forward, touching `forward`; `python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunks: -620,7 +620,10 @@ def forward_prepare(; symbols: forward_prepare, touching `forward_prepare`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +4/-1 (5 lines); hunks: -181,7 +181,10 @@ def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-1 (5 lines); hunks: -620,7 +620,10 @@ def forward_prepare(; symbols: forward_prepare
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-1; `python/sglang/srt/models/qwen3_moe.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`, `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20127 - [Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next

- Link: https://github.com/sgl-project/sglang/pull/20127
- Status/date: open / 2026-03-08
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +66/-25, 148 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`; technical summary: Covers "[Qwen] Handle tie_word_embeddings for Qwen MoE and Qwen3Next"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_next.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunks: -62,7 +62,7; -947,13 +947,20 @@ def __init__(; symbols: __init__, load_weights, touching `__init__, load_weights`; `python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunks: -743,13 +743,20 @@ def __init__(; -850,6 +857,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights, touching `__init__, load_weights`; `python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunks: -888,14 +888,17 @@ def __init__(; -936,9 +939,11 @@ def get_embed_and_head(self):; symbols: __init__, get_embed_and_head, set_embed_and_head, load_weights, touching `__init__, get_embed_and_head, set_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +25/-8 (33 lines); hunks: -62,7 +62,7; -947,13 +947,20 @@ def __init__(; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen2_moe.py` modified +24/-7 (31 lines); hunks: -743,13 +743,20 @@ def __init__(; -850,6 +857,16 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.T...; symbols: __init__, load_weights
  - `python/sglang/srt/models/qwen3_next.py` modified +17/-10 (27 lines); hunks: -888,14 +888,17 @@ def __init__(; -936,9 +939,11 @@ def get_embed_and_head(self):; symbols: __init__, get_embed_and_head, set_embed_and_head, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +25/-8; `python/sglang/srt/models/qwen2_moe.py` modified +24/-7; `python/sglang/srt/models/qwen3_next.py` modified +17/-10
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen2_moe.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3_next.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20474 - Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3

- Link: https://github.com/sgl-project/sglang/pull/20474
- Status/date: open / 2026-03-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +159/-7, 221 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `test/srt/xpu/test_qwen3.py`; technical summary: Covers "Intel XPU: Qwen3 support (layernorm/MRoPE) + test_qwen3"; the main implementation surface is `python/sglang/srt/layers/rotary_embedding/mrope.py`, `python/sglang/srt/layers/attention/fla/layernorm_gated.py`, `test/srt/xpu/test_qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunks: -243,6 +243,15 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index, touching `forward_npu, forward_xpu, get_rope_index`; `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunks: -21,11 +21,13; -172,6 +174,8 @@ def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count, touching `_layer_norm_fwd_1pass_kernel, _get_sm_count`; `test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: TestQwen3, setUpClass, tearDownClass, get_request_json, touching `TestQwen3, setUpClass, tearDownClass`; `docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunks: -20,6 +20,17 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +49,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda....
- Code diff details:
  - `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0 (9 lines); hunks: -243,6 +243,15 @@ def forward_npu(; symbols: forward_npu, forward_xpu, get_rope_index
  - `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0 (4 lines); hunks: -21,11 +21,13; -172,6 +174,8 @@ def _layer_norm_fwd_1pass_kernel(; symbols: _layer_norm_fwd_1pass_kernel, _get_sm_count
  - `test/srt/xpu/test_qwen3.py` added +133/-0 (133 lines); hunks: -0,0 +1,133; symbols: TestQwen3, setUpClass, tearDownClass, get_request_json
  - `docker/xpu.Dockerfile` modified +11/-6 (17 lines); hunks: -20,6 +20,17 @@ ARG SG_LANG_KERNEL_BRANCH=main; -38,12 +49,6 @@ RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda...
  - `.github/workflows/pr-test-xpu.yml` modified +1/-1 (2 lines); hunks: -120,7 +120,7 @@ jobs:
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/rotary_embedding/mrope.py` modified +9/-0; `python/sglang/srt/layers/attention/fla/layernorm_gated.py` modified +4/-0
  - tests: `test/srt/xpu/test_qwen3.py` added +133/-0; `test/srt/run_suite.py` modified +1/-0
  - other: `docker/xpu.Dockerfile` modified +11/-6
  - ci: `.github/workflows/pr-test-xpu.yml` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/srt/run_suite.py`, `test/srt/xpu/test_qwen3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17784 - Upgrade transformers==5.3.0

- Link: https://github.com/sgl-project/sglang/pull/17784
- Status/date: merged / 2026-03-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 95 files, +1136/-343, 2752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upgrade transformers==5.3.0"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`; technical summary: Covers "Upgrade transformers==5.3.0"; the main implementation surface is `python/sglang/srt/models/gemma3_causal.py`, `python/sglang/srt/layers/rotary_embedding/factory.py`, `python/sglang/srt/configs/model_config.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update, touching `__init__, Gemma3RotaryEmbedding, _dynamic_frequency_update`; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope, touching `_get_rope_param, get_rope`; `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes, touching `ModelImpl, is_deepseek_nsa, _derive_model_shapes`; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__, touching `compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope`.
- Code diff details:
  - `python/sglang/srt/models/gemma3_causal.py` modified +87/-14 (101 lines); hunks: -166,18 +166,36 @@ def __init__(; -325,9 +343,10 @@ class Gemma3RotaryEmbedding(nn.Module):; symbols: __init__, Gemma3RotaryEmbedding, _dynamic_frequency_update
  - `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13 (76 lines); hunks: -2,6 +2,7; -26,6 +27,29; symbols: _get_rope_param, get_rope
  - `python/sglang/srt/configs/model_config.py` modified +38/-18 (56 lines); hunks: -51,10 +51,20 @@ class ModelImpl(str, Enum):; -63,7 +73,7 @@ def is_deepseek_nsa(config: PretrainedConfig) -> bool:; symbols: ModelImpl, is_deepseek_nsa, _derive_model_shapes
  - `python/sglang/srt/models/qwen3_moe.py` modified +14/-7 (21 lines); hunks: -115,12 +115,19 @@ def compute_yarn_parameters(; -130,7 +137,7 @@ def compute_yarn_parameters(; symbols: compute_yarn_parameters, forward_prepare_native, apply_qk_norm_rope, __init__
  - `python/sglang/srt/models/midashenglm.py` modified +6/-14 (20 lines); hunks: -476,20 +476,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/gemma3_causal.py` modified +87/-14; `python/sglang/srt/layers/rotary_embedding/factory.py` modified +63/-13; `python/sglang/srt/configs/model_config.py` modified +38/-18; `python/sglang/srt/models/qwen3_moe.py` modified +14/-7; `python/sglang/srt/models/midashenglm.py` modified +6/-14; `python/sglang/srt/models/glm4.py` modified +3/-14
- Risk and verification: The diff ships test coverage in `python/sglang/test/runners.py`, `test/registered/core/test_score_api.py`, `test/registered/quant/test_awq.py`, `test/registered/rl/test_multi_instance_release_memory_occupation.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #20931 - [Bugifx] qwen3 rope parameter compatibility

- Link: https://github.com/sgl-project/sglang/pull/20931
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `46a76af97bec`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-3, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugifx] qwen3 rope parameter compatibility"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[Bugifx] qwen3 rope parameter compatibility"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__, touching `forward_prepare_native, apply_qk_norm_rope, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -78,6 +78,7; -566,7 +567,7 @@ def forward_prepare_native(; symbols: forward_prepare_native, apply_qk_norm_rope, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18233 - Support Qwen3 MoE context parallel

- Link: https://github.com/sgl-project/sglang/pull/18233
- Status/date: merged / 2026-03-22
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `bb737d7a829b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +968/-73, 1552 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Qwen3 MoE context parallel"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Support Qwen3 MoE context parallel"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +35/-5 (40 lines); hunks: -26,11 +26,14; -59,6 +62,11; symbols: __init__, forward_normal, get_input_embeddings, forward, touching `__init__, forward_normal, get_input_embeddings`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +35/-5 (40 lines); hunks: -26,11 +26,14; -59,6 +62,11; symbols: __init__, forward_normal, get_input_embeddings, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +35/-5
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/test_flashattn_backend.py`, `test/registered/4-gpu-models/test_qwen3_30b.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`, `test/registered/unit/managers/test_prefill_adder.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21195 - Enable the qwen3 test

- Link: https://github.com/sgl-project/sglang/pull/21195
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `dac148167c80`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-5, 32 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable the qwen3 test"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Enable the qwen3 test"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunks: -33,6 +33,7; -321,6 +322,9 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-0 (4 lines); hunks: -33,6 +33,7; -321,6 +322,9 @@ def forward_normal(; symbols: forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -33,6 +33,7 @@
+    moe_expert_parallel_all_reduce,
@@ -321,6 +322,9 @@ def forward_normal(
+        if self.ep_size > 1 and not should_allreduce_fusion:
+            final_hidden_states = moe_expert_parallel_all_reduce(final_hidden_states)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +4/-0
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_qwen3_30b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21412 - [Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints

- Link: https://github.com/sgl-project/sglang/pull/21412
- Status/date: open / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Bugfix] Fix Qwen3 RoPE config compatibility for old-style checkpoints"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunks: -31,6 +31,7; -216,8 +217,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +2/-2 (4 lines); hunks: -31,6 +31,7; -216,8 +217,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -31,6 +31,7 @@
+from sglang.srt.utils.hf_transformers_utils import get_rope_config
@@ -216,8 +217,7 @@ def __init__(
-        rope_theta = config.rope_parameters["rope_theta"]
-        rope_scaling = config.rope_parameters
+        rope_theta, rope_scaling = get_rope_config(config)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19059 - [jit_kernel] Add fused_qknorm_rope JIT kernel

- Link: https://github.com/sgl-project/sglang/pull/19059
- Status/date: merged / 2026-03-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +1127/-3, 1152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[jit_kernel] Add fused_qknorm_rope JIT kernel"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`; technical summary: Covers "[jit_kernel] Add fused_qknorm_rope JIT kernel"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunks: -91,7 +91,10; -503,12 +506,18 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave, touching `_compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads`; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunks: -0,0 +1,307; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: bench_fused_qknorm_rope, calculate_diff, touching `bench_fused_qknorm_rope, calculate_diff`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +12/-3 (15 lines); hunks: -91,7 +91,10; -503,12 +506,18 @@ def __init__(; symbols: __init__
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0 (444 lines); hunks: -0,0 +1,444; symbols: _compute_inv_freq_yarn, fused_qk_norm_rope_ref, rms_norm_heads, apply_interleave
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0 (307 lines); hunks: -0,0 +1,307
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0 (183 lines); hunks: -0,0 +1,183; symbols: bench_fused_qknorm_rope, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0 (181 lines); hunks: -0,0 +1,181; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, can_use_fused_qk_norm_rope, fused_qk_norm_rope
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +12/-3; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` added +307/-0; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` added +183/-0; `python/sglang/jit_kernel/fused_qknorm_rope.py` added +181/-0
  - tests: `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` added +444/-0
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21770 - [Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon

- Link: https://github.com/sgl-project/sglang/pull/21770
- Status/date: open / 2026-03-31
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +159/-0, 161 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`; technical summary: Covers "[Apple][MLX][Test] Add Qwen3 correctness and accuracy tests for Apple Silicon"; the main implementation surface is `test/registered/models/test_qwen3_mlx_correctness.py`, `test/registered/models/test_qwen3_mlx_accuracy.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat, touching `TestQwen3MlxCorrectness, setUpClass, tearDownClass`; `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy, touching `TestQwen3MlxAccuracy, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: TestQwen3MlxCorrectness, setUpClass, tearDownClass, _chat
  - `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0 (70 lines); hunks: -0,0 +1,70; symbols: TestQwen3MlxAccuracy, setUpClass, tearDownClass, test_gsm8k_accuracy
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/models/test_qwen3_mlx_correctness.py` added +89/-0; `test/registered/models/test_qwen3_mlx_accuracy.py` added +70/-0
- Risk and verification: The diff ships test coverage in `test/registered/models/test_qwen3_mlx_accuracy.py`, `test/registered/models/test_qwen3_mlx_correctness.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21654 - Fused_qknorm_rope kernel optimization: up to 2.4× faster

- Link: https://github.com/sgl-project/sglang/pull/21654
- Status/date: merged / 2026-04-01
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +208/-77, 545 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fused_qknorm_rope kernel optimization: up to 2.4× faster"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`; technical summary: Covers "Fused_qknorm_rope kernel optimization: up to 2.4× faster"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh`, `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -513,6 +513,7 @@ def __init__(; -521,6 +522,7 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunks: -39,11 +39,11 @@ namespace {; -68,11 +68,14 @@ compute_freq_yarn(float base, int rotary_dim, int half_dim,...; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunks: -1,8 +1,8; -39,7 +39,7; symbols: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff, touching `bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff`; `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunks: -13,17 +13,20; -55,24 +58,25 @@ def fused_qk_norm_rope_out(; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out, touching `_jit_fused_qknorm_rope_module, fused_qk_norm_rope_out`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +2/-0 (2 lines); hunks: -513,6 +513,7 @@ def __init__(; -521,6 +522,7 @@ def __init__(; symbols: __init__
  - `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55 (149 lines); hunks: -39,11 +39,11 @@ namespace {; -68,11 +68,14 @@ compute_freq_yarn(float base, int rotary_dim, int half_dim,...
  - `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4 (89 lines); hunks: -1,8 +1,8; -39,7 +39,7; symbols: bench_fused_qknorm_rope, bench_fused_qknorm_rope_production, calculate_diff
  - `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16 (41 lines); hunks: -13,17 +13,20; -55,24 +58,25 @@ def fused_qk_norm_rope_out(; symbols: _jit_fused_qknorm_rope_module, fused_qk_norm_rope_out
  - `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2 (4 lines); hunks: -122,7 +122,7 @@ def apply_interleave(x):; -231,7 +231,7 @@ def test_fused_qknorm_rope_partial_rotary(head_dim, is_neox):; symbols: apply_interleave, apply_neox, test_fused_qknorm_rope_partial_rotary
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +2/-0; `python/sglang/jit_kernel/csrc/elementwise/fused_qknorm_rope.cuh` modified +94/-55; `python/sglang/jit_kernel/benchmark/bench_fused_qknorm_rope.py` modified +85/-4; `python/sglang/jit_kernel/fused_qknorm_rope.py` modified +25/-16
  - tests: `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `python/sglang/jit_kernel/tests/test_fused_qknorm_rope.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21458 - [AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write

- Link: https://github.com/sgl-project/sglang/pull/21458
- Status/date: merged / 2026-04-01
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `912494f59665`, `a188208e9a03`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +101/-3, 152 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[AMD] Optimize Qwen3-VL decode - fuse QK-norm + 3D mRoPE + KV cache write"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope, touching `__init__, forward_prepare_native, forward_prepare_npu`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +101/-3 (104 lines); hunks: -19,6 +19,7; -30,13 +31,25; symbols: __init__, forward_prepare_native, forward_prepare_npu, forward_prepare_aiter_fused_mrope
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +101/-3
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22429 - [NPU]add Qwen3-32b and Qwen3-8b low latency md

- Link: https://github.com/sgl-project/sglang/pull/22429
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +296/-0, 310 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]add Qwen3-32b and Qwen3-8b low latency md"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `docs/platforms/ascend/ascend_npu_best_practice.md`; technical summary: Covers "[NPU]add Qwen3-32b and Qwen3-8b low latency md"; the main implementation surface is `docs/platforms/ascend/ascend_npu_best_practice.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunks: -37,6 +37,10 @@ you encounter issues or have any questions, please [open an i...; -2345,6 +2349,298 @@ We tested it based on the `RANDOM` dataset..
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0 (296 lines); hunks: -37,6 +37,10 @@ you encounter issues or have any questions, please [open an i...; -2345,6 +2349,298 @@ We tested it based on the `RANDOM` dataset.
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +296/-0
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22450 - [NPU] Add Qwen3-14B low latency doc

- Link: https://github.com/sgl-project/sglang/pull/22450
- Status/date: open / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +323/-0, 344 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add Qwen3-14B low latency doc"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `docs/platforms/ascend/ascend_npu_best_practice.md`; technical summary: Covers "[NPU] Add Qwen3-14B low latency doc"; the main implementation surface is `docs/platforms/ascend/ascend_npu_best_practice.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunks: -39,6 +39,9 @@ you encounter issues or have any questions, please [open an is...; -53,6 +56,7 @@ you encounter issues or have any questions, please [open an is....
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0 (323 lines); hunks: -39,6 +39,9 @@ you encounter issues or have any questions, please [open an is...; -53,6 +56,7 @@ you encounter issues or have any questions, please [open an is...
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +323/-0
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22358 - Enable DFLASH support for additional model backends

- Link: https://github.com/sgl-project/sglang/pull/22358
- Status/date: merged / 2026-04-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +152/-5, 299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable DFLASH support for additional model backends"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`; technical summary: Covers "Enable DFLASH support for additional model backends"; the main implementation surface is `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/kimi_k25.py`, `python/sglang/srt/models/qwen3_next.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture, touching `forward, get_layer, get_input_embeddings`; `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings`; `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head, touching `set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward`; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture, touching `__init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_5.py` modified +34/-5 (39 lines); hunks: -574,8 +574,15 @@ def forward(; -825,10 +832,16 @@ def forward(; symbols: forward, get_layer, get_input_embeddings, set_dflash_layers_to_capture
  - `python/sglang/srt/models/kimi_k25.py` modified +24/-0 (24 lines); hunks: -849,6 +849,30 @@ def set_eagle3_layers_to_capture(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, get_input_embeddings, lm_head
  - `python/sglang/srt/models/qwen3_next.py` modified +20/-0 (20 lines); hunks: -813,6 +813,11 @@ def set_eagle3_layers_to_capture(self, layers_to_capture: l...; -947,6 +952,9 @@ def forward(; symbols: set_eagle3_layers_to_capture, set_dflash_layers_to_capture, forward, get_embed_and_head
  - `python/sglang/srt/models/qwen3_moe.py` modified +17/-0 (17 lines); hunks: -924,6 +924,11 @@ def __init__(; -1079,6 +1084,18 @@ def set_eagle3_layers_to_capture(self, layer_ids: Optiona...; symbols: __init__, set_dflash_layers_to_capture, Qwen3MoeForCausalLM, set_eagle3_layers_to_capture
  - `python/sglang/srt/models/qwen3_vl.py` modified +16/-0 (16 lines); hunks: -1122,6 +1122,7 @@ def __init__(; -1246,19 +1247,34 @@ def forward(; symbols: __init__, forward, set_dflash_layers_to_capture, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_5.py` modified +34/-5; `python/sglang/srt/models/kimi_k25.py` modified +24/-0; `python/sglang/srt/models/qwen3_next.py` modified +20/-0; `python/sglang/srt/models/qwen3_moe.py` modified +17/-0; `python/sglang/srt/models/qwen3_vl.py` modified +16/-0; `python/sglang/srt/models/gpt_oss.py` modified +15/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/gpt_oss.py`, `python/sglang/srt/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22529 - [Model] Support sliding window attention for Qwen3

- Link: https://github.com/sgl-project/sglang/pull/22529
- Status/date: open / 2026-04-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-0, 71 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support sliding window attention for Qwen3"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[Model] Support sliding window attention for Qwen3"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunks: -36,6 +36,17; -71,6 +82,7 @@ def __init__(; symbols: get_attention_sliding_window_size, __init__, touching `get_attention_sliding_window_size, __init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +29/-0 (29 lines); hunks: -36,6 +36,17; -71,6 +82,7 @@ def __init__(; symbols: get_attention_sliding_window_size, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +29/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22446 - [NPU] add qwen3-30b-a3b low latency example

- Link: https://github.com/sgl-project/sglang/pull/22446
- Status/date: merged / 2026-04-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +130/-0, 141 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] add qwen3-30b-a3b low latency example"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `docs/platforms/ascend/ascend_npu_best_practice.md`; technical summary: Covers "[NPU] add qwen3-30b-a3b low latency example"; the main implementation surface is `docs/platforms/ascend/ascend_npu_best_practice.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunks: -41,6 +41,8 @@ you encounter issues or have any questions, please [open an is...; -2779,3 +2781,131 @@ We tested it based on the `RANDOM` dataset..
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0 (130 lines); hunks: -41,6 +41,8 @@ you encounter issues or have any questions, please [open an is...; -2779,3 +2781,131 @@ We tested it based on the `RANDOM` dataset.
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +130/-0
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22687 - [NPU]qwen3-8b and 32b md bugfix

- Link: https://github.com/sgl-project/sglang/pull/22687
- Status/date: merged / 2026-04-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-8, 68 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]qwen3-8b and 32b md bugfix"; model line: Qwen3 Core; category: bug fix; main diff: `docs/platforms/ascend/ascend_npu_best_practice.md`; technical summary: Covers "[NPU]qwen3-8b and 32b md bugfix"; the main implementation surface is `docs/platforms/ascend/ascend_npu_best_practice.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunks: -2397,7 +2397,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`; -2410,7 +2409,7 @@ python -m sglang.launch_server --model-path $MODEL_PATH \.
- Code diff details:
  - `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8 (12 lines); hunks: -2397,7 +2397,6 @@ LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`; -2410,7 +2409,7 @@ python -m sglang.launch_server --model-path $MODEL_PATH \
- Key code excerpts:

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

- Reviewed files:
  - docs: `docs/platforms/ascend/ascend_npu_best_practice.md` modified +4/-8
- Risk and verification: This is mostly docs/examples in `docs/platforms/ascend/ascend_npu_best_practice.md`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #22739 - Restore Qwen3 rope config fallback

- Link: https://github.com/sgl-project/sglang/pull/22739
- Status/date: merged / 2026-04-14
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `520ce526b919`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +10/-2, 19 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Restore Qwen3 rope config fallback"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Restore Qwen3 rope config fallback"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +10/-2 (12 lines); hunks: -316,8 +316,16 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +10/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22003 - Support moe_dp_size = 1 for various attention_cp_size

- Link: https://github.com/sgl-project/sglang/pull/22003
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +276/-25, 485 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support moe_dp_size = 1 for various attention_cp_size"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Support moe_dp_size = 1 for various attention_cp_size"; the main implementation surface is `python/sglang/srt/layers/communicator.py`, `python/sglang/srt/layers/dp_attention.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode, touching `ScatterMode, model_input_output, _compute_layer_input_mode`; `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size, touching `attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank`; `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings, touching `__init__, get_input_embeddings`; `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split, touching `is_prefill_cp_in_seq_split, can_cp_split`.
- Code diff details:
  - `python/sglang/srt/layers/communicator.py` modified +164/-10 (174 lines); hunks: -50,8 +50,12; -188,11 +192,13 @@ class ScatterMode(Enum):; symbols: ScatterMode, model_input_output, _compute_layer_input_mode, _compute_mlp_mode
  - `python/sglang/srt/layers/dp_attention.py` modified +28/-0 (28 lines); hunks: -18,6 +18,9; -580,5 +583,30 @@ def attn_cp_all_gather_into_tensor(output: torch.Tensor, in...; symbols: attn_cp_all_gather_into_tensor, get_moe_cp_group, get_moe_cp_rank, get_moe_cp_size
  - `python/sglang/srt/models/qwen3_moe.py` modified +4/-3 (7 lines); hunks: -968,9 +968,10 @@ def __init__(; symbols: __init__, get_input_embeddings
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4 (6 lines); hunks: -43,16 +43,14 @@ def is_prefill_cp_in_seq_split():; symbols: is_prefill_cp_in_seq_split, can_cp_split
  - `python/sglang/srt/models/qwen2_moe.py` modified +5/-1 (6 lines); hunks: -33,6 +33,9; -709,6 +712,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/layers/communicator.py` modified +164/-10; `python/sglang/srt/layers/dp_attention.py` modified +28/-0; `python/sglang/srt/models/qwen3_moe.py` modified +4/-3; `python/sglang/srt/layers/utils/cp_utils.py` modified +2/-4; `python/sglang/srt/models/qwen2_moe.py` modified +5/-1; `python/sglang/srt/distributed/parallel_state.py` modified +13/-7
  - tests: `test/registered/4-gpu-models/test_qwen3_30b.py` modified +55/-0
- Risk and verification: The diff ships test coverage in `test/registered/4-gpu-models/test_qwen3_30b.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23372 - [NPU] Add CI tests for Speculative Decoding

- Link: https://github.com/sgl-project/sglang/pull/23372
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +759/-14, 829 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Add CI tests for Speculative Decoding"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`; technical summary: Covers "[NPU] Add CI tests for Speculative Decoding"; the main implementation surface is `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +193/-0 (193 lines); hunks: -0,0 +1,193; symbols: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode, touching `TestAscendSpeculativeAttentionMode, setUpClass, start_prefill`; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +164/-0 (164 lines); hunks: -0,0 +1,164; symbols: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info, touching `TestNpuSpeculativeDraftParams, setUpClass, tearDownClass`; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k, touching `TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k`; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k, touching `TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass`.
- Code diff details:
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +193/-0 (193 lines); hunks: -0,0 +1,193; symbols: TestAscendSpeculativeAttentionMode, setUpClass, start_prefill, start_decode
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +164/-0 (164 lines); hunks: -0,0 +1,164; symbols: TestNpuSpeculativeDraftParams, setUpClass, tearDownClass, test_draft_params_via_server_info
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +161/-0 (161 lines); hunks: -0,0 +1,161; symbols: TestNpuSpeculativeTokenMap, test_eagle3_ignores_token_map_gsm8k, test_eagle_with_valid_token_map_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +110/-0 (110 lines); hunks: -0,0 +1,110; symbols: TestAscendSpeculativeDraftAttentionAndMoeRunner, setUpClass, tearDownClass, test_a_gsm8k
  - `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +104/-0 (104 lines); hunks: -0,0 +1,104; symbols: TestAscendSpeculativeMoeA2ABackend, setUpClass, test_a_gsm8k
- Key code excerpts:

```diff
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py
@@ -0,0 +1,193 @@
+import logging
+import os
+import socket
+import unittest
+from types import SimpleNamespace
+from urllib.parse import urlparse
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py
@@ -0,0 +1,164 @@
+import logging
+import os
+import unittest
+import requests
+from sglang.srt.utils import kill_process_tree
+from sglang.test.ascend.test_ascend_utils import (
diff -- test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py
@@ -0,0 +1,161 @@
```

- Reviewed files:
  - tests: `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py` added +193/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_multi_npu.py` added +164/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_token_map.py` added +161/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py` added +110/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_moe_a2a_backend.py` added +104/-0; `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py` modified +14/-12
- Risk and verification: The diff ships test coverage in `python/sglang/test/ascend/test_ascend_utils.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_eagle3.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_attention_mode.py`, `test/registered/ascend/basic_function/speculative_inference/test_npu_speculative_draft_attention_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23397 - [alignment-sglang] PR3: Dense Deterministic Math

- Link: https://github.com/sgl-project/sglang/pull/23397
- Status/date: open / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 83 files, +4952/-581, 7494 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[alignment-sglang] PR3: Dense Deterministic Math"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `test/srt/models/test_params_mapping.py`, `python/sglang/srt/layers/on_policy_utils.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "[alignment-sglang] PR3: Dense Deterministic Math"; the main implementation surface is `test/srt/models/test_params_mapping.py`, `python/sglang/srt/layers/on_policy_utils.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/models/test_params_mapping.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: _make_model, _deepseek_mutate, _deepseek_scale_remap, _glm4lite_mutate, touching `_make_model, _deepseek_mutate, _deepseek_scale_remap`; `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target, touching `_get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled`; `python/sglang/srt/model_executor/model_runner.py` modified +166/-7 (173 lines); hunks: -72,7 +72,10; -112,6 +115,7; symbols: __init__, initialize, touching `__init__, initialize`; `python/sglang/srt/models/qwen3_vl.py` modified +60/-112 (172 lines); hunks: -431,75 +431,73 @@ def rot_pos_emb(; -607,61 +605,6 @@ def bucket_flashinfer_max_seqlen(self, real_max_seqlen: int...; symbols: rot_pos_emb, _get_interpolation_indices, fast_pos_embed_interpolate, _calculate_indices_and_weights, touching `rot_pos_emb, _get_interpolation_indices, fast_pos_embed_interpolate`.
- Code diff details:
  - `test/srt/models/test_params_mapping.py` added +292/-0 (292 lines); hunks: -0,0 +1,292; symbols: _make_model, _deepseek_mutate, _deepseek_scale_remap, _glm4lite_mutate
  - `python/sglang/srt/layers/on_policy_utils.py` added +222/-0 (222 lines); hunks: -0,0 +1,222; symbols: _get_server_args, get_rl_on_policy_target, is_true_on_policy_enabled, is_tp_invariant_target
  - `python/sglang/srt/model_executor/model_runner.py` modified +166/-7 (173 lines); hunks: -72,7 +72,10; -112,6 +115,7; symbols: __init__, initialize
  - `python/sglang/srt/models/qwen3_vl.py` modified +60/-112 (172 lines); hunks: -431,75 +431,73 @@ def rot_pos_emb(; -607,61 +605,6 @@ def bucket_flashinfer_max_seqlen(self, real_max_seqlen: int...; symbols: rot_pos_emb, _get_interpolation_indices, fast_pos_embed_interpolate, _calculate_indices_and_weights
  - `python/sglang/srt/models/deepseek_v2.py` modified +100/-9 (109 lines); hunks: -152,6 +152,7; -167,8 +168,6; symbols: forward, __init__, op_prepare
- Key code excerpts:

```diff
diff -- test/srt/models/test_params_mapping.py
@@ -0,0 +1,292 @@
+"""Unit tests for ParameterMapper."""
+from types import SimpleNamespace
+import pytest
+from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
+from sglang.srt.model_loader.parameter_mapper import ParameterMapper
+_DEEPSEEK_N_ROUTED = 4
diff -- python/sglang/srt/layers/on_policy_utils.py
@@ -0,0 +1,222 @@
+from __future__ import annotations
+import contextlib
+import os
+from typing import Any, Iterator, Optional
+import torch
+ROW_LINEAR_INV_BLOCK_K = 128
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -72,7 +72,10 @@
```

- Reviewed files:
  - tests: `test/srt/models/test_params_mapping.py` added +292/-0
  - runtime: `python/sglang/srt/layers/on_policy_utils.py` added +222/-0; `python/sglang/srt/model_executor/model_runner.py` modified +166/-7; `python/sglang/srt/models/qwen3_vl.py` modified +60/-112; `python/sglang/srt/models/deepseek_v2.py` modified +100/-9; `python/sglang/srt/models/qwen3_moe.py` modified +59/-22; `python/sglang/srt/models/glm4_moe.py` modified +32/-38
- Risk and verification: The diff ships test coverage in `test/registered/8-gpu-models/test_deepseek_v32_indexcache.py`, `test/registered/8-gpu-models/test_nvidia_nemotron_3_super_nightly.py`, `test/registered/8-gpu-models/test_qwen3_235b.py`, `test/registered/core/test_dense_deterministic_math.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23731 - Fix Qwen3 MoE double-reduce when DP attention + EP + reduce_scatterv (#23729)

- Link: https://github.com/sgl-project/sglang/pull/23731
- Status/date: merged / 2026-04-25
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `71029abd640f`, `99b59b279ce2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-1, 29 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Qwen3 MoE double-reduce when DP attention + EP + reduce_scatterv (#23729)"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fix Qwen3 MoE double-reduce when DP attention + EP + reduce_scatterv (#23729)"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -50,6 +50,7; -331,14 +332,19 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +7/-1 (8 lines); hunks: -50,6 +50,7; -331,14 +332,19 @@ def forward_normal(; symbols: forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -50,6 +50,7 @@
+    should_use_dp_reduce_scatterv,
@@ -331,14 +332,19 @@ def forward_normal(
-        if self.ep_size > 1 and not should_allreduce_fusion:
+        if (
+            self.ep_size > 1
+            and not should_allreduce_fusion
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +7/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23734 - Fix Qwen3 MoE: also guard EP all-reduce with not use_reduce_scatter (follow-up to #23731)

- Link: https://github.com/sgl-project/sglang/pull/23734
- Status/date: merged / 2026-04-26
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`; associated commits `71029abd640f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-0, 8 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Qwen3 MoE: also guard EP all-reduce with not use_reduce_scatter (follow-up to #23731)"; model line: Qwen3 Core; category: bug fix; main diff: `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "Fix Qwen3 MoE: also guard EP all-reduce with not use_reduce_scatter (follow-up to #23731)"; the main implementation surface is `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -335,6 +335,7 @@ def forward_normal(; symbols: forward_normal, touching `forward_normal`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe.py` modified +1/-0 (1 lines); hunks: -335,6 +335,7 @@ def forward_normal(; symbols: forward_normal
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -335,6 +335,7 @@ def forward_normal(
+            and not use_reduce_scatter
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_moe.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19484 - [CPU] Add Qwen3.5 model optimization for CPU

- Link: https://github.com/sgl-project/sglang/pull/19484
- Status/date: merged / 2026-04-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 20 files, +768/-209, 1454 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU] Add Qwen3.5 model optimization for CPU"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`; technical summary: Covers "[CPU] Add Qwen3.5 model optimization for CPU"; the main implementation surface is `python/sglang/srt/configs/update_config.py`, `python/sglang/srt/models/qwen3_5.py`, `python/sglang/srt/models/qwen3_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim, touching `get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim`; `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights, touching `__init__, weight_loader, forward`; `python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__, touching `Qwen3_VisionMLP, __init__`; `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/configs/update_config.py` modified +178/-75 (253 lines); hunks: -1,7 +1,13; -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):; symbols: get_moe_padding_size, get_num_heads_padding_size, resolve_head_dim
  - `python/sglang/srt/models/qwen3_5.py` modified +37/-4 (41 lines); hunks: -124,8 +124,16 @@ def __init__(; -321,7 +329,20 @@ def weight_loader(param, loaded_weight, loaded_shard_id=None):; symbols: __init__, weight_loader, forward, load_weights
  - `python/sglang/srt/models/qwen3_vl.py` modified +29/-6 (35 lines); hunks: -72,7 +72,13; -87,6 +93,9; symbols: Qwen3_VisionMLP, __init__
  - `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11 (30 lines); hunks: -375,14 +375,22 @@ def forward(; symbols: forward
  - `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2 (20 lines); hunks: -1,3 +1,4; -29,7 +30,12; symbols: mamba_v2_sharded_weight_loader, loader
- Key code excerpts:

```diff
diff -- python/sglang/srt/configs/update_config.py
@@ -1,7 +1,13 @@
+import logging
+from sglang.srt.utils import (
+    log_debug_on_rank0,
+)
+logger = logging.getLogger(__name__)
@@ -40,7 +46,14 @@ def get_moe_padding_size(weight_block_size):
diff -- python/sglang/srt/models/qwen3_5.py
@@ -124,8 +124,16 @@ def __init__(
-        self.num_v_heads = config.linear_num_value_heads
-        self.num_k_heads = config.linear_num_key_heads
+        self.num_v_heads = (
+            config.linear_num_value_heads
+            if not _is_cpu
+            else config.linear_num_value_heads_cpu
diff -- python/sglang/srt/models/qwen3_vl.py
@@ -72,7 +72,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/configs/update_config.py` modified +178/-75; `python/sglang/srt/models/qwen3_5.py` modified +37/-4; `python/sglang/srt/models/qwen3_vl.py` modified +29/-6; `python/sglang/srt/layers/attention/fla/fused_norm_gate.py` modified +19/-11; `python/sglang/srt/layers/attention/mamba/mamba.py` modified +18/-2; `python/sglang/srt/models/qwen3_next.py` modified +14/-5
- Risk and verification: The diff ships test coverage in `test/srt/cpu/test_mamba.py`, `test/srt/cpu/test_qwen3.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23434 - [Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model

- Link: https://github.com/sgl-project/sglang/pull/23434
- Status/date: merged / 2026-04-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-0, 10 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3_classification.py`; technical summary: Covers "[Model] Qwen3ForPooledOutput: forward get_input_embeddings to inner model"; the main implementation surface is `python/sglang/srt/models/qwen3_classification.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunks: -55,6 +55,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, touching `__init__, get_input_embeddings, forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_classification.py` modified +3/-0 (3 lines); hunks: -55,6 +55,9 @@ def __init__(; symbols: __init__, get_input_embeddings, forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_classification.py
@@ -55,6 +55,9 @@ def __init__(
+    def get_input_embeddings(self) -> nn.Embedding:
+        return self.model.get_input_embeddings()
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_classification.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3_classification.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20520 - [NPU]TP Communications compression For Qwen3 models for NPU

- Link: https://github.com/sgl-project/sglang/pull/20520
- Status/date: merged / 2026-05-02
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `83bf5d6869c3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +191/-10, 346 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU]TP Communications compression For Qwen3 models for NPU"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "[NPU]TP Communications compression For Qwen3 models for NPU"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +1/-1 (2 lines); hunks: -419,7 +419,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -419,7 +419,7 @@ def forward(
-        hidden_states = self.mlp(hidden_states)
+        hidden_states = self.mlp(hidden_states, forward_batch=forward_batch)
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +1/-1
- Risk and verification: The diff ships test coverage in `test/registered/ascend/llm_models/test_npu_llama_2_7b_communications_compression.py`, `test/registered/ascend/llm_models/test_npu_qwen3_8b_communications_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22674 - [NPU] Support Qwen3.5-MoE and Qwen3-Next quantization

- Link: https://github.com/sgl-project/sglang/pull/22674
- Status/date: closed / 2026-05-07
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-0, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NPU] Support Qwen3.5-MoE and Qwen3-Next quantization"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/model_loader/loader.py`; technical summary: Covers "[NPU] Support Qwen3.5-MoE and Qwen3-Next quantization"; the main implementation surface is `python/sglang/srt/model_loader/loader.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def _get_quantization_config(; symbols: _get_quantization_config, touching `_get_quantization_config`.
- Code diff details:
  - `python/sglang/srt/model_loader/loader.py` modified +2/-0 (2 lines); hunks: -215,6 +215,8 @@ def _get_quantization_config(; symbols: _get_quantization_config
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_loader/loader.py
@@ -215,6 +215,8 @@ def _get_quantization_config(
+                    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
+                    "in_proj_ba": ["in_proj_b", "in_proj_a"],
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_loader/loader.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/model_loader/loader.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25825 - [Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool

- Link: https://github.com/sgl-project/sglang/pull/25825
- Status/date: merged / 2026-05-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +59/-8, 326 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`; technical summary: Covers "[Refactor] Pass PP start_layer via model constructor instead of forward_batch.token_to_kv_pool"; the main implementation surface is `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/qwen2.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu, touching `__init__, forward_prepare_native, forward_prepare_npu`; `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare, touching `__init__, forward_prepare`; `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__, touching `__init__`; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/llama.py` modified +16/-2 (18 lines); hunks: -27,6 +27,7; -131,6 +132,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
  - `python/sglang/srt/models/glm4_moe.py` modified +12/-1 (13 lines); hunks: -28,6 +28,7; -187,6 +188,7 @@ def __init__(; symbols: __init__, forward_prepare
  - `python/sglang/srt/models/qwen2.py` modified +9/-0 (9 lines); hunks: -24,6 +24,7; -200,12 +201,14 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen2_moe.py` modified +9/-0 (9 lines); hunks: -32,6 +32,7; -600,13 +601,15 @@ def __init__(; symbols: __init__
  - `python/sglang/srt/models/qwen3.py` modified +5/-1 (6 lines); hunks: -64,6 +64,7 @@ def __init__(; -76,6 +77,7 @@ def __init__(; symbols: __init__, forward_prepare_native, forward_prepare_npu
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/llama.py
@@ -27,6 +27,7 @@
+    get_pp_indices,
@@ -131,6 +132,7 @@ def __init__(
+        start_layer: int = 0,
@@ -141,6 +143,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -210,7 +213,7 @@ def forward_prepare_native(self, positions, hidden_states):
diff -- python/sglang/srt/models/glm4_moe.py
@@ -28,6 +28,7 @@
+    get_pp_indices,
@@ -187,6 +188,7 @@ def __init__(
+        start_layer: int = 0,
@@ -201,6 +203,7 @@ def __init__(
+        self.start_layer = start_layer
@@ -312,7 +315,7 @@ def forward_prepare(
diff -- python/sglang/srt/models/qwen2.py
@@ -24,6 +24,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/llama.py` modified +16/-2; `python/sglang/srt/models/glm4_moe.py` modified +12/-1; `python/sglang/srt/models/qwen2.py` modified +9/-0; `python/sglang/srt/models/qwen2_moe.py` modified +9/-0; `python/sglang/srt/models/qwen3.py` modified +5/-1; `python/sglang/srt/models/qwen3_moe.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/llama.py`, `python/sglang/srt/models/llama_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25983 - feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext

- Link: https://github.com/sgl-project/sglang/pull/25983
- Status/date: merged / 2026-05-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 77 files, +1227/-905, 5236 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`; technical summary: Covers "feat(model_runner): remove pool/backend refs from ForwardBatch via ForwardContext"; the main implementation surface is `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/model_executor/cuda_graph_runner.py`, `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode, touching `get_spec_info, run_once, maybe_init_ngram_embedding`; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once, touching `capture_one_batch_size, run_once`; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size, touching `warmup_compile, _cache_loc_dtype, capture_one_batch_size`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp, touching `_get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp`.
- Code diff details:
  - `python/sglang/srt/model_executor/model_runner.py` modified +107/-84 (191 lines); hunks: -146,6 +146,11; -2638,9 +2643,6 @@ def get_spec_info():; symbols: get_spec_info, run_once, maybe_init_ngram_embedding, forward_decode
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67 (137 lines); hunks: -65,6 +65,7; -1016,9 +1017,6 @@ def capture_one_batch_size(; symbols: capture_one_batch_size, run_once
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58 (118 lines); hunks: -58,6 +58,7; -387,9 +388,6 @@ def warmup_compile(self, num_tokens: int):; symbols: warmup_compile, _cache_loc_dtype, capture_one_batch_size
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44 (87 lines); hunks: -80,6 +80,11; -449,9 +454,9 @@ def _get_topk_paged(; symbols: _get_topk_paged, _get_topk_ragged, _get_topk_ragged_with_cp
  - `python/sglang/srt/model_executor/forward_context.py` added +84/-0 (84 lines); hunks: -0,0 +1,84; symbols: ForwardContext, set_forward_context, has_forward_context, get_forward_context
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -146,6 +146,11 @@
+from sglang.srt.model_executor.forward_context import (
+    ForwardContext,
+    forward_context,
+    has_forward_context,
+)
@@ -2638,9 +2643,6 @@ def get_spec_info():
diff -- python/sglang/srt/model_executor/cuda_graph_runner.py
@@ -65,6 +65,7 @@
+from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
@@ -1016,9 +1017,6 @@ def capture_one_batch_size(
-            req_to_token_pool=self.model_runner.req_to_token_pool,
-            token_to_kv_pool=self.model_runner.token_to_kv_pool,
-            attn_backend=attn_backend,
@@ -1040,85 +1038,90 @@ def capture_one_batch_size(
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -58,6 +58,7 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/model_runner.py` modified +107/-84; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +70/-67; `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` modified +60/-58; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +43/-44; `python/sglang/srt/model_executor/forward_context.py` added +84/-0; `python/sglang/srt/model_executor/cpu_graph_runner.py` modified +39/-38
- Risk and verification: The diff ships test coverage in `test/manual/attention/test_flashattn_backend.py`, `test/manual/attention/test_flashattn_mla_backend.py`, `test/manual/attention/test_prefix_chunk_info.py`, `test/manual/attention/test_trtllm_mla_backend.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #23292 - [CP] 1/N: Support MLA Prefill Context Parallel

- Link: https://github.com/sgl-project/sglang/pull/23292
- Status/date: merged / 2026-05-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 21 files, +900/-161, 1566 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CP] 1/N: Support MLA Prefill Context Parallel"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py`; technical summary: Covers "[CP] 1/N: Support MLA Prefill Context Parallel"; the main implementation surface is `python/sglang/srt/layers/attention/flashattention_backend.py`, `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/layers/utils/cp_utils.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56 (184 lines); hunks: -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -627,36 +646,43 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, _fa_cp_attn, _mla_cp_attn, touching `init_forward_metadata, forward_extend, _fa_cp_attn`; `python/sglang/srt/models/deepseek_v2.py` modified +73/-14 (87 lines); hunks: -123,9 +123,12; -339,6 +342,8 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19 (55 lines); hunks: -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():; -395,6 +417,7 @@ def prepare_context_parallel_metadata(; symbols: is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp, can_cp_split, touching `is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp`; `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8 (39 lines); hunks: -43,9 +43,12; -136,6 +139,14 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56 (184 lines); hunks: -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):; -627,36 +646,43 @@ def forward_extend(; symbols: init_forward_metadata, forward_extend, _fa_cp_attn, _mla_cp_attn
  - `python/sglang/srt/models/deepseek_v2.py` modified +73/-14 (87 lines); hunks: -123,9 +123,12; -339,6 +342,8 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19 (55 lines); hunks: -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():; -395,6 +417,7 @@ def prepare_context_parallel_metadata(; symbols: is_prefill_cp_in_seq_split, is_mla_prefill_cp_enabled, mla_use_prefill_cp, can_cp_split
  - `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8 (39 lines); hunks: -43,9 +43,12; -136,6 +139,14 @@ def __init__(; symbols: __init__, forward
  - `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +14/-3 (17 lines); hunks: -56,6 +56,7; -567,7 +568,15 @@ def __init__(; symbols: __init__, capture_one_batch_size, replay_prepare
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/attention/flashattention_backend.py
@@ -508,6 +508,25 @@ def init_forward_metadata(self, forward_batch: ForwardBatch):
+            # MLA/MHA CP: prepare_mlp_sync_batch pads extend tokens up to
+            # lcm(attn_tp_size, attn_cp_size), so cache_seqlens_cp can exceed
+            # seq_lens_cpu.max(). Widen page_table by the pad delta to keep
+            # FA3's causal reads in-bounds; widened columns index KV slot 0
+            # (req_to_token is zero-init) and outputs for padding queries are
+            # discarded downstream.
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -123,9 +123,12 @@
+    can_cp_split,
+    is_prefill_context_parallel_enabled,
+    mla_use_prefill_cp,
@@ -339,6 +342,8 @@ def __init__(
+        dsa_enable_prefill_cp: bool = False,
+        mla_enable_prefill_cp: bool = False,
diff -- python/sglang/srt/layers/utils/cp_utils.py
@@ -51,19 +51,41 @@ def is_prefill_cp_in_seq_split():
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/attention/flashattention_backend.py` modified +128/-56; `python/sglang/srt/models/deepseek_v2.py` modified +73/-14; `python/sglang/srt/layers/utils/cp_utils.py` modified +36/-19; `python/sglang/srt/models/deepseek_nextn.py` modified +31/-8; `python/sglang/srt/model_executor/cuda_graph_runner.py` modified +14/-3; `python/sglang/srt/layers/communicator.py` modified +10/-4
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_deepseek_v3_cp_single_node.py`, `test/registered/cp/test_deepseek_v4_flash_fp4_b200_cp.py`, `test/registered/cp/test_qwen3_30b.py`, `test/registered/kernels/test_cp_prefix_len_fa3_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25971 - [CPU Doc]Add Xeon CPU info in Qwen3 Cookbook

- Link: https://github.com/sgl-project/sglang/pull/25971
- Status/date: merged / 2026-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`, `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx`; associated commits `47617cc4df1e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-10, 126 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CPU Doc]Add Xeon CPU info in Qwen3 Cookbook"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`; technical summary: Covers "[CPU Doc]Add Xeon CPU info in Qwen3 Cookbook"; the main implementation surface is `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx`, `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx` modified +22/-9 (31 lines); hunks: -9,7 +9,8 @@ export const Qwen3Deployment = () => {; -19,7 +20,8 @@ export const Qwen3Deployment = () => {; `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` modified +4/-1 (5 lines); hunks: -26,13 +26,15 @@ SGLang offers multiple installation methods. You can choose...; -51,6 +53,7 @@ import { Qwen3Deployment } from "/src/snippets/autoregressive/....
- Code diff details:
  - `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx` modified +22/-9 (31 lines); hunks: -9,7 +9,8 @@ export const Qwen3Deployment = () => {; -19,7 +20,8 @@ export const Qwen3Deployment = () => {
  - `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` modified +4/-1 (5 lines); hunks: -26,13 +26,15 @@ SGLang offers multiple installation methods. You can choose...; -51,6 +53,7 @@ import { Qwen3Deployment } from "/src/snippets/autoregressive/...
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/qwen3-deployment.jsx
@@ -9,7 +9,8 @@ export const Qwen3Deployment = () => {
-      mi355x: { tp: 4, ep: 0, bf16: true, fp8: true }
+      mi355x: { tp: 4, ep: 0, bf16: true, fp8: true },
+      xeon: { tp: 6, ep: 0, bf16: true, fp8: true }
@@ -19,7 +20,8 @@ export const Qwen3Deployment = () => {
-      mi355x: { tp: 1, ep: 0, bf16: true, fp8: true }
+      mi355x: { tp: 1, ep: 0, bf16: true, fp8: true },
diff -- docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx
@@ -26,13 +26,15 @@ SGLang offers multiple installation methods. You can choose the most suitable in
+For SGLang CPU installation, please refer to the [CPU version installation guide](../../../docs/hardware-platforms/cpu_server#installation).
-The Qwen3 series offers models in various sizes and architectures, optimized for different hardware platforms including NVIDIA and AMD GPUs. The recommended launch configurations
+The Qwen3 series offers models in various sizes and architectures, optimized for different hardware platforms including NVIDIA GPUs, AMD GPUs, and Intel Xeon CPUs. The recommended
@@ -51,6 +53,7 @@ import { Qwen3Deployment } from "/src/snippets/autoregressive/qwen3-deployment.j
+- For configuring CPU service, please refer to the `Notes` part in the serving engine launching section in [the SGLang CPU server document](../../../docs/hardware-platforms/cpu_se
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx` modified +22/-9; `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx` modified +4/-1
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/Qwen/Qwen3.mdx`, `docs_new/src/snippets/autoregressive/qwen3-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23269 - Support batch size > 1 when enable CP

- Link: https://github.com/sgl-project/sglang/pull/23269
- Status/date: merged / 2026-05-27
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +268/-305, 797 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support batch size > 1 when enable CP"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`; technical summary: Covers "Support batch size > 1 when enable CP"; the main implementation surface is `python/sglang/srt/layers/utils/cp_utils.py`, `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`, `python/sglang/srt/model_executor/forward_batch_info.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153 (389 lines); hunks: -20,24 +20,41; -67,25 +84,45 @@ def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=...; symbols: ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp, can_cp_split, touching `ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp`; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4 (12 lines); hunks: -1455,10 +1455,14 @@ def forward_cuda(; symbols: forward_cuda, touching `forward_cuda`; `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2 (5 lines); hunks: -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch, touching `prepare_mlp_sync_batch`; `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1 (4 lines); hunks: -133,7 +133,9 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens, touching `cal_padded_tokens`.
- Code diff details:
  - `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153 (389 lines); hunks: -20,24 +20,41; -67,25 +84,45 @@ def mla_use_prefill_cp(forward_batch, mla_enable_prefill_cp=...; symbols: ContextParallelMetadata, is_prefill_context_parallel_enabled, mla_use_prefill_cp, can_cp_split
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4 (12 lines); hunks: -1455,10 +1455,14 @@ def forward_cuda(; symbols: forward_cuda
  - `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2 (5 lines); hunks: -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):; symbols: prepare_mlp_sync_batch
  - `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1 (4 lines); hunks: -133,7 +133,9 @@ def cal_padded_tokens(forward_batch: "ForwardBatch"):; symbols: cal_padded_tokens
  - `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2 (4 lines); hunks: -311,7 +311,7 @@ def forward(; -320,7 +320,7 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/utils/cp_utils.py
@@ -20,24 +20,41 @@
+    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
-    max_rank_len: List[int] = None
-    per_rank_actual_token: List[int] = None
-    reverse_split_len: List[int] = None
+    reverse_split_len: List[int] = None
+    # Per-rank-aggregate lists have length cp_size.
diff -- python/sglang/srt/layers/attention/dsa/dsa_indexer.py
@@ -1455,10 +1455,14 @@ def forward_cuda(
-                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev
-                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next
-                    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev
-                    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next
+                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
+                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
diff -- python/sglang/srt/model_executor/forward_batch_info.py
@@ -889,10 +889,11 @@ def prepare_mlp_sync_batch(self, model_runner: ModelRunner):
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/utils/cp_utils.py` modified +236/-153; `python/sglang/srt/layers/attention/dsa/dsa_indexer.py` modified +8/-4; `python/sglang/srt/model_executor/forward_batch_info.py` modified +3/-2; `python/sglang/srt/layers/attention/dsa/utils.py` modified +3/-1; `python/sglang/srt/models/deepseek_nextn.py` modified +2/-2; `python/sglang/srt/models/deepseek_v2.py` modified +2/-2
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_qwen3_30b.py`, `test/registered/kernels/test_cp_prefix_len_fa3_parity.py`, `test/registered/kernels/test_mla_cp_fa3_parity.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26673 - [refactor] remove unused op_mlp

- Link: https://github.com/sgl-project/sglang/pull/26673
- Status/date: merged / 2026-05-29
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +0/-53, 95 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[refactor] remove unused op_mlp"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; technical summary: Covers "[refactor] remove unused op_mlp"; the main implementation surface is `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`; `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer, touching `op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer`.
- Code diff details:
  - `python/sglang/srt/models/deepseek_v2.py` modified +0/-13 (13 lines); hunks: -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe.py` modified +0/-13 (13 lines); hunks: -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13 (13 lines); hunks: -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/minimax_m2.py` modified +0/-6 (6 lines); hunks: -1069,12 +1069,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
  - `python/sglang/srt/models/mimo_v2.py` modified +0/-4 (4 lines); hunks: -808,10 +808,6 @@ def op_comm_prepare_mlp(self, state):; symbols: op_comm_prepare_mlp, op_mlp, op_comm_postprocess_layer
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/deepseek_v2.py
@@ -2114,19 +2114,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe.py
@@ -1017,19 +1017,6 @@ def op_comm_prepare_mlp(self, state):
-    def op_mlp(self, state):
-        hidden_states = state.pop("hidden_states_mlp_input")
-        if not (
-            enable_moe_dense_fully_dp()
-            and (not self.is_layer_sparse)
-            and hidden_states.shape[0] == 0
diff -- python/sglang/srt/models/glm4_moe_lite.py
@@ -737,19 +737,6 @@ def op_comm_prepare_mlp(self, state):
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/deepseek_v2.py` modified +0/-13; `python/sglang/srt/models/glm4_moe.py` modified +0/-13; `python/sglang/srt/models/glm4_moe_lite.py` modified +0/-13; `python/sglang/srt/models/minimax_m2.py` modified +0/-6; `python/sglang/srt/models/mimo_v2.py` modified +0/-4; `python/sglang/srt/models/qwen3_moe.py` modified +0/-4
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/deepseek_v2.py`, `python/sglang/srt/models/glm4_moe.py`, `python/sglang/srt/models/glm4_moe_lite.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26468 - [Model] Add Qwen3-MoE MTP

- Link: https://github.com/sgl-project/sglang/pull/26468
- Status/date: merged / 2026-05-29
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3_moe_mtp.py`; associated commits `cf66693b3530`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +163/-4, 206 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add Qwen3-MoE MTP"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/qwen3_moe_mtp.py`, `python/sglang/srt/models/qwen3_moe.py`; technical summary: Covers "[Model] Add Qwen3-MoE MTP"; the main implementation surface is `python/sglang/srt/models/qwen3_moe_mtp.py`, `python/sglang/srt/models/qwen3_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3_moe_mtp.py` added +131/-0 (131 lines); hunks: -0,0 +1,131; symbols: Qwen3MoeForCausalLMMTP, __init__, set_embed_and_head, forward, touching `Qwen3MoeForCausalLMMTP, __init__, set_embed_and_head`; `python/sglang/srt/models/qwen3_moe.py` modified +18/-1 (19 lines); hunks: -1111,7 +1111,9 @@ def set_dflash_layers_to_capture(self, layer_ids: List[int]):; -1132,6 +1134,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: set_dflash_layers_to_capture, load_weights, touching `set_dflash_layers_to_capture, load_weights`.
- Code diff details:
  - `python/sglang/srt/models/qwen3_moe_mtp.py` added +131/-0 (131 lines); hunks: -0,0 +1,131; symbols: Qwen3MoeForCausalLMMTP, __init__, set_embed_and_head, forward
  - `python/sglang/srt/models/qwen3_moe.py` modified +18/-1 (19 lines); hunks: -1111,7 +1111,9 @@ def set_dflash_layers_to_capture(self, layer_ids: List[int]):; -1132,6 +1134,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch...; symbols: set_dflash_layers_to_capture, load_weights
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3_moe_mtp.py
@@ -0,0 +1,131 @@
+# Copyright 2023-2024 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/models/qwen3_moe.py
@@ -1111,7 +1111,9 @@ def set_dflash_layers_to_capture(self, layer_ids: List[int]):
-    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+    def load_weights(
+        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
+    ):
@@ -1132,6 +1134,21 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
+            if is_mtp:
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3_moe_mtp.py` added +131/-0; `python/sglang/srt/models/qwen3_moe.py` modified +18/-1
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/configs/model_config.py`, `python/sglang/srt/models/qwen3_moe.py`, `python/sglang/srt/models/qwen3_moe_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26798 - Make qwen3's set_embed_and_head idempotent

- Link: https://github.com/sgl-project/sglang/pull/26798
- Status/date: merged / 2026-05-31
- Trace source: `git log --name-only -- <model-files>` found it through `python/sglang/srt/models/qwen3.py`; associated commits `13ca55afa3f8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 13 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Make qwen3's set_embed_and_head idempotent"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/models/qwen3.py`; technical summary: Covers "Make qwen3's set_embed_and_head idempotent"; the main implementation surface is `python/sglang/srt/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head, touching `get_embed_and_head, set_embed_and_head`.
- Code diff details:
  - `python/sglang/srt/models/qwen3.py` modified +4/-2 (6 lines); hunks: -674,8 +674,10 @@ def get_embed_and_head(self):; symbols: get_embed_and_head, set_embed_and_head
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/qwen3.py
@@ -674,8 +674,10 @@ def get_embed_and_head(self):
-        del self.model.embed_tokens.weight
-        del self.lm_head.weight
+        if hasattr(self.model.embed_tokens, "weight"):
+            del self.model.embed_tokens.weight
+        if hasattr(self.lm_head, "weight"):
+            del self.lm_head.weight
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/qwen3.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `python/sglang/srt/models/qwen3.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25813 - docs(cookbook): port popular model usage guides into cookbook pages

- Link: https://github.com/sgl-project/sglang/pull/25813
- Status/date: merged / 2026-06-02
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 47 files, +1262/-2154, 4187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "docs(cookbook): port popular model usage guides into cookbook pages"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`; technical summary: Covers "docs(cookbook): port popular model usage guides into cookbook pages"; the main implementation surface is `docs_new/docs/basic_usage/deepseek_v32.mdx`, `docs_new/docs/basic_usage/deepseek_v3.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64, touching `image_to_base64`.
- Code diff details:
  - `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601 (601 lines); hunks: -1,601 +0,0
  - `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375 (375 lines); hunks: -1,375 +0,0
  - `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3 (247 lines); hunks: -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose t...; -37,7 +58,18 @@ import { DeepSeekV32Deployment } from "/src/snippets/autoregr...
  - `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26 (182 lines); hunks: -10,7 +10,7 @@ GLM-4.6V series model includes two versions: GLM-4.6V (106B),...; -70,14 +70,56 @@ import { GLM46VDeployment } from "/src/snippets/autoregressi...; symbols: image_to_base64
  - `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181 (181 lines); hunks: -1,181 +0,0
- Key code excerpts:

```diff
diff -- docs_new/docs/basic_usage/deepseek_v32.mdx
@@ -1,601 +0,0 @@
-title: "DeepSeek V3.2/GLM-5 Usage"
-metatags:
-    description: "Deploy DeepSeek V3.2/GLM-5 with SGLang: DeepSeek Sparse Attention (DSA), long-context optimization, MTP speculative decoding, function calling. Supports H200, B2
-DeepSeek-V3.2 model family equips DeepSeek-V3.1-Terminus with DeepSeek Sparse Attention (DSA) through continued training. With DSA, a fine-grained sparse attention mechanism power
-Note: This document is originally written for the usage of [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) model. The usage of [DeepSeek-V3.2](https://hu
-## Installation
diff -- docs_new/docs/basic_usage/deepseek_v3.mdx
@@ -1,375 +0,0 @@
-title: "DeepSeek V3/V3.1/R1 Usage"
-metatags:
-    description: "Deploy DeepSeek V3/R1 with SGLang: MLA optimization, FP8 quantization, multi-node TP, DP attention, MTP speculative decoding. Supports H200, B200, MI300X, A100."
-SGLang provides many optimizations specifically designed for the DeepSeek models, making it the inference engine recommended by the official [DeepSeek team](https://github.com/dee
-This document outlines current optimizations for DeepSeek.
-For an overview of the implemented features see the completed [Roadmap](https://github.com/sgl-project/sglang/issues/2591).
diff -- docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx
@@ -24,6 +24,27 @@ SGLang offers multiple installation methods. You can choose the most suitable in
```

- Reviewed files:
  - docs: `docs_new/docs/basic_usage/deepseek_v32.mdx` removed +0/-601; `docs_new/docs/basic_usage/deepseek_v3.mdx` removed +0/-375; `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-V3_2.mdx` modified +244/-3; `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx` modified +156/-26; `docs_new/docs/basic_usage/gpt_oss.mdx` removed +0/-181; `docs_new/docs/basic_usage/glmv.mdx` removed +0/-139
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR-2.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-OCR.mdx`, `docs_new/cookbook/autoregressive/DeepSeek/DeepSeek-R1.mdx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #23906 - [Refactor] Cuda Graph Runner/Backend Refactor

- Link: https://github.com/sgl-project/sglang/pull/23906
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 160 files, +5197/-3068, 12233 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Cuda Graph Runner/Backend Refactor"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`; technical summary: Covers "[Refactor] Cuda Graph Runner/Backend Refactor"; the main implementation surface is `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`, `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool, touching `freeze_gc, _to_torch, patch_model`; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype, touching `PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled`; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode, touching `_make_graph_key, build_replay_fb_view, _allocate_decode_buffers`; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers, touching `BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank`.
- Code diff details:
  - `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860 (860 lines); hunks: -1,860 +0,0; symbols: freeze_gc, _to_torch, patch_model, get_global_graph_memory_pool
  - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0 (846 lines); hunks: -0,0 +1,846; symbols: PrefillCudaGraphRunner, __init__, _is_mamba_track_enabled, _cache_loc_dtype
  - `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463 (757 lines); hunks: -1,4 +1,4; -11,33 +11,36; symbols: _make_graph_key, build_replay_fb_view, _allocate_decode_buffers, get_is_capture_mode
  - `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541 (541 lines); hunks: -1,541 +0,0; symbols: BreakableCudaGraphRunner, __init__, _has_inactive_dp_rank, _init_buffers
  - `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0 (442 lines); hunks: -0,0 +1,442; symbols: _grouped_foreach_copy_, foreach_copy, DecodeInputBuffers, create
- Key code excerpts:

```diff
diff -- python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
@@ -1,860 +0,0 @@
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
-# You may obtain a copy of the License at
-#
-#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
@@ -0,0 +1,846 @@
+# Copyright 2023-2026 SGLang Team
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
+# You may obtain a copy of the License at
+#
+#     http://www.apache.org/licenses/LICENSE-2.0
diff -- python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py
@@ -1,4 +1,4 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` removed +0/-860; `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` added +846/-0; `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` renamed +294/-463; `python/sglang/srt/model_executor/breakable_cuda_graph_runner.py` removed +0/-541; `python/sglang/srt/model_executor/runner_utils/buffers.py` added +442/-0; `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py` added +225/-0
- Risk and verification: The diff ships test coverage in `python/sglang/test/doc_patch.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #9147 - support Qwen3-MoE-w4afp8

- Link: https://github.com/sgl-project/sglang/pull/9147
- Status/date: closed / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 636 files, +14735/-62339, 94998 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support Qwen3-MoE-w4afp8"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py`; technical summary: Covers "support Qwen3-MoE-w4afp8"; the main implementation surface is `python/sglang/srt/models/phi4mm_utils.py`, `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py`, `python/sglang/srt/entrypoints/openai/serving_responses.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunks: -1,1917 +0,0; symbols: BlockBase, __init__, get_activation, adaptive_enc_mask, touching `BlockBase, __init__, get_activation`; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunks: -1,1700 +0,0; symbols: DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config, touching `DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__`; `python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273 (1273 lines); hunks: -1,1273 +0,0; symbols: OpenAIServingResponses, __init__, _request_id_prefix, create_responses, touching `OpenAIServingResponses, __init__, _request_id_prefix`; `python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260 (1260 lines); hunks: -1,1260 +0,0; symbols: ConformerEncoderLayer, __init__, forward, TransformerEncoderBase, touching `ConformerEncoderLayer, __init__, forward`.
- Code diff details:
  - `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917 (1917 lines); hunks: -1,1917 +0,0; symbols: BlockBase, __init__, get_activation, adaptive_enc_mask
  - `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700 (1700 lines); hunks: -1,1700 +0,0; symbols: DualChunkFlashAttentionMetadata, DualChunkFlashAttentionBackend, __init__, get_sparse_attention_config
  - `python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273 (1273 lines); hunks: -1,1273 +0,0; symbols: OpenAIServingResponses, __init__, _request_id_prefix, create_responses
  - `python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260 (1260 lines); hunks: -1,1260 +0,0; symbols: ConformerEncoderLayer, __init__, forward, TransformerEncoderBase
  - `python/sglang/srt/models/gpt_oss.py` removed +0/-1134 (1134 lines); hunks: -1,1134 +0,0; symbols: GptOssConfig, __init__, get_attention_sliding_window_size, GptOssSparseMoeBlock
- Key code excerpts:

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

- Reviewed files:
  - runtime: `python/sglang/srt/models/phi4mm_utils.py` removed +0/-1917; `python/sglang/srt/layers/attention/dual_chunk_flashattention_backend.py` removed +0/-1700; `python/sglang/srt/entrypoints/openai/serving_responses.py` removed +0/-1273; `python/sglang/srt/models/phi4mm_audio.py` removed +0/-1260; `python/sglang/srt/models/gpt_oss.py` removed +0/-1134; `python/sglang/srt/layers/moe/ep_moe/layer.py` modified +856/-275
- Risk and verification: The diff ships test coverage in `python/sglang/test/attention/test_trtllm_mla_backend.py`, `python/sglang/test/few_shot_gsm8k.py`, `python/sglang/test/few_shot_gsm8k_engine.py`, `python/sglang/test/run_eval.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22837 - [Bug] Qwen3 reasoning detector silently swallows tool_call when is missing

- Link: https://github.com/sgl-project/sglang/pull/22837
- Status/date: closed / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +43/-0, 57 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bug] Qwen3 reasoning detector silently swallows tool_call when is missing"; model line: Qwen3 Core; category: bug fix; main diff: `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`; technical summary: Covers "[Bug] Qwen3 reasoning detector silently swallows tool_call when is missing"; the main implementation surface is `test/registered/unit/parser/test_reasoning_parser.py`, `python/sglang/srt/parser/reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunks: -269,6 +269,48 @@ def test_streaming_qwen3_forced_reasoning_format(self):; symbols: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector, touching `test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close`; `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0 (42 lines); hunks: -269,6 +269,48 @@ def test_streaming_qwen3_forced_reasoning_format(self):; symbols: test_streaming_qwen3_forced_reasoning_format, test_detect_and_parse_tool_call_without_think_close, test_streaming_tool_call_without_think_close, TestKimiDetector
  - `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0 (1 lines); hunks: -242,6 +242,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - tests: `test/registered/unit/parser/test_reasoning_parser.py` modified +42/-0
  - runtime: `python/sglang/srt/parser/reasoning_parser.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `test/registered/unit/parser/test_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28567 - Add get_parallel(): a structured accessor for parallel-topology state

- Link: https://github.com/sgl-project/sglang/pull/28567
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 184 files, +1865/-1727, 8932 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_parallel(): a structured accessor for parallel-topology state"; model line: Qwen3 Core; category: model support/runtime entry; main diff: `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`; technical summary: Covers "Add get_parallel(): a structured accessor for parallel-topology state"; the main implementation surface is `python/sglang/srt/models/apertus.py`, `python/sglang/srt/models/solar.py`, `python/sglang/srt/models/gpt_oss.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention, touching `ApertusMLP, __init__, forward`; `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales, touching `__init__, forward, load_kv_cache_scales`; `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__, touching `_resolve_moe_input_pad_multiple, __init__`; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__, touching `__init__`.
- Code diff details:
  - `python/sglang/srt/models/apertus.py` modified +686/-687 (1373 lines); hunks: -1,687 +1,686; symbols: ApertusMLP, __init__, forward, ApertusAttention
  - `python/sglang/srt/models/solar.py` modified +28/-27 (55 lines); hunks: -1,37 +1,14; -54,6 +31,30; symbols: __init__, forward, load_kv_cache_scales
  - `python/sglang/srt/models/gpt_oss.py` modified +17/-24 (41 lines); hunks: -28,21 +28,13; -76,6 +68,7; symbols: _resolve_moe_input_pad_multiple, __init__
  - `python/sglang/srt/models/deepseek_v2.py` modified +14/-23 (37 lines); hunks: -47,9 +47,7; -72,12 +70,6; symbols: __init__
  - `python/sglang/srt/layers/communicator.py` modified +13/-19 (32 lines); hunks: -23,8 +23,6; -44,12 +42,7; symbols: apply_aiter_all_reduce_fusion, init_context, should_fuse_mlp_allreduce_with_next_layer, is_same_group_size
- Key code excerpts:

```diff
diff -- python/sglang/srt/models/apertus.py
@@ -1,687 +1,686 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-# Copyright 2025 The SwissAI Initiative
-# Copyright 2023-2024 SGLang Team
-# Licensed under the Apache License, Version 2.0 (the "License");
-# you may not use this file except in compliance with the License.
diff -- python/sglang/srt/models/solar.py
@@ -1,37 +1,14 @@
-# Adapted from
-# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
-# Copyright 2023 The vLLM team.
-# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
-#
-# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
diff -- python/sglang/srt/models/gpt_oss.py
@@ -28,21 +28,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/models/apertus.py` modified +686/-687; `python/sglang/srt/models/solar.py` modified +28/-27; `python/sglang/srt/models/gpt_oss.py` modified +17/-24; `python/sglang/srt/models/deepseek_v2.py` modified +14/-23; `python/sglang/srt/layers/communicator.py` modified +13/-19; `python/sglang/srt/models/qwen3_moe.py` modified +12/-18
- Risk and verification: The diff ships test coverage in `python/sglang/test/kits/attention_unittest/attention_methods/dense_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsa_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dsv4_attention.py`, `python/sglang/test/kits/attention_unittest/attention_methods/dual_chunk_attention.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28421 - [3/N][CP] Implement zigzag CP strategy

- Link: https://github.com/sgl-project/sglang/pull/28421
- Status/date: merged / 2026-06-18
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1091/-56, 1441 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[3/N][CP] Implement zigzag CP strategy"; model line: Qwen3 Core; category: model implementation change; main diff: `python/sglang/srt/layers/cp/zigzag.py`, `python/sglang/srt/layers/cp/utils.py`, `python/sglang/srt/model_executor/model_runner.py`; technical summary: Covers "[3/N][CP] Implement zigzag CP strategy"; the main implementation surface is `python/sglang/srt/layers/cp/zigzag.py`, `python/sglang/srt/layers/cp/utils.py`, `python/sglang/srt/model_executor/model_runner.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `python/sglang/srt/layers/cp/zigzag.py` modified +264/-15 (279 lines); hunks: -30,15 +30,29; -85,40 +99,207 @@ def can_apply(self, num_tokens: int, forward_batch) -> bool:; symbols: can_apply, build_metadata, shard_hidden_states, shard_position_ids, touching `can_apply, build_metadata, shard_hidden_states`; `python/sglang/srt/layers/cp/utils.py` modified +101/-1 (102 lines); hunks: -12,13 +12,16; -30,6 +33,96; symbols: enable_cp_v2, is_cp_v2_active, prepare_cp_forward, cp_split_before_forward, touching `enable_cp_v2, is_cp_v2_active, prepare_cp_forward`; `python/sglang/srt/model_executor/model_runner.py` modified +64/-3 (67 lines); hunks: -123,6 +123,13; -3367,6 +3374,8 @@ def forward_extend(; symbols: forward_extend, touching `forward_extend`; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +39/-17 (56 lines); hunks: -13,6 +13,8; -811,18 +813,26 @@ def forward_extend(; symbols: forward_extend, _fa_cp_attn, touching `forward_extend, _fa_cp_attn`.
- Code diff details:
  - `python/sglang/srt/layers/cp/zigzag.py` modified +264/-15 (279 lines); hunks: -30,15 +30,29; -85,40 +99,207 @@ def can_apply(self, num_tokens: int, forward_batch) -> bool:; symbols: can_apply, build_metadata, shard_hidden_states, shard_position_ids
  - `python/sglang/srt/layers/cp/utils.py` modified +101/-1 (102 lines); hunks: -12,13 +12,16; -30,6 +33,96; symbols: enable_cp_v2, is_cp_v2_active, prepare_cp_forward, cp_split_before_forward
  - `python/sglang/srt/model_executor/model_runner.py` modified +64/-3 (67 lines); hunks: -123,6 +123,13; -3367,6 +3374,8 @@ def forward_extend(; symbols: forward_extend
  - `python/sglang/srt/layers/attention/flashattention_backend.py` modified +39/-17 (56 lines); hunks: -13,6 +13,8; -811,18 +813,26 @@ def forward_extend(; symbols: forward_extend, _fa_cp_attn
  - `python/sglang/srt/layers/cp/base.py` modified +4/-8 (12 lines); hunks: -185,6 +185,7 @@ def materialize_full_kv(; -235,7 +236,7 @@ def init_cp_strategy(server_args: ServerArgs) -> None:; symbols: materialize_full_kv, init_cp_strategy, _get_cp_strategy, get_cp_strategy
- Key code excerpts:

```diff
diff -- python/sglang/srt/layers/cp/zigzag.py
@@ -30,15 +30,29 @@
+from contextlib import nullcontext
+from itertools import accumulate
+import torch
+import torch.nn.functional as F
+from sglang.srt.distributed.device_communicators.pynccl_allocator import (
+    use_symmetric_memory,
diff -- python/sglang/srt/layers/cp/utils.py
@@ -12,13 +12,16 @@
-"""Public import facade for context parallel strategy helpers."""
+"""Public import facade and runtime helpers for context parallel strategies."""
+from typing import Any, Optional, Tuple
+    get_cp_strategy,
@@ -30,6 +33,96 @@
+CP_V2_DEFAULT_MODEL_CLASSES = frozenset(
diff -- python/sglang/srt/model_executor/model_runner.py
@@ -123,6 +123,13 @@
```

- Reviewed files:
  - runtime: `python/sglang/srt/layers/cp/zigzag.py` modified +264/-15; `python/sglang/srt/layers/cp/utils.py` modified +101/-1; `python/sglang/srt/model_executor/model_runner.py` modified +64/-3; `python/sglang/srt/layers/attention/flashattention_backend.py` modified +39/-17; `python/sglang/srt/layers/cp/base.py` modified +4/-8; `python/sglang/srt/layers/cp/interleave.py` modified +3/-1
- Risk and verification: The diff ships test coverage in `test/registered/cp/test_cp_strategy_unit.py`, `test/registered/cp/test_gqa_prefill_cp_legacy.py`, `test/registered/cp/test_gqa_preill_cp.py`, `test/registered/unit/server_args/test_server_args.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #28697 - [docs] Add B300 cookbook deployment options

- Link: https://github.com/sgl-project/sglang/pull/28697
- Status/date: merged / 2026-06-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 27 files, +503/-69, 1291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[docs] Add B300 cookbook deployment options"; model line: Qwen3 Core; category: performance/backend optimization; main diff: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`; technical summary: Covers "[docs] Add B300 cookbook deployment options"; the main implementation surface is `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`, `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {.
- Code diff details:
  - `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0 (167 lines); hunks: -0,0 +1,167
  - `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2 (70 lines); hunks: -9,6 +9,11 @@ const lookupData = {; -182,6 +187,66 @@ const lookupData = {
  - `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16 (56 lines); hunks: -4,6 +4,7 @@ export const GLM5Deployment = () => {; -13,6 +14,7 @@ export const GLM5Deployment = () => {
  - `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10 (39 lines); hunks: -3,7 +3,7 @@ export const DeepSeekV32Deployment = () => {; -12,6 +12,7 @@ export const DeepSeekV32Deployment = () => {
  - `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15 (38 lines); hunks: -8,19 +8,19 @@ export const Qwen35Deployment = () => {; -149,7 +149,7 @@ export const Qwen35Deployment = () => {
- Key code excerpts:

```diff
diff -- docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx
@@ -0,0 +1,167 @@
+export const InternS1Deployment = () => {
+  const options = {
+    hardware: {
+      name: 'hardware',
+      title: 'Hardware Platform',
+      items: [
diff -- docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx
@@ -9,6 +9,11 @@ const lookupData = {
+      {
+        "id": "b300",
+        "label": "B300",
+        "default": false
+      },
@@ -182,6 +187,66 @@ const lookupData = {
diff -- docs_new/src/snippets/autoregressive/glm-5-deployment.jsx
@@ -4,6 +4,7 @@ export const GLM5Deployment = () => {
```

- Reviewed files:
  - docs: `docs_new/src/snippets/autoregressive/intern-s1-deployment.jsx` added +167/-0; `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx` modified +68/-2; `docs_new/src/snippets/autoregressive/glm-5-deployment.jsx` modified +40/-16; `docs_new/src/snippets/autoregressive/deepseek-v32-deployment.jsx` modified +29/-10; `docs_new/src/snippets/autoregressive/qwen35-deployment.jsx` modified +23/-15; `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx` modified +16/-13
- Risk and verification: This is mostly docs/examples in `docs_new/cookbook/autoregressive/InternLM/Intern-S1.mdx`, `docs_new/src/snippets/autoregressive/deepseek-math-v2-deployment.jsx`, `docs_new/src/snippets/autoregressive/deepseek-r1-advanced-deployment.jsx`; validation should confirm the documented command still maps to current CLI flags and model repo names.

### PR #28810 - [CI] Remove deprecated test/srt legacy CI setup

- Link: https://github.com/sgl-project/sglang/pull/28810
- Status/date: merged / 2026-06-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 29 files, +2/-5773, 5826 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[CI] Remove deprecated test/srt legacy CI setup"; model line: Qwen3 Core; category: docs/tests/CI; main diff: `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`; technical summary: Covers "[CI] Remove deprecated test/srt legacy CI setup"; the main implementation surface is `test/srt/cpu/test_qkv_proj_with_rope.py`, `test/srt/cpu/utils.py`, `test/srt/cpu/test_norm.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8, touching `layernorm, rotary_emb, native_torch`; `test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul, touching `parametrize, decorator, wrapper`; `test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native, touching `TestNorm, _forward_native, _norm`; `test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink, touching `TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend`.
- Code diff details:
  - `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: layernorm, rotary_emb, native_torch, native_torch_int8
  - `test/srt/cpu/utils.py` removed +0/-440 (440 lines); hunks: -1,440 +0,0; symbols: parametrize, decorator, wrapper, SiluAndMul
  - `test/srt/cpu/test_norm.py` removed +0/-432 (432 lines); hunks: -1,432 +0,0; symbols: TestNorm, _forward_native, _norm, _gemma3_rmsnorm_native
  - `test/srt/cpu/test_extend.py` removed +0/-400 (400 lines); hunks: -1,400 +0,0; symbols: TestExtendAttention, _scaled_dot_product_attention, _run_sdpa_forward_extend, _run_sdpa_forward_extend_sink
  - `test/srt/cpu/test_mamba.py` removed +0/-394 (394 lines); hunks: -1,394 +0,0; symbols: l2norm, torch_chunk_gated_delta_rule, chunk_gated_delta_rule_update, torch_recurrent_gated_delta_rule
- Key code excerpts:

```diff
diff -- test/srt/cpu/test_qkv_proj_with_rope.py
@@ -1,440 +0,0 @@
-import unittest
-import torch
-from utils import (
-    convert_weight,
-    native_w8a8_per_token_matmul,
-    per_token_quant_int8,
diff -- test/srt/cpu/utils.py
@@ -1,440 +0,0 @@
-import itertools
-import math
-import torch
-import torch.nn.functional as F
-precision = {
-    torch.bfloat16: 1e-2,
diff -- test/srt/cpu/test_norm.py
@@ -1,432 +0,0 @@
```

- Reviewed files:
  - tests: `test/srt/cpu/test_qkv_proj_with_rope.py` removed +0/-440; `test/srt/cpu/utils.py` removed +0/-440; `test/srt/cpu/test_norm.py` removed +0/-432; `test/srt/cpu/test_extend.py` removed +0/-400; `test/srt/cpu/test_mamba.py` removed +0/-394; `test/srt/cpu/test_moe.py` removed +0/-352
- Risk and verification: The diff ships test coverage in `test/README.md`, `test/srt/cpu/arm64/test_moe.py`, `test/srt/cpu/test_activation.py`, `test/srt/cpu/test_binding.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
