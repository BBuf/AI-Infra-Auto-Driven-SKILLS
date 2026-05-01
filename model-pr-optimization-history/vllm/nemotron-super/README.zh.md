# vllm Nemotron Super 模型 PR 优化历史

## 覆盖范围

- 重做日期: 2026-05-01
- 源码基线: `vllm-project/vllm` 当前追溯 worktree commit `7075df79b3`
- PR 收集规则: 先从模型实现、配置、processor、parser、docs/tests 等相关文件执行 `git log --name-only -- <model-files>`，再按 commit subject 的模型关键词过滤，最后用 GitHub Pull Request files API 读取每个 PR 的最终 diff。
- 额外保留规则: 原 history/skill 已显式引用但未出现在当前实现文件 git trace 中的 PR 会保留，并在卡片里标注来源。

## 模型实现文件覆盖

| 文件 | git 追溯到的 PR |
| --- | --- |
| `examples/pooling/embed/template/nemotron_embed_vl.jinja` | [#35297](https://github.com/vllm-project/vllm/pull/35297) |
| `examples/pooling/score/template/nemotron-rerank.jinja` | 无直接 PR 号提交 |
| `examples/pooling/score/template/nemotron-vl-rerank.jinja` | [#35735](https://github.com/vllm-project/vllm/pull/35735) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` | [#34725](https://github.com/vllm-project/vllm/pull/34725) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` | [#34725](https://github.com/vllm-project/vllm/pull/34725) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-vllm-cutlass.yaml` | 无直接 PR 号提交 |
| `tests/models/language/pooling_mteb_test/test_nemotron.py` | 无直接 PR 号提交 |
| `tests/models/multimodal/generation/test_nemotron_parse.py` | [#30864](https://github.com/vllm-project/vllm/pull/30864), [#37407](https://github.com/vllm-project/vllm/pull/37407) |
| `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` | [#35735](https://github.com/vllm-project/vllm/pull/35735), [#37613](https://github.com/vllm-project/vllm/pull/37613) |
| `tests/models/multimodal/processing/test_nemotron_vl.py` | [#20349](https://github.com/vllm-project/vllm/pull/20349), [#22739](https://github.com/vllm-project/vllm/pull/22739) |
| `tests/reasoning/test_nemotron_v3_reasoning_parser.py` | [#36393](https://github.com/vllm-project/vllm/pull/36393), [#36635](https://github.com/vllm-project/vllm/pull/36635) |
| `vllm/model_executor/models/nano_nemotron_vl.py` | [#23644](https://github.com/vllm-project/vllm/pull/23644), [#25708](https://github.com/vllm-project/vllm/pull/25708), [#26186](https://github.com/vllm-project/vllm/pull/26186), [#26269](https://github.com/vllm-project/vllm/pull/26269), [#27107](https://github.com/vllm-project/vllm/pull/27107), [#30864](https://github.com/vllm-project/vllm/pull/30864), [#32121](https://github.com/vllm-project/vllm/pull/32121), [#32682](https://github.com/vllm-project/vllm/pull/32682), [#35100](https://github.com/vllm-project/vllm/pull/35100), [#35539](https://github.com/vllm-project/vllm/pull/35539), [#35657](https://github.com/vllm-project/vllm/pull/35657), [#36808](https://github.com/vllm-project/vllm/pull/36808), ... (21 total) |
| `vllm/model_executor/models/nemotron.py` | [#6611](https://github.com/vllm-project/vllm/pull/6611), [#7611](https://github.com/vllm-project/vllm/pull/7611) |
| `vllm/model_executor/models/nemotron_h.py` | [#18863](https://github.com/vllm-project/vllm/pull/18863), [#19249](https://github.com/vllm-project/vllm/pull/19249), [#22349](https://github.com/vllm-project/vllm/pull/22349), [#25863](https://github.com/vllm-project/vllm/pull/25863), [#27968](https://github.com/vllm-project/vllm/pull/27968), [#30802](https://github.com/vllm-project/vllm/pull/30802), [#31539](https://github.com/vllm-project/vllm/pull/31539), [#31807](https://github.com/vllm-project/vllm/pull/31807), [#31898](https://github.com/vllm-project/vllm/pull/31898), [#32265](https://github.com/vllm-project/vllm/pull/32265), [#32549](https://github.com/vllm-project/vllm/pull/32549), [#32669](https://github.com/vllm-project/vllm/pull/32669), ... (17 total) |
| `vllm/model_executor/models/nemotron_h_mtp.py` | [#33726](https://github.com/vllm-project/vllm/pull/33726), [#37803](https://github.com/vllm-project/vllm/pull/37803) |
| `vllm/model_executor/models/nemotron_nas.py` | [#15008](https://github.com/vllm-project/vllm/pull/15008), [#18427](https://github.com/vllm-project/vllm/pull/18427), [#30795](https://github.com/vllm-project/vllm/pull/30795) |
| `vllm/model_executor/models/nemotron_parse.py` | [#30864](https://github.com/vllm-project/vllm/pull/30864), [#33189](https://github.com/vllm-project/vllm/pull/33189), [#37407](https://github.com/vllm-project/vllm/pull/37407), [#37456](https://github.com/vllm-project/vllm/pull/37456) |
| `vllm/model_executor/models/nemotron_vl.py` | [#20349](https://github.com/vllm-project/vllm/pull/20349), [#22023](https://github.com/vllm-project/vllm/pull/22023), [#22739](https://github.com/vllm-project/vllm/pull/22739), [#35297](https://github.com/vllm-project/vllm/pull/35297), [#35735](https://github.com/vllm-project/vllm/pull/35735), [#36192](https://github.com/vllm-project/vllm/pull/36192) |
| `vllm/reasoning/nemotron_v3_reasoning_parser.py` | [#36393](https://github.com/vllm-project/vllm/pull/36393), [#36635](https://github.com/vllm-project/vllm/pull/36635) |
| `vllm/transformers_utils/configs/nemotron.py` | [#6611](https://github.com/vllm-project/vllm/pull/6611), [#7611](https://github.com/vllm-project/vllm/pull/7611), [#20349](https://github.com/vllm-project/vllm/pull/20349) |
| `vllm/transformers_utils/configs/nemotron_h.py` | [#18863](https://github.com/vllm-project/vllm/pull/18863), [#22349](https://github.com/vllm-project/vllm/pull/22349), [#25863](https://github.com/vllm-project/vllm/pull/25863), [#33726](https://github.com/vllm-project/vllm/pull/33726) |
| `vllm/transformers_utils/processors/nano_nemotron_vl.py` | [#36808](https://github.com/vllm-project/vllm/pull/36808), [#37903](https://github.com/vllm-project/vllm/pull/37903), [#38538](https://github.com/vllm-project/vllm/pull/38538), [#38655](https://github.com/vllm-project/vllm/pull/38655), [#40283](https://github.com/vllm-project/vllm/pull/40283) |
| `vllm/transformers_utils/processors/nemotron_vl.py` | 无直接 PR 号提交 |

## PR 覆盖总览

- git 追溯 PR 数: 60
- 原文档显式引用补充 PR 数: 2
- 当前文档总 PR 数: 62
- 文件追溯命令: `git log --name-only -- <model-files>`
- diff 审计来源: GitHub Pull Request files API

## 时间线

| 日期 | PR | 状态 | 标题 | 主要文件 |
| --- | --- | --- | --- | --- |
| 2024-07-26 | [#6611](https://github.com/vllm-project/vllm/pull/6611) | merged | [Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron) | `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py` |
| 2024-08-16 | [#7611](https://github.com/vllm-project/vllm/pull/7611) | merged | [Model] Align nemotron config with final HF state and fix lm-eval-small | `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/nemotron.py` |
| 2025-03-31 | [#15008](https://github.com/vllm-project/vllm/pull/15008) | merged | [Model] Update support for NemotronNAS models | `vllm/model_executor/models/nemotron_nas.py` |
| 2025-05-26 | [#18427](https://github.com/vllm-project/vllm/pull/18427) | merged | [Model] Add support for YARN in NemotronNAS models | `vllm/model_executor/models/nemotron_nas.py` |
| 2025-06-05 | [#18863](https://github.com/vllm-project/vllm/pull/18863) | merged | [Model] NemotronH support | `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py` |
| 2025-06-06 | [#19249](https://github.com/vllm-project/vllm/pull/19249) | merged | [Model] Optimize nemotron_h implementation | `vllm/model_executor/models/nemotron_h.py` |
| 2025-07-17 | [#20349](https://github.com/vllm-project/vllm/pull/20349) | merged | [VLM] Add Nemotron-Nano-VL-8B-V1 support | `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py` |
| 2025-08-11 | [#22349](https://github.com/vllm-project/vllm/pull/22349) | merged | [Model] NemotronH Support | `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py` |
| 2025-08-13 | [#22739](https://github.com/vllm-project/vllm/pull/22739) | merged | [Bugfix] Fix Nemotron VL image processing | `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py` |
| 2025-08-19 | [#22023](https://github.com/vllm-project/vllm/pull/22023) | merged | Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema | `vllm/model_executor/models/nemotron_vl.py` |
| 2025-09-10 | [#23644](https://github.com/vllm-project/vllm/pull/23644) | merged | Support for NemotronH Nano VLM | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2025-09-25 | [#25708](https://github.com/vllm-project/vllm/pull/25708) | merged | [Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2 | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2025-09-26 | [#22980](https://github.com/vllm-project/vllm/pull/22980) | merged | EVS Support (Video tokens pruning) | `vllm/multimodal/evs.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py` |
| 2025-10-04 | [#26186](https://github.com/vllm-project/vllm/pull/26186) | merged | Fix issue of using only the part of video frame [Nemotron Nano] | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2025-10-06 | [#26269](https://github.com/vllm-project/vllm/pull/26269) | merged | [Model] EVS support for nano_nemotron_vl | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2025-10-20 | [#27107](https://github.com/vllm-project/vllm/pull/27107) | merged | Nemotron Nano V2 VL + EVS Video Support | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2025-10-23 | [#25863](https://github.com/vllm-project/vllm/pull/25863) | merged | [Model] Add MoE support for NemotronH | `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py` |
| 2025-11-04 | [#27968](https://github.com/vllm-project/vllm/pull/27968) | merged | [Model][Bugfix] fix pipeline parallelism support for NemotronH | `vllm/model_executor/models/nemotron_h.py` |
| 2025-12-17 | [#30795](https://github.com/vllm-project/vllm/pull/30795) | merged | Fix nemotron_nas intermediate_size computation | `vllm/model_executor/models/nemotron_nas.py` |
| 2025-12-31 | [#31539](https://github.com/vllm-project/vllm/pull/31539) | merged | Add get_expert_mapping to NemotronHModel (for LoRA support) | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-05 | [#30864](https://github.com/vllm-project/vllm/pull/30864) | merged | [Model] Nemotron Parse 1.1 Support | `vllm/model_executor/models/nemotron_parse.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-01-06 | [#31807](https://github.com/vllm-project/vllm/pull/31807) | merged | [NemotronH] Use ReplicatedLinear for fc1_latent_proj | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-07 | [#31898](https://github.com/vllm-project/vllm/pull/31898) | merged | Enable quantized attention in NemotronH models | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-19 | [#30802](https://github.com/vllm-project/vllm/pull/30802) | merged | Add support for LoRA adapters in Nemotron-H models | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-19 | [#32121](https://github.com/vllm-project/vllm/pull/32121) | merged | support dynamic resolution image encoding for Nemotron Nano VL | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-01-21 | [#32682](https://github.com/vllm-project/vllm/pull/32682) | merged | [Bugfix] Fix Nemotron-Nano-v2-vlm static resolution | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-01-27 | [#32265](https://github.com/vllm-project/vllm/pull/32265) | merged | [LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-27 | [#32549](https://github.com/vllm-project/vllm/pull/32549) | merged | Support heterogeneous NemotronHPuzzle model | `vllm/model_executor/models/nemotron_h.py` |
| 2026-01-29 | [#33189](https://github.com/vllm-project/vllm/pull/33189) | merged | [Misc][Build] Lazy load cv2 in nemotron_parse.py | `vllm/model_executor/models/nemotron_parse.py` |
| 2026-01-29 | [#32669](https://github.com/vllm-project/vllm/pull/32669) | merged | Bugfix: Pass router logits dtype in nemotron shared experts | `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-02 | [#32790](https://github.com/vllm-project/vllm/pull/32790) | merged | [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H) | `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-12 | [#33506](https://github.com/vllm-project/vllm/pull/33506) | merged | [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4 | `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` |
| 2026-02-16 | [#34582](https://github.com/vllm-project/vllm/pull/34582) | merged | [NemotronH] Do not force router to run in fp32 | `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-18 | [#34725](https://github.com/vllm-project/vllm/pull/34725) | merged | [Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4 | `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` |
| 2026-02-19 | [#34808](https://github.com/vllm-project/vllm/pull/34808) | merged | Revert "[NemotronH] Do not force router to run in fp32 (#34582)" | `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-24 | [#33726](https://github.com/vllm-project/vllm/pull/33726) | merged | [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support | `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-26 | [#35297](https://github.com/vllm-project/vllm/pull/35297) | merged | [Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model | `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/embed/template/nemotron_embed_vl.jinja` |
| 2026-02-26 | [#35396](https://github.com/vllm-project/vllm/pull/35396) | merged | Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models | `vllm/model_executor/models/nemotron_h.py` |
| 2026-02-27 | [#35100](https://github.com/vllm-project/vllm/pull/35100) | merged | Support parakeet as audio encoder for nemotron-nano-vl | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-03-03 | [#35735](https://github.com/vllm-project/vllm/pull/35735) | merged | [Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2 | `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/score/template/nemotron-vl-rerank.jinja` |
| 2026-03-04 | [#35539](https://github.com/vllm-project/vllm/pull/35539) | merged | Support Audio Extraction from MP4 Video for Nemotron Nano VL | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-03-06 | [#36192](https://github.com/vllm-project/vllm/pull/36192) | merged | [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25 | `vllm/model_executor/models/nemotron_vl.py` |
| 2026-03-08 | [#35657](https://github.com/vllm-project/vllm/pull/35657) | merged | [Model] Nano Nemotron VL - fast media preprocessing | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-03-09 | [#36393](https://github.com/vllm-project/vllm/pull/36393) | merged | add nemotron v3 reasoning parser | `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py` |
| 2026-03-11 | [#36635](https://github.com/vllm-project/vllm/pull/36635) | merged | [NemotronH] Small fix reasoning parser | `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py` |
| 2026-03-18 | [#37456](https://github.com/vllm-project/vllm/pull/37456) | merged | [Model] Remove unnecessary processor definition for Nemotron Parse | `vllm/transformers_utils/processors/nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py` |
| 2026-03-19 | [#36808](https://github.com/vllm-project/vllm/pull/36808) | merged | Support temporal compression for Nemotron-3-VL videos | `vllm/transformers_utils/processors/nano_nemotron_vl.py`, `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-03-19 | [#37407](https://github.com/vllm-project/vllm/pull/37407) | merged | [Bugfix] Fix Nemotron Parse loading | `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py` |
| 2026-03-20 | [#37613](https://github.com/vllm-project/vllm/pull/37613) | merged | [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests | `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` |
| 2026-03-22 | [#37803](https://github.com/vllm-project/vllm/pull/37803) | merged | Enable `NemotronHPuzzle` + `NemotronHMTP` | `vllm/model_executor/models/nemotron_h_mtp.py` |
| 2026-03-24 | [#36803](https://github.com/vllm-project/vllm/pull/36803) | merged | [Test] E2E Nemotron-3-Super tests | `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` |
| 2026-03-24 | [#37903](https://github.com/vllm-project/vllm/pull/37903) | merged | nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths | `vllm/transformers_utils/processors/nano_nemotron_vl.py` |
| 2026-03-26 | [#38018](https://github.com/vllm-project/vllm/pull/38018) | merged | [Model] Use helper function to run MM processors with token inputs (where applicable) | `vllm/transformers_utils/processors/pixtral.py`, `vllm/transformers_utils/processors/voxtral.py`, `vllm/multimodal/processing/processor.py` |
| 2026-03-30 | [#38567](https://github.com/vllm-project/vllm/pull/38567) | merged | Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018 | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-04-03 | [#38655](https://github.com/vllm-project/vllm/pull/38655) | merged | Fix Nano Nemotron VL regressions | `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py` |
| 2026-04-05 | [#39029](https://github.com/vllm-project/vllm/pull/39029) | merged | nano_nemotron_vl: fix tensor device mismatch exception when video profiling | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-04-07 | [#38727](https://github.com/vllm-project/vllm/pull/38727) | merged | nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-04-09 | [#38538](https://github.com/vllm-project/vllm/pull/38538) | merged | nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time | `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py` |
| 2026-04-10 | [#37580](https://github.com/vllm-project/vllm/pull/37580) | merged | Nemotron Nano VL: Streamline pixel shuffle | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-04-15 | [#39901](https://github.com/vllm-project/vllm/pull/39901) | merged | FIX: support language_model.backbone naming in NemotronH Nano VL quantization config | `vllm/model_executor/models/nano_nemotron_vl.py` |
| 2026-04-19 | [#40283](https://github.com/vllm-project/vllm/pull/40283) | merged | Optimize nemotron VL image/video preprocessing | `vllm/transformers_utils/processors/nano_nemotron_vl.py` |
| 2026-04-24 | [#40724](https://github.com/vllm-project/vllm/pull/40724) | merged | Fix Nano Nemotron VL static image inputs | `vllm/model_executor/models/nano_nemotron_vl.py` |

## 逐 PR diff 审计卡

### PR #6611 - [Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)

- 链接: https://github.com/vllm-project/vllm/pull/6611
- 状态/时间: merged / 2024-07-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`；关联提交 `07278c37ddd8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+776/-1，可读 patch 847 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`；技术摘要: 覆盖「[Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)」；主要实现面是 `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: _cast_if_autocast_enabled, NemotronLayerNorm1P, __init__, forward，涉及 `_cast_if_autocast_enabled, NemotronLayerNorm1P, __init__`；`vllm/transformers_utils/configs/nemotron.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: NemotronConfig, to, __init__, _rope_scaling_validation，涉及 `NemotronConfig, to, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: _cast_if_autocast_enabled, NemotronLayerNorm1P, __init__, forward
  - `vllm/transformers_utils/configs/nemotron.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: NemotronConfig, to, __init__, _rope_scaling_validation
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron.py
@@ -0,0 +1,531 @@
+# coding=utf-8
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -0,0 +1,209 @@
+# coding=utf-8
+# Copyright 2024 HuggingFace Inc. team. All rights reserved.
+# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron.py` added +531/-0; `vllm/transformers_utils/configs/nemotron.py` added +209/-0
- 验证与风险: runtime 路径改动集中在 `.buildkite/lm-eval-harness/configs/Minitron-4B-Base.yaml`, `.buildkite/lm-eval-harness/configs/models-small.txt`, `vllm/model_executor/layers/activation.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #7611 - [Model] Align nemotron config with final HF state and fix lm-eval-small

- 链接: https://github.com/vllm-project/vllm/pull/7611
- 状态/时间: merged / 2024-08-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`；关联提交 `44f26a946645`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+29/-35，可读 patch 181 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Align nemotron config with final HF state and fix lm-eval-small」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/nemotron.py`；技术摘要: 覆盖「[Model] Align nemotron config with final HF state and fix lm-eval-small」；主要实现面是 `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/nemotron.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/configs/nemotron.py` modified +18/-24 (42 lines); hunks: -35,20 +35,20 @@ class NemotronConfig(PretrainedConfig):; -63,38 +63,33 @@ class NemotronConfig(PretrainedConfig):; symbols: NemotronConfig, __init__，涉及 `NemotronConfig, __init__`；`vllm/model_executor/models/nemotron.py` modified +3/-3 (6 lines); hunks: -53,7 +53,7; -161,7 +161,7 @@ def __init__(; symbols: _cast_if_autocast_enabled, __init__，涉及 `_cast_if_autocast_enabled, __init__`。
- 代码 diff 细节:
  - `vllm/transformers_utils/configs/nemotron.py` modified +18/-24 (42 lines); hunks: -35,20 +35,20 @@ class NemotronConfig(PretrainedConfig):; -63,38 +63,33 @@ class NemotronConfig(PretrainedConfig):; symbols: NemotronConfig, __init__
  - `vllm/model_executor/models/nemotron.py` modified +3/-3 (6 lines); hunks: -53,7 +53,7; -161,7 +161,7 @@ def __init__(; symbols: _cast_if_autocast_enabled, __init__
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -35,20 +35,20 @@ class NemotronConfig(PretrainedConfig):
-        vocab_size (`int`, *optional*, defaults to 32000):
+        vocab_size (`int`, *optional*, defaults to 256000):
-        hidden_size (`int`, *optional*, defaults to 4096):
+        hidden_size (`int`, *optional*, defaults to 6144):
-        intermediate_size (`int`, *optional*, defaults to 11008):
+        intermediate_size (`int`, *optional*, defaults to 24576):
diff -- vllm/model_executor/models/nemotron.py
@@ -53,7 +53,7 @@
-# - Adds a rotary_percent to RoPE
+# - Adds a partial_rotary_factor to RoPE
@@ -161,7 +161,7 @@ def __init__(
-        self.rotary_percent = config.rope_percent
+        self.partial_rotary_factor = config.partial_rotary_factor
@@ -187,7 +187,7 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/configs/nemotron.py` modified +18/-24; `vllm/model_executor/models/nemotron.py` modified +3/-3
- 验证与风险: runtime 路径改动集中在 `.buildkite/lm-eval-harness/configs/Minitron-4B-Base-FP8.yaml`, `.buildkite/lm-eval-harness/configs/models-small.txt`, `vllm/model_executor/layers/rotary_embedding.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #15008 - [Model] Update support for NemotronNAS models

- 链接: https://github.com/vllm-project/vllm/pull/15008
- 状态/时间: merged / 2025-03-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_nas.py`；关联提交 `3aa2b6a63714`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+524/-133，可读 patch 764 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Update support for NemotronNAS models」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/nemotron_nas.py`；技术摘要: 覆盖「[Model] Update support for NemotronNAS models」；主要实现面是 `vllm/model_executor/models/nemotron_nas.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_nas.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__，涉及 `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_nas.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_nas.py
@@ -0,0 +1,454 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from
+# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
+# Copyright 2023 The vLLM team.
+# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
+#
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` added +454/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #18427 - [Model] Add support for YARN in NemotronNAS models

- 链接: https://github.com/vllm-project/vllm/pull/18427
- 状态/时间: merged / 2025-05-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_nas.py`；关联提交 `6d68030f1cac`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+67/-16，可读 patch 129 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for YARN in NemotronNAS models」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_nas.py`；技术摘要: 覆盖「[Model] Add support for YARN in NemotronNAS models」；主要实现面是 `vllm/model_executor/models/nemotron_nas.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_nas.py` modified +46/-2 (48 lines); hunks: -23,18 +23,20; -62,6 +64,48 @@ def _find_multiple(n: int, k: int) -> int:; symbols: _find_multiple, DeciLMAttention, __init__, _init_rotary_emb，涉及 `_find_multiple, DeciLMAttention, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_nas.py` modified +46/-2 (48 lines); hunks: -23,18 +23,20; -62,6 +64,48 @@ def _find_multiple(n: int, k: int) -> int:; symbols: _find_multiple, DeciLMAttention, __init__, _init_rotary_emb
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_nas.py
@@ -23,18 +23,20 @@
-from typing import Optional, Union
+from typing import Any, Optional, Union
+from vllm.attention import AttentionType
+from vllm.model_executor.layers.rotary_embedding import get_rope
@@ -62,6 +64,48 @@ def _find_multiple(n: int, k: int) -> int:
+class DeciLMAttention(LlamaAttention):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` modified +46/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/nemotron_nas.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #18863 - [Model] NemotronH support

- 链接: https://github.com/vllm-project/vllm/pull/18863
- 状态/时间: merged / 2025-06-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；关联提交 `cb6d572e85a3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+829/-0，可读 patch 866 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] NemotronH support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；技术摘要: 覆盖「[Model] NemotronH support」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunks: -0,0 +1,565; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer，涉及 `NemotronHMLP, __init__, forward`；`vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: NemotronHConfig, to, __init__, layers_block_type，涉及 `NemotronHConfig, to, __init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunks: -0,0 +1,565; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: NemotronHConfig, to, __init__, layers_block_type
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -0,0 +1,565 @@
+# SPDX-License-Identifier: Apache-2.0
+# Adapted from https://github.com/vllm-project/vllm/blob/94d8ec8d2bcb4ec55e33022b313c7e978edf05e1/vllm/model_executor/models/bamba.py
+# Copyright 2024 HuggingFace Inc. team. All rights reserved.
+# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
diff -- vllm/transformers_utils/configs/nemotron_h.py
@@ -0,0 +1,258 @@
+# SPDX-License-Identifier: Apache-2.0
+# Copyright 2024 HuggingFace Inc. team. All rights reserved.
+# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# you may not use this file except in compliance with the License.
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` added +565/-0; `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #19249 - [Model] Optimize nemotron_h implementation

- 链接: https://github.com/vllm-project/vllm/pull/19249
- 状态/时间: merged / 2025-06-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `7661e92ef85e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+16/-8，可读 patch 88 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Optimize nemotron_h implementation」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/nemotron_h.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +16/-8 (24 lines); hunks: -1,4 +1,5; -29,7 +30,7; symbols: __init__, NemotronHForCausalLM，涉及 `__init__, NemotronHForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +16/-8 (24 lines); hunks: -1,4 +1,5; -29,7 +30,7; symbols: __init__, NemotronHForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -1,4 +1,5 @@
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
@@ -29,7 +30,7 @@
-from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
+from vllm.model_executor.layers.linear import (ColumnParallelLinear,
@@ -63,19 +64,22 @@ def __init__(
+        prefix: str = "",
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +16/-8
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #20349 - [VLM] Add Nemotron-Nano-VL-8B-V1 support

- 链接: https://github.com/vllm-project/vllm/pull/20349
- 状态/时间: merged / 2025-07-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`；关联提交 `4ef00b5caca4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 11 个文件，+701/-3，可读 patch 837 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[VLM] Add Nemotron-Nano-VL-8B-V1 support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`；技术摘要: 覆盖「[VLM] Add Nemotron-Nano-VL-8B-V1 support」；主要实现面是 `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_vl.py` added +505/-0 (505 lines); hunks: -0,0 +1,505; symbols: NemotronVLProcessor, __init__, image_token_id, _preprocess_image，涉及 `NemotronVLProcessor, __init__, image_token_id`；`tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: _get_expected_num_patches, _run_check, test_processor_override，涉及 `_get_expected_num_patches, _run_check, test_processor_override`；`vllm/transformers_utils/configs/nemotron.py` modified +1/-1 (2 lines); hunks: -202,4 +202,4 @@ def _rope_scaling_validation(self):; symbols: _rope_scaling_validation，涉及 `_rope_scaling_validation`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_vl.py` added +505/-0 (505 lines); hunks: -0,0 +1,505; symbols: NemotronVLProcessor, __init__, image_token_id, _preprocess_image
  - `tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: _get_expected_num_patches, _run_check, test_processor_override
  - `vllm/transformers_utils/configs/nemotron.py` modified +1/-1 (2 lines); hunks: -202,4 +202,4 @@ def _rope_scaling_validation(self):; symbols: _rope_scaling_validation
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -0,0 +1,505 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
+# --------------------------------------------------------
+# InternVL
+# Copyright (c) 2023 OpenGVLab
diff -- tests/models/multimodal/processing/test_nemotron_vl.py
@@ -0,0 +1,134 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Tests for Nemotron-Nano-VL's multimodal preprocessing kwargs."""
+from collections.abc import Mapping
+from typing import Optional
+import pytest
diff -- vllm/transformers_utils/configs/nemotron.py
@@ -202,4 +202,4 @@ def _rope_scaling_validation(self):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` added +505/-0; `vllm/transformers_utils/configs/nemotron.py` modified +1/-1
  - tests: `tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22349 - [Model] NemotronH Support

- 链接: https://github.com/vllm-project/vllm/pull/22349
- 状态/时间: merged / 2025-08-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；关联提交 `14a5d903ab82`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+23/-7，可读 patch 91 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] NemotronH Support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；技术摘要: 覆盖「[Model] NemotronH Support」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +21/-5 (26 lines); hunks: -64,20 +64,32 @@ class NemotronHMLP(nn.Module):; -110,6 +122,7 @@ def __init__(; symbols: NemotronHMLP, __init__，涉及 `NemotronHMLP, __init__`；`vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2 (4 lines); hunks: -151,7 +151,7 @@ def __init__(; -194,7 +194,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +21/-5 (26 lines); hunks: -64,20 +64,32 @@ class NemotronHMLP(nn.Module):; -110,6 +122,7 @@ def __init__(; symbols: NemotronHMLP, __init__
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2 (4 lines); hunks: -151,7 +151,7 @@ def __init__(; -194,7 +194,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -64,20 +64,32 @@ class NemotronHMLP(nn.Module):
+        layer_idx: int,
+        hybrid_override_pattern = config.hybrid_override_pattern
+        mlp_index = hybrid_override_pattern[:layer_idx + 1].count("-") - 1
+        if isinstance(config.intermediate_size, list):
+            if len(config.intermediate_size) == 1:
+                intermediate_size = config.intermediate_size[0]
diff -- vllm/transformers_utils/configs/nemotron_h.py
@@ -151,7 +151,7 @@ def __init__(
-        attention_head_dim=128,
+        head_dim=128,
@@ -194,7 +194,7 @@ def __init__(
-        self.attention_head_dim = attention_head_dim
+        self.head_dim = head_dim
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +21/-5; `vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #22739 - [Bugfix] Fix Nemotron VL image processing

- 链接: https://github.com/vllm-project/vllm/pull/22739
- 状态/时间: merged / 2025-08-13
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`；关联提交 `a01e0018b50f`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+190/-4，可读 patch 234 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Nemotron VL image processing」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Nemotron VL image processing」；主要实现面是 `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_vl.py` modified +186/-0 (186 lines); hunks: -13,6 +13,7; -27,6 +28,7; symbols: build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets, dynamic_preprocess_nemotron_vl，涉及 `build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets`；`tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -23,15 +23,15 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches，涉及 `_get_expected_num_patches`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_vl.py` modified +186/-0 (186 lines); hunks: -13,6 +13,7; -27,6 +28,7; symbols: build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets, dynamic_preprocess_nemotron_vl
  - `tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -23,15 +23,15 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -13,6 +13,7 @@
+import torchvision.transforms as T
@@ -27,6 +28,7 @@
+from vllm.multimodal.image import convert_image_mode
@@ -44,6 +46,146 @@
+def build_transform(input_size: int):
+    return T.Compose([
diff -- tests/models/multimodal/processing/test_nemotron_vl.py
@@ -23,15 +23,15 @@ def _get_expected_num_patches(
-    from vllm.model_executor.models.internvl import (
-        calculate_internvl_targets, get_internvl_target_ratios)
+    from vllm.model_executor.models.nemotron_vl import (
+        calculate_nemotron_vl_targets, get_nemotron_vl_target_ratios)
-    blocks, _, _ = calculate_internvl_targets(
+    blocks, _, _ = calculate_nemotron_vl_targets(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +186/-0
  - tests: `tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/models/multimodal/processing/test_nemotron_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22023 - Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema

- 链接: https://github.com/vllm-project/vllm/pull/22023
- 状态/时间: merged / 2025-08-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_vl.py`；关联提交 `e75f34226161`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-23，可读 patch 43 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nemotron_vl.py`；技术摘要: 覆盖「Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema」；主要实现面是 `vllm/model_executor/models/nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_vl.py` modified +5/-23 (28 lines); hunks: -458,27 +458,6 @@ def extract_feature(self, pixel_values: torch.Tensor) -> to...; -516,9 +495,12 @@ def _parse_and_validate_image_input(; symbols: extract_feature, _validate_pixel_values, _validate_shape, _parse_and_validate_image_input，涉及 `extract_feature, _validate_pixel_values, _validate_shape`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_vl.py` modified +5/-23 (28 lines); hunks: -458,27 +458,6 @@ def extract_feature(self, pixel_values: torch.Tensor) -> to...; -516,9 +495,12 @@ def _parse_and_validate_image_input(; symbols: extract_feature, _validate_pixel_values, _validate_shape, _parse_and_validate_image_input
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -458,27 +458,6 @@ def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
-    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
-        #use force_image_size to get image_size
-        h = w = self.config.force_image_size
-        expected_dims = (3, h, w)
-        def _validate_shape(d: torch.Tensor):
-            actual_dims = tuple(d.shape)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +5/-23
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #23644 - Support for NemotronH Nano VLM

- 链接: https://github.com/vllm-project/vllm/pull/23644
- 状态/时间: merged / 2025-09-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `72d30108a0fe`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+1400/-1，可读 patch 1423 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support for NemotronH Nano VLM」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Support for NemotronH Nano VLM」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0 (1395 lines); hunks: -0,0 +1,1395; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs，涉及 `NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0 (1395 lines); hunks: -0,0 +1,1395; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -0,0 +1,1395 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# --------------------------------------------------------
+# Adapted from
+# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/internvl.py
+# under Apache-2.0 License
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #25708 - [Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2

- 链接: https://github.com/vllm-project/vllm/pull/25708
- 状态/时间: merged / 2025-09-25
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `57329a8c013c`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+6/-6，可读 patch 47 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「[Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -869,8 +869,8 @@ def get_dummy_mm_data(; -1249,7 +1249,7 @@ def print_architecture(self,; symbols: get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2, get_placeholder_str，涉及 `get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -869,8 +869,8 @@ def get_dummy_mm_data(; -1249,7 +1249,7 @@ def print_architecture(self,; symbols: get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2, get_placeholder_str
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -869,8 +869,8 @@ def get_dummy_mm_data(
-class NemotronH_Nano_VL(nn.Module, HasInnerState, IsHybrid,
-                        SupportsMultiModal):
+class NemotronH_Nano_VL_V2(nn.Module, HasInnerState, IsHybrid,
+                           SupportsMultiModal):
@@ -1249,7 +1249,7 @@ def print_architecture(self,
-            print("NemotronH_Nano_VL Model Architecture")
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #22980 - EVS Support (Video tokens pruning)

- 链接: https://github.com/vllm-project/vllm/pull/22980
- 状态/时间: merged / 2025-09-26
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+783/-39，可读 patch 1076 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「EVS Support (Video tokens pruning)」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/multimodal/evs.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`；技术摘要: 覆盖「EVS Support (Video tokens pruning)」；主要实现面是 `vllm/multimodal/evs.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/multimodal/evs.py` added +273/-0 (273 lines); hunks: -0,0 +1,273; symbols: compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media, recompute_mrope_positions，涉及 `compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media`；`vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12 (238 lines); hunks: -25,9 +25,9; -58,15 +58,22; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs，涉及 `Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs`；`tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos，涉及 `qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos`；`vllm/model_executor/models/interfaces.py` modified +55/-0 (55 lines); hunks: -115,6 +115,42 @@ def get_input_embeddings(; -142,6 +178,25 @@ def supports_multimodal_encoder_tp_data(; symbols: get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions, supports_multimodal，涉及 `get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions`。
- 代码 diff 细节:
  - `vllm/multimodal/evs.py` added +273/-0 (273 lines); hunks: -0,0 +1,273; symbols: compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media, recompute_mrope_positions
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12 (238 lines); hunks: -25,9 +25,9; -58,15 +58,22; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos
  - `vllm/model_executor/models/interfaces.py` modified +55/-0 (55 lines); hunks: -115,6 +115,42 @@ def get_input_embeddings(; -142,6 +178,25 @@ def supports_multimodal_encoder_tp_data(; symbols: get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions, supports_multimodal
  - `vllm/config/multimodal.py` modified +9/-0 (9 lines); hunks: -78,6 +78,11 @@ class MultiModalConfig:; -118,3 +123,7 @@ def merge_mm_processor_kwargs(; symbols: MultiModalConfig, compute_hash, merge_mm_processor_kwargs, is_multimodal_pruning_enabled
- 关键代码摘录:

```diff
diff -- vllm/multimodal/evs.py
@@ -0,0 +1,273 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
+#
+# NVIDIA CORPORATION and its licensors retain all intellectual property
+# and proprietary rights in and to this software, related documentation
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -25,9 +25,9 @@
-from collections.abc import Iterable, Mapping
+from collections.abc import Iterable, Mapping, Sequence
-from typing import Annotated, Callable, Literal, Optional, Union
+from typing import Annotated, Any, Callable, Literal, Optional, Union
@@ -58,15 +58,22 @@
-from vllm.multimodal.inputs import MultiModalFieldConfig
diff -- tests/models/multimodal/generation/test_qwen2_5_vl.py
@@ -0,0 +1,132 @@
```

- 已读文件:
  - runtime: `vllm/multimodal/evs.py` added +273/-0; `vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12; `vllm/model_executor/models/interfaces.py` modified +55/-0; `vllm/config/multimodal.py` modified +9/-0; `vllm/v1/worker/gpu_model_runner.py` modified +67/-16; `vllm/config/model.py` modified +16/-11
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_qwen2_5_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #26186 - Fix issue of using only the part of video frame [Nemotron Nano]

- 链接: https://github.com/vllm-project/vllm/pull/26186
- 状态/时间: merged / 2025-10-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `5a05f2660370`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-1，可读 patch 9 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix issue of using only the part of video frame [Nemotron Nano]」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Fix issue of using only the part of video frame [Nemotron Nano]」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def video_to_pixel_values(; symbols: video_to_pixel_values，涉及 `video_to_pixel_values`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def video_to_pixel_values(; symbols: video_to_pixel_values
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -208,7 +208,7 @@ def video_to_pixel_values(
-        frames_tensors.append(pil_frame[0])
+        frames_tensors.append(pil_frame[-1])
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #26269 - [Model] EVS support for nano_nemotron_vl

- 链接: https://github.com/vllm-project/vllm/pull/26269
- 状态/时间: merged / 2025-10-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `b8f603cebe39`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+224/-31，可读 patch 447 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] EVS support for nano_nemotron_vl」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「[Model] EVS support for nano_nemotron_vl」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19 (226 lines); hunks: -30,6 +30,7; -44,6 +45,10; symbols: __init__, supports_video, _preprocess_video, get_image_repl，涉及 `__init__, supports_video, _preprocess_video`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19 (226 lines); hunks: -30,6 +30,7; -44,6 +45,10; symbols: __init__, supports_video, _preprocess_video, get_image_repl
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -30,6 +30,7 @@
+    SupportsMultiModalPruning,
@@ -44,6 +45,10 @@
+from vllm.multimodal.evs import (
+    compute_retained_tokens_count,
+    compute_retention_mask,
+)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/multimodal/evs.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27107 - Nemotron Nano V2 VL + EVS Video Support

- 链接: https://github.com/vllm-project/vllm/pull/27107
- 状态/时间: merged / 2025-10-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `e93ff6c8b92b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+316/-105，可读 patch 771 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron Nano V2 VL + EVS Video Support」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Nemotron Nano V2 VL + EVS Video Support」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49 (315 lines); hunks: -14,14 +14,15; -53,12 +54,14; symbols: NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values, input_conditioner，涉及 `NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49 (315 lines); hunks: -14,14 +14,15; -53,12 +54,14; symbols: NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values, input_conditioner
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -14,14 +14,15 @@
+import regex as re
-from vllm.config.multimodal import BaseDummyOptions
+from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
@@ -53,12 +54,14 @@
+    VideoItem,
+    MultiModalDataParser,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`, `vllm/multimodal/profiling.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #25863 - [Model] Add MoE support for NemotronH

- 链接: https://github.com/vllm-project/vllm/pull/25863
- 状态/时间: merged / 2025-10-23
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；关联提交 `61089465a610`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+413/-39，可读 patch 765 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add MoE support for NemotronH」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`；技术摘要: 覆盖「[Model] Add MoE support for NemotronH」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunks: -18,21 +18,27; -54,16 +60,19; symbols: NemotronHMLP, __init__, forward, NemotronHMoE，涉及 `NemotronHMLP, __init__, forward`；`vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunks: -185,6 +185,15 @@ def __init__(; -241,6 +250,15 @@ def __init__(; symbols: __init__, layers_block_type，涉及 `__init__, layers_block_type`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunks: -18,21 +18,27; -54,16 +60,19; symbols: NemotronHMLP, __init__, forward, NemotronHMoE
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunks: -185,6 +185,15 @@ def __init__(; -241,6 +250,15 @@ def __init__(; symbols: __init__, layers_block_type
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -18,21 +18,27 @@
-from collections.abc import Iterable
+import typing
+from collections.abc import Callable, Iterable
-from vllm.distributed import get_tensor_model_parallel_world_size
+from vllm.config.parallel import ParallelConfig
+from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
diff -- vllm/transformers_utils/configs/nemotron_h.py
@@ -185,6 +185,15 @@ def __init__(
+        n_routed_experts=8,
+        n_shared_experts=1,
+        moe_intermediate_size=7688,
+        moe_shared_expert_intermediate_size=7688,
+        num_experts_per_tok=2,
+        routed_scaling_factor=1.0,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +329/-27; `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #27968 - [Model][Bugfix] fix pipeline parallelism support for NemotronH

- 链接: https://github.com/vllm-project/vllm/pull/27968
- 状态/时间: merged / 2025-11-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `77f8001f5330`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+13/-5，可读 patch 67 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Bugfix] fix pipeline parallelism support for NemotronH」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[Model][Bugfix] fix pipeline parallelism support for NemotronH」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +13/-5 (18 lines); hunks: -20,6 +20,7; -549,7 +550,7 @@ def get_layer(prefix: str):; symbols: get_layer, forward, load_weights，涉及 `get_layer, forward, load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +13/-5 (18 lines); hunks: -20,6 +20,7; -549,7 +550,7 @@ def get_layer(prefix: str):; symbols: get_layer, forward, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -20,6 +20,7 @@
+from itertools import islice
@@ -549,7 +550,7 @@ def get_layer(prefix: str):
-        self.make_empty_intmd_tensors = make_empty_intermediate_tensors_factory(
+        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
@@ -564,7 +565,7 @@ def forward(
-    ) -> torch.Tensor:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +13/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30795 - Fix nemotron_nas intermediate_size computation

- 链接: https://github.com/vllm-project/vllm/pull/30795
- 状态/时间: merged / 2025-12-17
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_nas.py`；关联提交 `f5db6385a19b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-4，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix nemotron_nas intermediate_size computation」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_nas.py`；技术摘要: 覆盖「Fix nemotron_nas intermediate_size computation」；主要实现面是 `vllm/model_executor/models/nemotron_nas.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_nas.py` modified +7/-4 (11 lines); hunks: -169,10 +169,13 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_nas.py` modified +7/-4 (11 lines); hunks: -169,10 +169,13 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_nas.py
@@ -169,10 +169,13 @@ def __init__(
-            ffn_mult = block_config.ffn.ffn_mult
-            intermediate_size = _ffn_mult_to_intermediate_size(
-                ffn_mult, config.hidden_size
-            )
+            if hasattr(block_config.ffn, "ffn_mult"):
+                ffn_mult = block_config.ffn.ffn_mult
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` modified +7/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_nas.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31539 - Add get_expert_mapping to NemotronHModel (for LoRA support)

- 链接: https://github.com/vllm-project/vllm/pull/31539
- 状态/时间: merged / 2025-12-31
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `108a2728f74d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-10，可读 patch 38 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add get_expert_mapping to NemotronHModel (for LoRA support)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Add get_expert_mapping to NemotronHModel (for LoRA support)」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +14/-10 (24 lines); hunks: -632,14 +632,7 @@ def forward(; -653,8 +646,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights, get_expert_mapping，涉及 `forward, load_weights, get_expert_mapping`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +14/-10 (24 lines); hunks: -632,14 +632,7 @@ def forward(; -653,8 +646,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights, get_expert_mapping
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -632,14 +632,7 @@ def forward(
-    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        stacked_params_mapping = [
-            # (param_name, shard_name, shard_id)
-            ("qkv_proj", "q_proj", "q"),
-            ("qkv_proj", "k_proj", "k"),
-            ("qkv_proj", "v_proj", "v"),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +14/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30864 - [Model] Nemotron Parse 1.1 Support

- 链接: https://github.com/vllm-project/vllm/pull/30864
- 状态/时间: merged / 2026-01-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/nemotron_parse.py`；关联提交 `ee212918250a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 13 个文件，+1117/-31，可读 patch 1329 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Nemotron Parse 1.1 Support」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_parse.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「[Model] Nemotron Parse 1.1 Support」；主要实现面是 `vllm/model_executor/models/nemotron_parse.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_parse.py` added +958/-0 (958 lines); hunks: -0,0 +1,958; symbols: BartScaledWordEmbedding, __init__, forward, BartParallelLMHead，涉及 `BartScaledWordEmbedding, __init__, forward`；`tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: run_test, test_models，涉及 `run_test, test_models`；`vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7 (9 lines); hunks: -1220,7 +1220,7 @@ def extract_feature(self, pixel_values):; -1695,12 +1695,7 @@ def get_vit_model_from_radio_config(self, hf_config):; symbols: extract_feature, get_vit_model_from_radio_config，涉及 `extract_feature, get_vit_model_from_radio_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_parse.py` added +958/-0 (958 lines); hunks: -0,0 +1,958; symbols: BartScaledWordEmbedding, __init__, forward, BartParallelLMHead
  - `tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: run_test, test_models
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7 (9 lines); hunks: -1220,7 +1220,7 @@ def extract_feature(self, pixel_values):; -1695,12 +1695,7 @@ def get_vit_model_from_radio_config(self, hf_config):; symbols: extract_feature, get_vit_model_from_radio_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_parse.py
@@ -0,0 +1,958 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+#
+# Adapted from https://github.com/amalad/vllm/blob/nemotron_parse/vllm/model_executor/models/nemotron_parse.py
+# that's based on https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/hf_nemotron_parse_modeling.py
+#
diff -- tests/models/multimodal/generation/test_nemotron_parse.py
@@ -0,0 +1,89 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from collections.abc import Sequence
+import pytest
+from transformers import AutoModel
+from tests.models.utils import check_logprobs_close
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1220,7 +1220,7 @@ def extract_feature(self, pixel_values):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_parse.py` added +958/-0; `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7
  - tests: `tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0
- 验证与风险: diff 自带测试面 `tests/conftest.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `tests/models/multimodal/pooling/test_radio.py`, `tests/models/multimodal/processing/test_common.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #31807 - [NemotronH] Use ReplicatedLinear for fc1_latent_proj

- 链接: https://github.com/vllm-project/vllm/pull/31807
- 状态/时间: merged / 2026-01-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `28c94770adfc`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+1/-5，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Use ReplicatedLinear for fc1_latent_proj」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +1/-5 (6 lines); hunks: -210,16 +210,12 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-5 (6 lines); hunks: -210,16 +210,12 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -210,16 +210,12 @@ def __init__(
-            # TODO: check if using ReplicatedLinear is better than
-            # ColumnParallelLinear + all_gather
-            self.fc1_latent_proj = ColumnParallelLinear(
+            self.fc1_latent_proj = ReplicatedLinear(
-                # We need to gather the output to prepare input for moe
-                gather_output=True,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-5
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #31898 - Enable quantized attention in NemotronH models

- 链接: https://github.com/vllm-project/vllm/pull/31898
- 状态/时间: merged / 2026-01-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `bf184a66218b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+4/-0，可读 patch 18 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable quantized attention in NemotronH models」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Enable quantized attention in NemotronH models」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -483,6 +483,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -483,6 +483,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -483,6 +483,7 @@ def __init__(
+            quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #30802 - Add support for LoRA adapters in Nemotron-H models

- 链接: https://github.com/vllm-project/vllm/pull/30802
- 状态/时间: merged / 2026-01-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `aa7f37ccfa16`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 10 个文件，+497/-27，可读 patch 717 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Add support for LoRA adapters in Nemotron-H models」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Add support for LoRA adapters in Nemotron-H models」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -747,6 +747,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM，涉及 `NemotronHForCausalLM`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -747,6 +747,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -747,6 +747,9 @@ class NemotronHForCausalLM(
+    # Relevant only if self.has_moe is True
+    is_non_gated_moe: bool = True
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-0
- 验证与风险: diff 自带测试面 `tests/lora/test_layers.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #32121 - support dynamic resolution image encoding for Nemotron Nano VL

- 链接: https://github.com/vllm-project/vllm/pull/32121
- 状态/时间: merged / 2026-01-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `cd3ac5b79703`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+755/-164，可读 patch 1299 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「support dynamic resolution image encoding for Nemotron Nano VL」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「support dynamic resolution image encoding for Nemotron Nano VL」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141 (685 lines); hunks: -8,11 +8,15; -23,6 +27,7; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs, calculate_timestamps，涉及 `NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141 (685 lines); hunks: -8,11 +8,15; -23,6 +27,7; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs, calculate_timestamps
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -8,11 +8,15 @@
+import math
+from dataclasses import dataclass
+from functools import cached_property
+import einops
@@ -23,6 +27,7 @@
+from vllm.logger import init_logger
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32682 - [Bugfix] Fix Nemotron-Nano-v2-vlm static resolution

- 链接: https://github.com/vllm-project/vllm/pull/32682
- 状态/时间: merged / 2026-01-21
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `27ca95b3c9e6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Nemotron-Nano-v2-vlm static resolution」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「[Bugfix] Fix Nemotron-Nano-v2-vlm static resolution」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1678,7 +1678,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic，涉及 `_parse_and_validate_image_input, _process_image_input_dynamic`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1678,7 +1678,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1678,7 +1678,9 @@ def _parse_and_validate_image_input(
-            return NanoNemotronVLImagePixelInputs(**kwargs)
+            return NanoNemotronVLImagePixelInputs(
+                num_patches=kwargs.pop("image_num_patches"), **kwargs
+            )
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32265 - [LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models

- 链接: https://github.com/vllm-project/vllm/pull/32265
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `f3a5ee705fa9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+31/-0，可读 patch 106 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -771,6 +771,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM, get_mamba_state_dtype_from_config，涉及 `NemotronHForCausalLM, get_mamba_state_dtype_from_config`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -771,6 +771,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM, get_mamba_state_dtype_from_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -771,6 +771,9 @@ class NemotronHForCausalLM(
+    # Skip MTP (Multi-Token Prediction) layers during LoRA loading
+    lora_skip_prefixes = ["mtp."]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-0
- 验证与风险: runtime 路径改动集中在 `vllm/lora/lora_model.py`, `vllm/lora/worker_manager.py`, `vllm/model_executor/models/interfaces.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32549 - Support heterogeneous NemotronHPuzzle model

- 链接: https://github.com/vllm-project/vllm/pull/32549
- 状态/时间: merged / 2026-01-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `83fb2d09e8f6`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+75/-5，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support heterogeneous NemotronHPuzzle model」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Support heterogeneous NemotronHPuzzle model」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +43/-3 (46 lines); hunks: -354,8 +354,12 @@ def __init__(; -479,6 +483,9 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +43/-3 (46 lines); hunks: -354,8 +354,12 @@ def __init__(; -479,6 +483,9 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -354,8 +354,12 @@ def __init__(
+        # Get per-layer config for heterogeneous models if exsist
+        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
+        layer_config = get_layer_config(layer_idx) if get_layer_config else config
-            config,
+            layer_config,
@@ -479,6 +483,9 @@ def __init__(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +43/-3
- 验证与风险: diff 自带测试面 `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33189 - [Misc][Build] Lazy load cv2 in nemotron_parse.py

- 链接: https://github.com/vllm-project/vllm/pull/33189
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_parse.py`；关联提交 `9e138cb01d65`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 26 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Misc][Build] Lazy load cv2 in nemotron_parse.py」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_parse.py`；技术摘要: 覆盖「[Misc][Build] Lazy load cv2 in nemotron_parse.py」；主要实现面是 `vllm/model_executor/models/nemotron_parse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_parse.py` modified +4/-1 (5 lines); hunks: -11,7 +11,6; -416,6 +415,8 @@ def _create_transforms(self):; symbols: _create_transforms, _resize_with_aspect_ratio，涉及 `_create_transforms, _resize_with_aspect_ratio`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_parse.py` modified +4/-1 (5 lines); hunks: -11,7 +11,6; -416,6 +415,8 @@ def _create_transforms(self):; symbols: _create_transforms, _resize_with_aspect_ratio
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_parse.py
@@ -11,7 +11,6 @@
-import cv2
@@ -416,6 +415,8 @@ def _create_transforms(self):
+        import cv2
@@ -457,6 +458,8 @@ def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
+        import cv2
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_parse.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_parse.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32669 - Bugfix: Pass router logits dtype in nemotron shared experts

- 链接: https://github.com/vllm-project/vllm/pull/32669
- 状态/时间: merged / 2026-01-29
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `e01ff5c070f4`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 22 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Bugfix: Pass router logits dtype in nemotron shared experts」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Bugfix: Pass router logits dtype in nemotron shared experts」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +3/-1 (4 lines); hunks: -145,11 +145,12 @@ def __init__(; -209,6 +210,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-1 (4 lines); hunks: -145,11 +145,12 @@ def __init__(; -209,6 +210,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -145,11 +145,12 @@ def __init__(
+        router_logits_dtype = torch.float32
-            params_dtype=torch.float32,
+            params_dtype=router_logits_dtype,
@@ -209,6 +210,7 @@ def __init__(
+            router_logits_dtype=router_logits_dtype,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #32790 - [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)

- 链接: https://github.com/vllm-project/vllm/pull/32790
- 状态/时间: merged / 2026-02-02
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `0aca8b8c628e`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+303/-58，可读 patch 499 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +30/-42 (72 lines); hunks: -188,10 +188,29 @@ def __init__(; -211,30 +230,9 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +30/-42 (72 lines); hunks: -188,10 +188,29 @@ def __init__(; -211,30 +230,9 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -188,10 +188,29 @@ def __init__(
+        if self.use_latent_moe:
+            self.fc1_latent_proj = ReplicatedLinear(
+                input_size=config.hidden_size,
+                output_size=self.moe_hidden_size,
+                bias=config.mlp_bias,
+                quant_config=quant_config,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +30/-42
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #33506 - [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4

- 链接: https://github.com/vllm-project/vllm/pull/33506
- 状态/时间: merged / 2026-02-12
- 反查来源: 保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+197/-45，可读 patch 562 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`；技术摘要: 覆盖「[Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4」；主要实现面是 `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5 (93 lines); hunks: -6,6 +6,7; -18,14 +19,28 @@ class FlashinferMoeBackend(Enum):; symbols: FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31, rotate_weights_for_fi_trtllm_fp8_per_tensor_moe，涉及 `FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31`；`vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19 (70 lines); hunks: -15,6 +15,10; -50,8 +54,8 @@ def _supports_current_device() -> bool:; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation，涉及 `_supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme`；`vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6 (14 lines); hunks: -35,8 +35,8 @@ def _supports_current_device() -> bool:; -52,8 +52,7 @@ def _supports_quant_scheme(; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation，涉及 `_supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme`；`vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3 (7 lines); hunks: -937,10 +937,11 @@ def apply_monolithic(; symbols: apply_monolithic，涉及 `apply_monolithic`。
- 代码 diff 细节:
  - `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5 (93 lines); hunks: -6,6 +6,7; -18,14 +19,28 @@ class FlashinferMoeBackend(Enum):; symbols: FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31, rotate_weights_for_fi_trtllm_fp8_per_tensor_moe
  - `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19 (70 lines); hunks: -15,6 +15,10; -50,8 +54,8 @@ def _supports_current_device() -> bool:; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6 (14 lines); hunks: -35,8 +35,8 @@ def _supports_current_device() -> bool:; -52,8 +52,7 @@ def _supports_quant_scheme(; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3 (7 lines); hunks: -937,10 +937,11 @@ def apply_monolithic(; symbols: apply_monolithic
  - `tests/kernels/moe/test_flashinfer.py` modified +46/-12 (58 lines); hunks: -71,7 +71,8 @@ def quant_fp8_per_tensor_batches(a):; -81,6 +82,20 @@ def quant_fp8_per_tensor_batches(a):; symbols: quant_fp8_per_tensor_batches, check_accuracy, TestData, make_moe_tensors_8bit
- 关键代码摘录:

```diff
diff -- vllm/model_executor/layers/quantization/utils/flashinfer_utils.py
@@ -6,6 +6,7 @@
+from vllm.model_executor.layers.fused_moe.activation import MoEActivation
@@ -18,14 +19,28 @@ class FlashinferMoeBackend(Enum):
+def activation_to_flashinfer_int(activation: MoEActivation) -> int:
+    from flashinfer.fused_moe.core import ActivationType
+    # silu and gelu are mapped to their gated versions SwiGLU and GeGLU respectively
+    ACTIVATION_TO_FI_ACTIVATION = {
diff -- vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py
@@ -15,6 +15,10 @@
+from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
+    activation_to_flashinfer_int,
+    align_fp4_moe_weights_for_fi,
+)
@@ -50,8 +54,8 @@ def _supports_current_device() -> bool:
-    """Does not support non-gated MoE (i.e. Nemotron-Nano)."""
diff -- vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py
@@ -35,8 +35,8 @@ def _supports_current_device() -> bool:
```

- 已读文件:
  - runtime: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5; `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19; `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6; `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3
  - tests: `tests/kernels/moe/test_flashinfer.py` modified +46/-12
- 验证与风险: diff 自带测试面 `tests/kernels/moe/test_flashinfer.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34582 - [NemotronH] Do not force router to run in fp32

- 链接: https://github.com/vllm-project/vllm/pull/34582
- 状态/时间: merged / 2026-02-16
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `3b30e6150777`, `3eff45d793da`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+5/-4，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Do not force router to run in fp32」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[NemotronH] Do not force router to run in fp32」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +1/-4 (5 lines); hunks: -148,12 +148,10 @@ def __init__(; -232,7 +230,6 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-4 (5 lines); hunks: -148,12 +148,10 @@ def __init__(; -232,7 +230,6 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -148,12 +148,10 @@ def __init__(
-        router_logits_dtype = torch.float32
-            params_dtype=router_logits_dtype,
@@ -232,7 +230,6 @@ def __init__(
-            router_logits_dtype=router_logits_dtype,
@@ -244,7 +241,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-4
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #34725 - [Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4

- 链接: https://github.com/vllm-project/vllm/pull/34725
- 状态/时间: merged / 2026-02-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`；关联提交 `caeb887bf633`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+20/-0，可读 patch 33 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`；技术摘要: 覆盖「[Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4」；主要实现面是 `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8；`tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml
@@ -0,0 +1,8 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
+accuracy_threshold: 0.29
+num_questions: 1319
+num_fewshot: 5
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
+env:
diff -- tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml
@@ -0,0 +1,8 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
+accuracy_threshold: 0.29
+num_questions: 1319
+num_fewshot: 5
+server_args: "--enforce-eager --max-model-len 8192 --tensor-parallel-size 2"
+env:
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0; `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/config-b200.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #34808 - Revert "[NemotronH] Do not force router to run in fp32 (#34582)"

- 链接: https://github.com/vllm-project/vllm/pull/34808
- 状态/时间: merged / 2026-02-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `3eff45d793da`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+4/-1，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Revert "[NemotronH] Do not force router to run in fp32 (#34582)"」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Revert "[NemotronH] Do not force router to run in fp32 (#34582)"」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +4/-1 (5 lines); hunks: -148,10 +148,12 @@ def __init__(; -230,6 +232,7 @@ def __init__(; symbols: __init__, forward，涉及 `__init__, forward`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +4/-1 (5 lines); hunks: -148,10 +148,12 @@ def __init__(; -230,6 +232,7 @@ def __init__(; symbols: __init__, forward
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -148,10 +148,12 @@ def __init__(
+        router_logits_dtype = torch.float32
+            params_dtype=router_logits_dtype,
@@ -230,6 +232,7 @@ def __init__(
+            router_logits_dtype=router_logits_dtype,
@@ -241,7 +244,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
-        router_logits, _ = self.gate(hidden_states)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +4/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #33726 - [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support

- 链接: https://github.com/vllm-project/vllm/pull/33726
- 状态/时间: merged / 2026-02-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`；关联提交 `f5972a872fa3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 19 个文件，+800/-158，可读 patch 1473 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「[Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support」；主要实现面是 `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunks: -0,0 +1,503; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer，涉及 `NemotronHMTPAttentionDecoderLayer, __init__, forward`；`vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3 (9 lines); hunks: -51,6 +51,8 @@ class NemotronHConfig(PretrainedConfig):; -150,6 +152,7 @@ def __init__(; symbols: NemotronHConfig, __init__，涉及 `NemotronHConfig, __init__`；`vllm/model_executor/models/nemotron_h.py` modified +8/-0 (8 lines); hunks: -636,6 +636,9 @@ def forward(; -702,6 +705,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, is_spec_layer, _get_max_n_routed_experts, load_weights，涉及 `forward, is_spec_layer, _get_max_n_routed_experts`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunks: -0,0 +1,503; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3 (9 lines); hunks: -51,6 +51,8 @@ class NemotronHConfig(PretrainedConfig):; -150,6 +152,7 @@ def __init__(; symbols: NemotronHConfig, __init__
  - `vllm/model_executor/models/nemotron_h.py` modified +8/-0 (8 lines); hunks: -636,6 +636,9 @@ def forward(; -702,6 +705,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, is_spec_layer, _get_max_n_routed_experts, load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h_mtp.py
@@ -0,0 +1,503 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""NemotronH-MTP model with attention layers."""
+import typing
+from collections.abc import Callable, Iterable
+import torch
diff -- vllm/transformers_utils/configs/nemotron_h.py
@@ -51,6 +51,8 @@ class NemotronHConfig(PretrainedConfig):
+        mtp_hybrid_override_pattern (`str`, *optional*, defaults to `"*E"`):
+            The pattern of the MTP layers.
@@ -150,6 +152,7 @@ def __init__(
+        mtp_hybrid_override_pattern="*E",
@@ -203,6 +206,7 @@ def __init__(
+        self.mtp_hybrid_override_pattern = mtp_hybrid_override_pattern
diff -- vllm/model_executor/models/nemotron_h.py
@@ -636,6 +636,9 @@ def forward(
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0; `vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3; `vllm/model_executor/models/nemotron_h.py` modified +8/-0
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/v1/attention/test_mamba_update_block_table.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35297 - [Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model

- 链接: https://github.com/vllm-project/vllm/pull/35297
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/embed/template/nemotron_embed_vl.jinja`, `vllm/model_executor/models/nemotron_vl.py`；关联提交 `111d86906999`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 8 个文件，+545/-31，可读 patch 752 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/embed/template/nemotron_embed_vl.jinja`；技术摘要: 覆盖「[Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model」；主要实现面是 `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/embed/template/nemotron_embed_vl.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_vl.py` modified +271/-31 (302 lines); hunks: -18,6 +18,7; -30,24 +31,28; symbols: build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor, __init__，涉及 `build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor`；`examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0 (20 lines); hunks: -0,0 +1,20。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_vl.py` modified +271/-31 (302 lines); hunks: -18,6 +18,7; -30,24 +31,28; symbols: build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor, __init__
  - `examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0 (20 lines); hunks: -0,0 +1,20
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -18,6 +18,7 @@
+from vllm.model_executor.layers.pooler import DispatchPooler
@@ -30,24 +31,28 @@
+from vllm.model_executor.models.siglip import SiglipVisionModel
+from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
-from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix
-IMG_START = "<img>"
diff -- examples/pooling/embed/template/nemotron_embed_vl.jinja
@@ -0,0 +1,20 @@
+{%- if messages | length > 1 -%}
+    {{ raise_exception('Embedding models should only embed one message at a time') }}
+{%- endif -%}
+{% set vars = namespace(prefix='', images=[], texts=[]) %}
+{%- for message in messages -%}
+    {%- if message['role'] == 'query' -%}
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +271/-31
  - docs: `examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_llama_nemotron_vl_embed.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35396 - Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models

- 链接: https://github.com/vllm-project/vllm/pull/35396
- 状态/时间: merged / 2026-02-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h.py`；关联提交 `832a780f3aed`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+5/-0，可读 patch 12 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `vllm/model_executor/models/nemotron_h.py`；技术摘要: 覆盖「Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models」；主要实现面是 `vllm/model_executor/models/nemotron_h.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h.py` modified +5/-0 (5 lines); hunks: -298,6 +298,11 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h.py` modified +5/-0 (5 lines); hunks: -298,6 +298,11 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -298,6 +298,11 @@ def __init__(
+        # Get per-layer config for heterogeneous models if exist
+        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
+        layer_config = get_layer_config(layer_idx) if get_layer_config else config
+        config = layer_config
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +5/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_h.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35100 - Support parakeet as audio encoder for nemotron-nano-vl

- 链接: https://github.com/vllm-project/vllm/pull/35100
- 状态/时间: merged / 2026-02-27
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `c8aca0c9e1b3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+448/-20，可读 patch 678 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support parakeet as audio encoder for nemotron-nano-vl」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20 (274 lines); hunks: -44,6 +44,7; -55,12 +56,14; symbols: NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video, _preprocess_audio，涉及 `NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20 (274 lines); hunks: -44,6 +44,7; -55,12 +56,14; symbols: NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video, _preprocess_audio
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -44,6 +44,7 @@
+from vllm.model_executor.models.parakeet import ParakeetExtractor, ProjectedParakeet
@@ -55,12 +56,14 @@
+    AudioItem,
+    AudioProcessorItems,
@@ -91,9 +94,29 @@
+class NanoNemotronVLAudioFeatureInputs(TensorSchema):
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/parakeet.py`, `vllm/transformers_utils/configs/parakeet.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35735 - [Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2

- 链接: https://github.com/vllm-project/vllm/pull/35735
- 状态/时间: merged / 2026-03-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `examples/pooling/score/template/nemotron-vl-rerank.jinja`, `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`；关联提交 `c8b678e53e37`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 9 个文件，+503/-149，可读 patch 723 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/score/template/nemotron-vl-rerank.jinja`；技术摘要: 覆盖「[Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2」；主要实现面是 `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/score/template/nemotron-vl-rerank.jinja`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0 (355 lines); hunks: -0,0 +1,355; symbols: _run_test, test_models_text, test_models_image, _pil_to_data_uri，涉及 `_run_test, test_models_text, test_models_image`；`vllm/model_executor/models/nemotron_vl.py` modified +57/-0 (57 lines); hunks: -7,6 +7,7; -18,6 +19,7; symbols: load_weights, LlamaNemotronVLForSequenceClassification, __init__，涉及 `load_weights, LlamaNemotronVLForSequenceClassification, __init__`；`examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0 (15 lines); hunks: -0,0 +1,15。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0 (355 lines); hunks: -0,0 +1,355; symbols: _run_test, test_models_text, test_models_image, _pil_to_data_uri
  - `vllm/model_executor/models/nemotron_vl.py` modified +57/-0 (57 lines); hunks: -7,6 +7,7; -18,6 +19,7; symbols: load_weights, LlamaNemotronVLForSequenceClassification, __init__
  - `examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0 (15 lines); hunks: -0,0 +1,15
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_llama_nemotron_vl.py
@@ -0,0 +1,355 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Tests for the LlamaNemotronVL model family:
+  - nvidia/llama-nemotron-embed-vl-1b-v2  (LlamaNemotronVLForCausalLM / embed)
+  - nvidia/llama-nemotron-rerank-vl-1b-v2
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -7,6 +7,7 @@
+import math
@@ -18,6 +19,7 @@
+from vllm.model_executor.layers.linear import ReplicatedLinear
@@ -42,6 +44,7 @@
+    SupportsCrossEncoding,
@@ -883,3 +886,57 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
diff -- examples/pooling/score/template/nemotron-vl-rerank.jinja
@@ -0,0 +1,15 @@
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +57/-0
  - docs: `examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `tests/models/multimodal/pooling/test_llama_nemotron_vl_embed.py`, `tests/models/registry.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #35539 - Support Audio Extraction from MP4 Video for Nemotron Nano VL

- 链接: https://github.com/vllm-project/vllm/pull/35539
- 状态/时间: merged / 2026-03-04
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `5d199ac8f25a`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+225/-1，可读 patch 293 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support Audio Extraction from MP4 Video for Nemotron Nano VL」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Support Audio Extraction from MP4 Video for Nemotron Nano VL」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1 (130 lines); hunks: -59,18 +59,25; -1381,6 +1388,127 @@ class NanoNemotronVLMultiModalProcessor(; symbols: NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply, _get_mm_fields_config，涉及 `NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1 (130 lines); hunks: -59,18 +59,25; -1381,6 +1388,127 @@ class NanoNemotronVLMultiModalProcessor(; symbols: NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply, _get_mm_fields_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -59,18 +59,25 @@
+    MultiModalInputs,
+from vllm.multimodal.media.audio import extract_audio_from_video_bytes
+    VideoProcessorItems,
+)
+from vllm.multimodal.processing import (
+    BaseDummyInputsBuilder,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/config.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/multimodal/media/audio.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- 链接: https://github.com/vllm-project/vllm/pull/36192
- 状态/时间: merged / 2026-03-06
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_vl.py`；关联提交 `00bd08edeee5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+7/-2，可读 patch 30 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nemotron_vl.py`；技术摘要: 覆盖「[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25」；主要实现面是 `vllm/model_executor/models/nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_vl.py` modified +5/-1 (6 lines); hunks: -402,6 +402,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -456,7 +457,10 @@ def _init_vision_model(; symbols: __init__, _init_vision_model, _init_mlp1，涉及 `__init__, _init_vision_model, _init_mlp1`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_vl.py` modified +5/-1 (6 lines); hunks: -402,6 +402,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -456,7 +457,10 @@ def _init_vision_model(; symbols: __init__, _init_vision_model, _init_mlp1
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_vl.py
@@ -402,6 +402,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
+        self.model_config = vllm_config.model_config
@@ -456,7 +457,10 @@ def _init_vision_model(
-        return AutoModel.from_config(config.vision_config, trust_remote_code=True)
+        return AutoModel.from_config(
+            config.vision_config,
+            trust_remote_code=self.model_config.trust_remote_code,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +5/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #35657 - [Model] Nano Nemotron VL - fast media preprocessing

- 链接: https://github.com/vllm-project/vllm/pull/35657
- 状态/时间: merged / 2026-03-08
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `b7332b058c3b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+80/-61，可读 patch 217 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Nano Nemotron VL - fast media preprocessing」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「[Model] Nano Nemotron VL - fast media preprocessing」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61 (141 lines); hunks: -17,11 +17,11; -214,7 +214,12 @@ class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):; symbols: NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values, video_to_pixel_values，涉及 `NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61 (141 lines); hunks: -17,11 +17,11; -214,7 +214,12 @@ class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):; symbols: NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values, video_to_pixel_values
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -17,11 +17,11 @@
+import numpy as np
-import torchvision.transforms as T
@@ -214,7 +214,12 @@ class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):
-    image, *, image_size=512, max_num_tiles=12, use_thumbnail=True, idx=0
+    image,
+    *,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36393 - add nemotron v3 reasoning parser

- 链接: https://github.com/vllm-project/vllm/pull/36393
- 状态/时间: merged / 2026-03-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`；关联提交 `203a7f27dac2`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+186/-0，可读 patch 195 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「add nemotron v3 reasoning parser」；模型线: Nemotron Super；类别: 文档/测试/CI；主要 diff: `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`；技术摘要: 覆盖「add nemotron v3 reasoning parser」；主要实现面是 `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: ReasoningCase, FakeNemotronTokenizer, __init__, get_vocab，涉及 `ReasoningCase, FakeNemotronTokenizer, __init__`；`vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: NemotronV3ReasoningParser, extract_reasoning，涉及 `NemotronV3ReasoningParser, extract_reasoning`。
- 代码 diff 细节:
  - `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: ReasoningCase, FakeNemotronTokenizer, __init__, get_vocab
  - `vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: NemotronV3ReasoningParser, extract_reasoning
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_nemotron_v3_reasoning_parser.py
@@ -0,0 +1,150 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import TypedDict
+import pytest
+import regex as re
+from tests.reasoning.utils import run_reasoning_extraction
diff -- vllm/reasoning/nemotron_v3_reasoning_parser.py
@@ -0,0 +1,32 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+)
+from vllm.entrypoints.openai.responses.protocol import (
```

- 已读文件:
  - tests: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0
  - runtime: `vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0
- 验证与风险: diff 自带测试面 `tests/reasoning/test_nemotron_v3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #36635 - [NemotronH] Small fix reasoning parser

- 链接: https://github.com/vllm-project/vllm/pull/36635
- 状态/时间: merged / 2026-03-11
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`；关联提交 `e661b9ee83d9`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+26/-1，可读 patch 41 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[NemotronH] Small fix reasoning parser」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`；未提供可用技术摘要。
- 实现要点: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0 (22 lines); hunks: -128,6 +128,28 @@ def test_nemotron_v3_without_thinking_returns_content(; symbols: test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning，涉及 `test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning`；`vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1 (5 lines); hunks: -24,7 +24,10 @@ def extract_reasoning(; symbols: extract_reasoning，涉及 `extract_reasoning`。
- 代码 diff 细节:
  - `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0 (22 lines); hunks: -128,6 +128,28 @@ def test_nemotron_v3_without_thinking_returns_content(; symbols: test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning
  - `vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1 (5 lines); hunks: -24,7 +24,10 @@ def extract_reasoning(; symbols: extract_reasoning
- 关键代码摘录:

```diff
diff -- tests/reasoning/test_nemotron_v3_reasoning_parser.py
@@ -128,6 +128,28 @@ def test_nemotron_v3_without_thinking_returns_content(
+def test_nemotron_v3_force_nonempty_content_returns_content(
+    tokenizer: FakeNemotronTokenizer,
+):
+    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
+    parser = parser_cls(tokenizer)
+    request = ChatCompletionRequest(
diff -- vllm/reasoning/nemotron_v3_reasoning_parser.py
@@ -24,7 +24,10 @@ def extract_reasoning(
-            and chat_template_kwargs.get("enable_thinking") is False
+            and (
+                chat_template_kwargs.get("enable_thinking") is False
+                or chat_template_kwargs.get("force_nonempty_content") is True
+            )
```

- 已读文件:
  - tests: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0
  - runtime: `vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1
- 验证与风险: diff 自带测试面 `tests/reasoning/test_nemotron_v3_reasoning_parser.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37456 - [Model] Remove unnecessary processor definition for Nemotron Parse

- 链接: https://github.com/vllm-project/vllm/pull/37456
- 状态/时间: merged / 2026-03-18
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_parse.py`；关联提交 `7476d148db99`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 3 个文件，+0/-259，可读 patch 288 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Remove unnecessary processor definition for Nemotron Parse」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/processors/nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`；技术摘要: 覆盖「[Model] Remove unnecessary processor definition for Nemotron Parse」；主要实现面是 `vllm/transformers_utils/processors/nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: NemotronParseImageProcessor, __init__, _create_transforms, _resize_with_aspect_ratio，涉及 `NemotronParseImageProcessor, __init__, _create_transforms`；`vllm/model_executor/models/nemotron_parse.py` modified +0/-12 (12 lines); hunks: -55,7 +55,6; -367,17 +366,6 @@ class NemotronParsePixelInputs(TensorSchema):; symbols: NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config, get_hf_processor，涉及 `NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: NemotronParseImageProcessor, __init__, _create_transforms, _resize_with_aspect_ratio
  - `vllm/model_executor/models/nemotron_parse.py` modified +0/-12 (12 lines); hunks: -55,7 +55,6; -367,17 +366,6 @@ class NemotronParsePixelInputs(TensorSchema):; symbols: NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config, get_hf_processor
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/nemotron_parse.py
@@ -1,245 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-#
-# Adapted from https://github.com/amalad/vllm/blob/nemotron_parse/vllm/model_executor/models/nemotron_parse.py
-# that's based on https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/blob/main/hf_nemotron_parse_modeling.py
-from typing import TypeVar
diff -- vllm/model_executor/models/nemotron_parse.py
@@ -55,7 +55,6 @@
-from vllm.transformers_utils.processors.nemotron_parse import NemotronParseProcessor
@@ -367,17 +366,6 @@ class NemotronParsePixelInputs(TensorSchema):
-    def get_hf_config(self):
-        return self.ctx.get_hf_config()
-    def get_hf_processor(self, **kwargs) -> NemotronParseProcessor:
-        return self.ctx.init_processor(
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245; `vllm/model_executor/models/nemotron_parse.py` modified +0/-12
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nemotron_parse.py`, `vllm/transformers_utils/processors/__init__.py`, `vllm/transformers_utils/processors/nemotron_parse.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36808 - Support temporal compression for Nemotron-3-VL videos

- 链接: https://github.com/vllm-project/vllm/pull/36808
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；关联提交 `0b6d52629fe8`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 5 个文件，+553/-130，可读 patch 1189 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Support temporal compression for Nemotron-3-VL videos」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Support temporal compression for Nemotron-3-VL videos」；主要实现面是 `vllm/transformers_utils/processors/nano_nemotron_vl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27 (232 lines); hunks: -11,6 +11,7; -43,6 +44,12; symbols: calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size, get_video_target_size_and_feature_size，涉及 `calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size`；`vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35 (210 lines); hunks: -8,6 +8,7; -77,6 +78,7; symbols: get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates, get_video_replacement_internvl，涉及 `get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27 (232 lines); hunks: -11,6 +11,7; -43,6 +44,12; symbols: calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size, get_video_target_size_and_feature_size
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35 (210 lines); hunks: -8,6 +8,7; -77,6 +78,7; symbols: get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates, get_video_replacement_internvl
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/nano_nemotron_vl.py
@@ -11,6 +11,7 @@
+from functools import cached_property
@@ -43,6 +44,12 @@
+# Configure PIL to handle large images without warnings
+# This prevents DecompressionBombWarning for legitimate large images
+Image.MAX_IMAGE_PIXELS = None  # Disable the limit entirely
+# Alternative: Set a specific higher limit
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -8,6 +8,7 @@
+import math
@@ -77,6 +78,7 @@
+from vllm.transformers_utils.processors.internvl import get_internvl_target_ratios
@@ -85,7 +87,7 @@
-    get_internvl_target_ratios,
+    get_video_target_size_and_feature_size,
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27; `vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`, `vllm/transformers_utils/configs/radio.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37407 - [Bugfix] Fix Nemotron Parse loading

- 链接: https://github.com/vllm-project/vllm/pull/37407
- 状态/时间: merged / 2026-03-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`；关联提交 `765e4610651b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 4 个文件，+49/-19，可读 patch 138 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Bugfix] Fix Nemotron Parse loading」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`；技术摘要: 覆盖「[Bugfix] Fix Nemotron Parse loading」；主要实现面是 `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11 (55 lines); hunks: -1,21 +1,53; -44,6 +76,8 @@ def run_test(; symbols: DummyLogprobs, __init__, __repr__, mask_bbox_tokens，涉及 `DummyLogprobs, __init__, __repr__`；`vllm/model_executor/models/nemotron_parse.py` modified +3/-2 (5 lines); hunks: -319,8 +319,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11 (55 lines); hunks: -1,21 +1,53; -44,6 +76,8 @@ def run_test(; symbols: DummyLogprobs, __init__, __repr__, mask_bbox_tokens
  - `vllm/model_executor/models/nemotron_parse.py` modified +3/-2 (5 lines); hunks: -319,8 +319,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/generation/test_nemotron_parse.py
@@ -1,21 +1,53 @@
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
+import regex as re
+from vllm.logprobs import Logprob, SampleLogprobs
+from vllm.tokenizers import TokenizerLike
-from ....utils import create_new_process_for_each_test
diff -- vllm/model_executor/models/nemotron_parse.py
@@ -319,8 +319,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
-            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
+            # MergedColumnParallelLinear uses integer indices (0, 1)
+            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", 0),
+            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", 1),
```

- 已读文件:
  - tests: `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11
  - runtime: `vllm/model_executor/models/nemotron_parse.py` modified +3/-2
- 验证与风险: diff 自带测试面 `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `tests/models/test_terratorch.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37613 - [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests

- 链接: https://github.com/vllm-project/vllm/pull/37613
- 状态/时间: merged / 2026-03-20
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`；关联提交 `fb4e8bf442c5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+8/-1，可读 patch 40 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`；技术摘要: 覆盖「[ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests」；主要实现面是 `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1 (9 lines); hunks: -22,8 +22,10; -70,6 +72,7 @@ def _run_test(; symbols: _run_test, _run_vllm_reranker, _run_reranker_test，涉及 `_run_test, _run_vllm_reranker, _run_reranker_test`。
- 代码 diff 细节:
  - `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1 (9 lines); hunks: -22,8 +22,10; -70,6 +72,7 @@ def _run_test(; symbols: _run_test, _run_vllm_reranker, _run_reranker_test
- 关键代码摘录:

```diff
diff -- tests/models/multimodal/pooling/test_llama_nemotron_vl.py
@@ -22,8 +22,10 @@
+from vllm.platforms import current_platform
+from ....utils import ROCM_ENGINE_KWARGS
@@ -70,6 +72,7 @@ def _run_test(
+        **ROCM_ENGINE_KWARGS,
@@ -250,6 +253,7 @@ def _run_vllm_reranker(
+        **ROCM_ENGINE_KWARGS,
```

- 已读文件:
  - tests: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1
- 验证与风险: diff 自带测试面 `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37803 - Enable `NemotronHPuzzle` + `NemotronHMTP`

- 链接: https://github.com/vllm-project/vllm/pull/37803
- 状态/时间: merged / 2026-03-22
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nemotron_h_mtp.py`；关联提交 `e74c17e15331`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+6/-3，可读 patch 28 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Enable `NemotronHPuzzle` + `NemotronHMTP`」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/model_executor/models/nemotron_h_mtp.py`；技术摘要: 覆盖「Enable `NemotronHPuzzle` + `NemotronHMTP`」；主要实现面是 `vllm/model_executor/models/nemotron_h_mtp.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunks: -395,13 +395,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights，涉及 `load_weights`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunks: -395,13 +395,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nemotron_h_mtp.py
@@ -395,13 +395,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        if hasattr(self.config, "n_routed_experts") and self.config.n_routed_experts:
+        num_experts = getattr(self.config, "n_routed_experts", None)
+        if getattr(self.config, "model_type", None) == "nemotron_h_puzzle":
+            num_experts = self.config.mtp_n_routed_experts
+        if num_experts is not None:
-                num_experts=self.config.n_routed_experts,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2
- 验证与风险: runtime 路径改动集中在 `vllm/config/speculative.py`, `vllm/model_executor/models/nemotron_h_mtp.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #36803 - [Test] E2E Nemotron-3-Super tests

- 链接: https://github.com/vllm-project/vllm/pull/36803
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`；关联提交 `56777b5c898d`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 6 个文件，+37/-0，可读 patch 55 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Test] E2E Nemotron-3-Super tests」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`；技术摘要: 覆盖「[Test] E2E Nemotron-3-Super tests」；主要实现面是 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11；`tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11；`tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11。
- 代码 diff 细节:
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- 关键代码摘录:

```diff
diff -- tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml
@@ -0,0 +1,11 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
+accuracy_threshold: 0.93
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
diff -- tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml
@@ -0,0 +1,11 @@
+model_name: "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
+accuracy_threshold: 0.93
+num_questions: 1319
+num_fewshot: 5
+startup_max_wait_seconds: 1200
+server_args: >-
diff -- tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml
@@ -0,0 +1,11 @@
```

- 已读文件:
  - tests: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0
- 验证与风险: diff 自带测试面 `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`, `tests/evals/gsm8k/configs/models-blackwell.txt`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #37903 - nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths

- 链接: https://github.com/vllm-project/vllm/pull/37903
- 状态/时间: merged / 2026-03-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/processors/nano_nemotron_vl.py`；关联提交 `a0d487b2e1d5`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+32/-44，可读 patch 122 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`；技术摘要: 覆盖「nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths」；主要实现面是 `vllm/transformers_utils/processors/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44 (76 lines); hunks: -8,6 +8,7; -66,6 +67,30 @@ def input_conditioner(x: torch.Tensor, norm_mean: torch.Tenso...; symbols: input_conditioner, _bicubic_from_ndarray, dynamic_preprocess, video_to_pixel_values，涉及 `input_conditioner, _bicubic_from_ndarray, dynamic_preprocess`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44 (76 lines); hunks: -8,6 +8,7; -66,6 +67,30 @@ def input_conditioner(x: torch.Tensor, norm_mean: torch.Tenso...; symbols: input_conditioner, _bicubic_from_ndarray, dynamic_preprocess, video_to_pixel_values
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/nano_nemotron_vl.py
@@ -8,6 +8,7 @@
+import warnings
@@ -66,6 +67,30 @@ def input_conditioner(x: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.
+def _bicubic_from_ndarray(
+    array: npt.NDArray[Any], *, size: tuple[int, int]
+) -> torch.Tensor:
+    """
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/processors/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38018 - [Model] Use helper function to run MM processors with token inputs (where applicable)

- 链接: https://github.com/vllm-project/vllm/pull/38018
- 状态/时间: merged / 2026-03-26
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `e812bf70bd66`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 12 个文件，+215/-145，可读 patch 595 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「[Model] Use helper function to run MM processors with token inputs (where applicable)」；模型线: Nemotron Super；类别: 模型支持/运行时入口；主要 diff: `vllm/transformers_utils/processors/pixtral.py`, `vllm/transformers_utils/processors/voxtral.py`, `vllm/multimodal/processing/processor.py`；技术摘要: 覆盖「[Model] Use helper function to run MM processors with token inputs (where applicable)」；主要实现面是 `vllm/transformers_utils/processors/pixtral.py`, `vllm/transformers_utils/processors/voxtral.py`, `vllm/multimodal/processing/processor.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/pixtral.py` modified +9/-58 (67 lines); hunks: -5,10 +5,7; -55,62 +52,16 @@ class MistralCommonPixtralProcessor(ProcessorMixin):; symbols: MistralCommonPixtralProcessor, __init__, image_break_id, image_token_id，涉及 `MistralCommonPixtralProcessor, __init__, image_break_id`；`vllm/transformers_utils/processors/voxtral.py` modified +8/-54 (62 lines); hunks: -8,9 +8,6; -62,58 +59,15 @@ class MistralCommonVoxtralProcessor(ProcessorMixin):; symbols: MistralCommonVoxtralProcessor, __init__, audio_token_id, begin_audio_token_id，涉及 `MistralCommonVoxtralProcessor, __init__, audio_token_id`；`vllm/multimodal/processing/processor.py` modified +41/-10 (51 lines); hunks: -5,8 +5,15; -21,6 +28,7; symbols: _apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main，涉及 `_apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main`；`vllm/transformers_utils/processors/isaac.py` modified +32/-16 (48 lines); hunks: -1,16 +1,15; -308,15 +307,22 @@ def process_vision_for_patches(; symbols: process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs, IsaacProcessorKwargs，涉及 `process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/pixtral.py` modified +9/-58 (67 lines); hunks: -5,10 +5,7; -55,62 +52,16 @@ class MistralCommonPixtralProcessor(ProcessorMixin):; symbols: MistralCommonPixtralProcessor, __init__, image_break_id, image_token_id
  - `vllm/transformers_utils/processors/voxtral.py` modified +8/-54 (62 lines); hunks: -8,9 +8,6; -62,58 +59,15 @@ class MistralCommonVoxtralProcessor(ProcessorMixin):; symbols: MistralCommonVoxtralProcessor, __init__, audio_token_id, begin_audio_token_id
  - `vllm/multimodal/processing/processor.py` modified +41/-10 (51 lines); hunks: -5,8 +5,15; -21,6 +28,7; symbols: _apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main
  - `vllm/transformers_utils/processors/isaac.py` modified +32/-16 (48 lines); hunks: -1,16 +1,15; -308,15 +307,22 @@ def process_vision_for_patches(; symbols: process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs, IsaacProcessorKwargs
  - `vllm/model_executor/models/pixtral.py` modified +23/-1 (24 lines); hunks: -12,7 +12,7; -62,6 +62,7; symbols: _get_mm_fields_config, _call_hf_processor, _get_prompt_updates
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/pixtral.py
@@ -5,10 +5,7 @@
-from transformers.audio_utils import AudioInput
-from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
-from transformers.video_utils import VideoInput
@@ -55,62 +52,16 @@ class MistralCommonPixtralProcessor(ProcessorMixin):
+        # Back-compatibility for Transformers v4
+        if not hasattr(self.tokenizer, "init_kwargs"):
diff -- vllm/transformers_utils/processors/voxtral.py
@@ -8,9 +8,6 @@
-from transformers.image_utils import ImageInput
-from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
-from transformers.video_utils import VideoInput
@@ -62,58 +59,15 @@ class MistralCommonVoxtralProcessor(ProcessorMixin):
+        # Back-compatibility for Transformers v4
+        if not hasattr(self.tokenizer, "init_kwargs"):
diff -- vllm/multimodal/processing/processor.py
@@ -5,8 +5,15 @@
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/pixtral.py` modified +9/-58; `vllm/transformers_utils/processors/voxtral.py` modified +8/-54; `vllm/multimodal/processing/processor.py` modified +41/-10; `vllm/transformers_utils/processors/isaac.py` modified +32/-16; `vllm/model_executor/models/pixtral.py` modified +23/-1; `vllm/model_executor/models/voxtral.py` modified +11/-3
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/keye.py`, `vllm/model_executor/models/keye_vl1_5.py`, `vllm/model_executor/models/kimi_k25.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38567 - Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018

- 链接: https://github.com/vllm-project/vllm/pull/38567
- 状态/时间: merged / 2026-03-30
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `e812bf70bd66`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+14/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0 (14 lines); hunks: -307,6 +307,20 @@ def get_num_frames_with_most_features(; symbols: get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor, _get_image_fields_config，涉及 `get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0 (14 lines); hunks: -307,6 +307,20 @@ def get_num_frames_with_most_features(; symbols: get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor, _get_image_fields_config
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -307,6 +307,20 @@ def get_num_frames_with_most_features(
+    def _call_hf_processor(
+        self,
+        prompt: str,
+        mm_data: Mapping[str, object],
+        mm_kwargs: Mapping[str, object],
+        tok_kwargs: Mapping[str, object],
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38655 - Fix Nano Nemotron VL regressions

- 链接: https://github.com/vllm-project/vllm/pull/38655
- 状态/时间: merged / 2026-04-03
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；关联提交 `fa9e68022d29`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 7 个文件，+84/-52，可读 patch 331 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Nano Nemotron VL regressions」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；技术摘要: 覆盖「Fix Nano Nemotron VL regressions」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23 (52 lines); hunks: -7,7 +7,6; -17,7 +16,7; symbols: get_hf_processor, is_dynamic_tiler, supports_video, supports_audio，涉及 `get_hf_processor, is_dynamic_tiler, supports_video`；`vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15 (32 lines); hunks: -356,15 +356,6 @@ def _images_to_pixel_values_lst(; -519,7 +510,6 @@ def compute_params(; symbols: _images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams, compute_params，涉及 `_images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23 (52 lines); hunks: -7,7 +7,6; -17,7 +16,7; symbols: get_hf_processor, is_dynamic_tiler, supports_video, supports_audio
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15 (32 lines); hunks: -356,15 +356,6 @@ def _images_to_pixel_values_lst(; -519,7 +510,6 @@ def compute_params(; symbols: _images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams, compute_params
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -7,7 +7,6 @@
-import copy
@@ -17,7 +16,7 @@
-from transformers import BatchFeature
+from transformers import BatchFeature, PretrainedConfig
@@ -210,11 +209,15 @@ def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor:
-        return self.get_hf_processor().dynamic_tiler is not None
diff -- vllm/transformers_utils/processors/nano_nemotron_vl.py
@@ -356,15 +356,6 @@ def _images_to_pixel_values_lst(
-    feature_size_cache: dict[Image.Image, int] = {}
-    @classmethod
-    def get_cached_feature_size(cls, image: Image.Image) -> int:
-        feature_size = cls.feature_size_cache[id(image)]
-        # hard assert that we only use the feature size once
-        del cls.feature_size_cache[id(image)]
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15
- 验证与风险: diff 自带测试面 `tests/models/registry.py`, `tests/models/utils.py`；如果继续改同一模型，优先复跑这些测试并补一个最小 launch/accuracy smoke。

### PR #39029 - nano_nemotron_vl: fix tensor device mismatch exception when video profiling

- 链接: https://github.com/vllm-project/vllm/pull/39029
- 状态/时间: merged / 2026-04-05
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `d56e95223917`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-2，可读 patch 16 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「nano_nemotron_vl: fix tensor device mismatch exception when video profiling」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；未提供可用技术摘要。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2 (5 lines); hunks: -1239,12 +1239,13 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings，涉及 `_create_final_video_embeddings`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2 (5 lines); hunks: -1239,12 +1239,13 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1239,12 +1239,13 @@ def _create_final_video_embeddings(
+        device = video_embeddings.device
-        repl_token_ids = torch.tensor(video_repl.full)
+        repl_token_ids = torch.tensor(video_repl.full, device=device)
-        embed_token_ids = torch.tensor(self._img_context_token_ids)
+        embed_token_ids = torch.tensor(self._img_context_token_ids, device=device)
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38727 - nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len

- 链接: https://github.com/vllm-project/vllm/pull/38727
- 状态/时间: merged / 2026-04-07
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `a9a0e0551f03`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+52/-10，可读 patch 84 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10 (62 lines); hunks: -288,6 +288,35 @@ def get_max_image_tokens(self) -> int:; -306,6 +335,26 @@ def get_num_frames_with_most_features(; symbols: get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features, get_mm_max_tokens_per_item，涉及 `get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10 (62 lines); hunks: -288,6 +288,35 @@ def get_max_image_tokens(self) -> int:; -306,6 +335,26 @@ def get_num_frames_with_most_features(; symbols: get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features, get_mm_max_tokens_per_item
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -288,6 +288,35 @@ def get_max_image_tokens(self) -> int:
+    def get_dummy_image_size_and_max_tokens(
+        self, mm_counts: Mapping[str, int]
+    ) -> tuple[tuple[int, int], int]:
+        processor = self.get_hf_processor()
+        num_images = mm_counts.get("image", 0)
+        if tiler := processor.dynamic_tiler:
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #38538 - nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time

- 链接: https://github.com/vllm-project/vllm/pull/38538
- 状态/时间: merged / 2026-04-09
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；关联提交 `df2503e125f3`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 2 个文件，+74/-18，可读 patch 162 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；技术摘要: 覆盖「nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18 (90 lines); hunks: -597,19 +597,26 @@ def _get_prompt_updates(; -618,7 +625,16 @@ def _extract_audio_from_videos(; symbols: _get_prompt_updates, _extract_audio_from_videos, apply，涉及 `_get_prompt_updates, _extract_audio_from_videos, apply`；`vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0 (2 lines); hunks: -771,6 +771,7 @@ def __init__(; -781,6 +782,7 @@ def __init__(; symbols: __init__，涉及 `__init__`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18 (90 lines); hunks: -597,19 +597,26 @@ def _get_prompt_updates(; -618,7 +625,16 @@ def _extract_audio_from_videos(; symbols: _get_prompt_updates, _extract_audio_from_videos, apply
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0 (2 lines); hunks: -771,6 +771,7 @@ def __init__(; -781,6 +782,7 @@ def __init__(; symbols: __init__
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -597,19 +597,26 @@ def _get_prompt_updates(
-    ) -> tuple[MultiModalDataItems, list[AudioItem]]:
+    ) -> tuple[MultiModalDataItems, list[AudioItem], list[bool]]:
+        Videos whose bytes are missing or that contain no audio stream are
+        silently skipped.  The returned *has_audio* mask is aligned with
+        the video list so callers know which ``<video>`` tokens need an
+        accompanying audio context.
diff -- vllm/transformers_utils/processors/nano_nemotron_vl.py
@@ -771,6 +771,7 @@ def __init__(
+        use_audio_in_video: bool = False,
@@ -781,6 +782,7 @@ def __init__(
+        self.use_audio_in_video = use_audio_in_video
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #37580 - Nemotron Nano VL: Streamline pixel shuffle

- 链接: https://github.com/vllm-project/vllm/pull/37580
- 状态/时间: merged / 2026-04-10
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `270e8a410254`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+11/-36，可读 patch 73 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Nemotron Nano VL: Streamline pixel shuffle」；模型线: Nemotron Super；类别: 模型实现调整；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Nemotron Nano VL: Streamline pixel shuffle」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36 (47 lines); hunks: -1005,38 +1005,27 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1045,22 +1034,8 @@ def pixel_shuffle_dynamic_res(; symbols: __init__, pixel_shuffle, pixel_shuffle_dynamic_res，涉及 `__init__, pixel_shuffle, pixel_shuffle_dynamic_res`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36 (47 lines); hunks: -1005,38 +1005,27 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1045,22 +1034,8 @@ def pixel_shuffle_dynamic_res(; symbols: __init__, pixel_shuffle, pixel_shuffle_dynamic_res
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1005,38 +1005,27 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-        n, w, h, c = x.size()
-        # N, W, H, C --> N, W, H * scale, C // scale
-        x = x.view(
-            n,
-            w,
-            int(h * scale_factor),
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #39901 - FIX: support language_model.backbone naming in NemotronH Nano VL quantization config

- 链接: https://github.com/vllm-project/vllm/pull/39901
- 状态/时间: merged / 2026-04-15
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `8b5531933a7b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+7/-0，可读 patch 21 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「FIX: support language_model.backbone naming in NemotronH Nano VL quantization config」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「FIX: support language_model.backbone naming in NemotronH Nano VL quantization config」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -37,6 +37,7; -903,6 +904,12 @@ class NemotronH_Nano_VL_V2(; symbols: NemotronH_Nano_VL_V2, get_placeholder_str，涉及 `NemotronH_Nano_VL_V2, get_placeholder_str`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -37,6 +37,7; -903,6 +904,12 @@ class NemotronH_Nano_VL_V2(; symbols: NemotronH_Nano_VL_V2, get_placeholder_str
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -37,6 +37,7 @@
+    WeightsMapper,
@@ -903,6 +904,12 @@ class NemotronH_Nano_VL_V2(
+    hf_to_vllm_mapper = WeightsMapper(
+        orig_to_new_prefix={
+            "language_model.backbone": "language_model.model",
+        },
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40283 - Optimize nemotron VL image/video preprocessing

- 链接: https://github.com/vllm-project/vllm/pull/40283
- 状态/时间: merged / 2026-04-19
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/transformers_utils/processors/nano_nemotron_vl.py`；关联提交 `982beae809b1`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+116/-98，可读 patch 384 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Optimize nemotron VL image/video preprocessing」；模型线: Nemotron Super；类别: 性能/后端优化；主要 diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`；技术摘要: 覆盖「Optimize nemotron VL image/video preprocessing」；主要实现面是 `vllm/transformers_utils/processors/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98 (214 lines); hunks: -8,7 +8,6; -26,7 +25,7; symbols: calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize, _bicubic_from_ndarray，涉及 `calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize`。
- 代码 diff 细节:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98 (214 lines); hunks: -8,7 +8,6; -26,7 +25,7; symbols: calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize, _bicubic_from_ndarray
- 关键代码摘录:

```diff
diff -- vllm/transformers_utils/processors/nano_nemotron_vl.py
@@ -8,7 +8,6 @@
-import warnings
@@ -26,7 +25,7 @@
-from vllm.multimodal.processing.processor import PromptUpdateDetails, _seq2tokens
+from vllm.multimodal.processing.processor import PromptUpdateDetails
@@ -63,42 +62,50 @@ def calculate_timestamps(
-def input_conditioner(x: torch.Tensor, norm_mean: torch.Tensor, norm_std: torch.Tensor):
```

- 已读文件:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98
- 验证与风险: runtime 路径改动集中在 `vllm/transformers_utils/processors/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

### PR #40724 - Fix Nano Nemotron VL static image inputs

- 链接: https://github.com/vllm-project/vllm/pull/40724
- 状态/时间: merged / 2026-04-24
- 反查来源: `git log --name-only -- <model-files>` 反查到 `vllm/model_executor/models/nano_nemotron_vl.py`；关联提交 `9ad5abe7722b`；保留自原 history/skill 显式引用
- 代码 diff 已读范围: GitHub Pull Request files API 返回 1 个文件，+3/-1，可读 patch 11 行；本卡优先审计模型相关文件和高变更量文件。
- 动机: 标题「Fix Nano Nemotron VL static image inputs」；模型线: Nemotron Super；类别: 缺陷修复；主要 diff: `vllm/model_executor/models/nano_nemotron_vl.py`；技术摘要: 覆盖「Fix Nano Nemotron VL static image inputs」；主要实现面是 `vllm/model_executor/models/nano_nemotron_vl.py`。下方保留文件级证据、代码摘录和验证风险。
- 实现要点: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1124,7 +1124,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic，涉及 `_parse_and_validate_image_input, _process_image_input_dynamic`。
- 代码 diff 细节:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1124,7 +1124,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic
- 关键代码摘录:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1124,7 +1124,9 @@ def _parse_and_validate_image_input(
-                num_patches=kwargs.pop("image_num_patches"), **kwargs
+                pixel_values_flat=pixel_values_flat,
+                num_patches=kwargs.pop("image_num_patches"),
+                **kwargs,
```

- 已读文件:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1
- 验证与风险: runtime 路径改动集中在 `vllm/model_executor/models/nano_nemotron_vl.py`；风险点是权重加载、并行切分、attention/MoE 后端和 parser 输出，需要至少做一次真实 checkpoint 或等价 mock smoke。

## 补漏结论

- 验收规则: 每个 PR 卡片必须保留反查来源、diff 范围、实现要点、代码摘录、已读文件和验证风险。
- 如果新模型文件落在当前过滤规则之外，先补文件过滤规则，再重新执行本轮 `git log --name-only -- <model-files>` 追溯。
