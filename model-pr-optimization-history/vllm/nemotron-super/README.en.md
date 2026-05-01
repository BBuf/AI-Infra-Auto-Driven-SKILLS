# vllm Nemotron Super Model PR Optimization History

## Scope

- Rebuilt on: 2026-05-01
- Source baseline: `vllm-project/vllm` trace worktree commit `7075df79b3`
- PR collection rule: run `git log --name-only -- <model-files>` on model implementation, config, processor, parser, docs/tests, filter by model keywords in commit subjects, then read each PR's final diff through the GitHub Pull Request files API.
- Preservation rule: PRs explicitly cited by the previous history/skill are retained even if current implementation files no longer trace to them, and the card marks that source.

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/pooling/embed/template/nemotron_embed_vl.jinja` | [#35297](https://github.com/vllm-project/vllm/pull/35297) |
| `examples/pooling/score/template/nemotron-rerank.jinja` | no direct PR-number commit |
| `examples/pooling/score/template/nemotron-vl-rerank.jinja` | [#35735](https://github.com/vllm-project/vllm/pull/35735) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` | [#36803](https://github.com/vllm-project/vllm/pull/36803) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` | [#34725](https://github.com/vllm-project/vllm/pull/34725) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` | [#34725](https://github.com/vllm-project/vllm/pull/34725) |
| `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-vllm-cutlass.yaml` | no direct PR-number commit |
| `tests/models/language/pooling_mteb_test/test_nemotron.py` | no direct PR-number commit |
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
| `vllm/transformers_utils/processors/nemotron_vl.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 60
- Extra PRs preserved from existing docs: 2
- Total PRs in this document: 62
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
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

## Per-PR Diff Audit Cards

### PR #6611 - [Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)

- Link: https://github.com/vllm-project/vllm/pull/6611
- Status/date: merged / 2024-07-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`; associated commits `07278c37ddd8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +776/-1, 847 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`; technical summary: Covers "[Model] Support Nemotron models (Nemotron-3, Nemotron-4, Minitron)"; the main implementation surface is `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: _cast_if_autocast_enabled, NemotronLayerNorm1P, __init__, forward, touching `_cast_if_autocast_enabled, NemotronLayerNorm1P, __init__`; `vllm/transformers_utils/configs/nemotron.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: NemotronConfig, to, __init__, _rope_scaling_validation, touching `NemotronConfig, to, __init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: _cast_if_autocast_enabled, NemotronLayerNorm1P, __init__, forward
  - `vllm/transformers_utils/configs/nemotron.py` added +209/-0 (209 lines); hunks: -0,0 +1,209; symbols: NemotronConfig, to, __init__, _rope_scaling_validation
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron.py` added +531/-0; `vllm/transformers_utils/configs/nemotron.py` added +209/-0
- Risk and verification: Runtime changes concentrate in `.buildkite/lm-eval-harness/configs/Minitron-4B-Base.yaml`, `.buildkite/lm-eval-harness/configs/models-small.txt`, `vllm/model_executor/layers/activation.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #7611 - [Model] Align nemotron config with final HF state and fix lm-eval-small

- Link: https://github.com/vllm-project/vllm/pull/7611
- Status/date: merged / 2024-08-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron.py`, `vllm/transformers_utils/configs/nemotron.py`; associated commits `44f26a946645`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +29/-35, 181 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Align nemotron config with final HF state and fix lm-eval-small"; model line: Nemotron Super; category: bug fix; main diff: `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/nemotron.py`; technical summary: Covers "[Model] Align nemotron config with final HF state and fix lm-eval-small"; the main implementation surface is `vllm/transformers_utils/configs/nemotron.py`, `vllm/model_executor/models/nemotron.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/configs/nemotron.py` modified +18/-24 (42 lines); hunks: -35,20 +35,20 @@ class NemotronConfig(PretrainedConfig):; -63,38 +63,33 @@ class NemotronConfig(PretrainedConfig):; symbols: NemotronConfig, __init__, touching `NemotronConfig, __init__`; `vllm/model_executor/models/nemotron.py` modified +3/-3 (6 lines); hunks: -53,7 +53,7; -161,7 +161,7 @@ def __init__(; symbols: _cast_if_autocast_enabled, __init__, touching `_cast_if_autocast_enabled, __init__`.
- Code diff details:
  - `vllm/transformers_utils/configs/nemotron.py` modified +18/-24 (42 lines); hunks: -35,20 +35,20 @@ class NemotronConfig(PretrainedConfig):; -63,38 +63,33 @@ class NemotronConfig(PretrainedConfig):; symbols: NemotronConfig, __init__
  - `vllm/model_executor/models/nemotron.py` modified +3/-3 (6 lines); hunks: -53,7 +53,7; -161,7 +161,7 @@ def __init__(; symbols: _cast_if_autocast_enabled, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/configs/nemotron.py` modified +18/-24; `vllm/model_executor/models/nemotron.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `.buildkite/lm-eval-harness/configs/Minitron-4B-Base-FP8.yaml`, `.buildkite/lm-eval-harness/configs/models-small.txt`, `vllm/model_executor/layers/rotary_embedding.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #15008 - [Model] Update support for NemotronNAS models

- Link: https://github.com/vllm-project/vllm/pull/15008
- Status/date: merged / 2025-03-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_nas.py`; associated commits `3aa2b6a63714`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +524/-133, 764 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Update support for NemotronNAS models"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/model_executor/models/nemotron_nas.py`; technical summary: Covers "[Model] Update support for NemotronNAS models"; the main implementation surface is `vllm/model_executor/models/nemotron_nas.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_nas.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__, touching `_ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_nas.py` added +454/-0 (454 lines); hunks: -0,0 +1,454; symbols: _ffn_mult_to_intermediate_size, _find_multiple, DeciLMDecoderLayer, __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` added +454/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18427 - [Model] Add support for YARN in NemotronNAS models

- Link: https://github.com/vllm-project/vllm/pull/18427
- Status/date: merged / 2025-05-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_nas.py`; associated commits `6d68030f1cac`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +67/-16, 129 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for YARN in NemotronNAS models"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_nas.py`; technical summary: Covers "[Model] Add support for YARN in NemotronNAS models"; the main implementation surface is `vllm/model_executor/models/nemotron_nas.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_nas.py` modified +46/-2 (48 lines); hunks: -23,18 +23,20; -62,6 +64,48 @@ def _find_multiple(n: int, k: int) -> int:; symbols: _find_multiple, DeciLMAttention, __init__, _init_rotary_emb, touching `_find_multiple, DeciLMAttention, __init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_nas.py` modified +46/-2 (48 lines); hunks: -23,18 +23,20; -62,6 +64,48 @@ def _find_multiple(n: int, k: int) -> int:; symbols: _find_multiple, DeciLMAttention, __init__, _init_rotary_emb
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` modified +46/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/nemotron_nas.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #18863 - [Model] NemotronH support

- Link: https://github.com/vllm-project/vllm/pull/18863
- Status/date: merged / 2025-06-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; associated commits `cb6d572e85a3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +829/-0, 866 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] NemotronH support"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; technical summary: Covers "[Model] NemotronH support"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunks: -0,0 +1,565; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer, touching `NemotronHMLP, __init__, forward`; `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: NemotronHConfig, to, __init__, layers_block_type, touching `NemotronHConfig, to, __init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` added +565/-0 (565 lines); hunks: -0,0 +1,565; symbols: NemotronHMLP, __init__, forward, NemotronHMLPDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0 (258 lines); hunks: -0,0 +1,258; symbols: NemotronHConfig, to, __init__, layers_block_type
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` added +565/-0; `vllm/transformers_utils/configs/nemotron_h.py` added +258/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #19249 - [Model] Optimize nemotron_h implementation

- Link: https://github.com/vllm-project/vllm/pull/19249
- Status/date: merged / 2025-06-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `7661e92ef85e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +16/-8, 88 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Optimize nemotron_h implementation"; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[Model] Optimize nemotron_h implementation"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +16/-8 (24 lines); hunks: -1,4 +1,5; -29,7 +30,7; symbols: __init__, NemotronHForCausalLM, touching `__init__, NemotronHForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +16/-8 (24 lines); hunks: -1,4 +1,5; -29,7 +30,7; symbols: __init__, NemotronHForCausalLM
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +16/-8
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20349 - [VLM] Add Nemotron-Nano-VL-8B-V1 support

- Link: https://github.com/vllm-project/vllm/pull/20349
- Status/date: merged / 2025-07-17
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`; associated commits `4ef00b5caca4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +701/-3, 837 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[VLM] Add Nemotron-Nano-VL-8B-V1 support"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`; technical summary: Covers "[VLM] Add Nemotron-Nano-VL-8B-V1 support"; the main implementation surface is `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/transformers_utils/configs/nemotron.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_vl.py` added +505/-0 (505 lines); hunks: -0,0 +1,505; symbols: NemotronVLProcessor, __init__, image_token_id, _preprocess_image, touching `NemotronVLProcessor, __init__, image_token_id`; `tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: _get_expected_num_patches, _run_check, test_processor_override, touching `_get_expected_num_patches, _run_check, test_processor_override`; `vllm/transformers_utils/configs/nemotron.py` modified +1/-1 (2 lines); hunks: -202,4 +202,4 @@ def _rope_scaling_validation(self):; symbols: _rope_scaling_validation, touching `_rope_scaling_validation`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_vl.py` added +505/-0 (505 lines); hunks: -0,0 +1,505; symbols: NemotronVLProcessor, __init__, image_token_id, _preprocess_image
  - `tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0 (134 lines); hunks: -0,0 +1,134; symbols: _get_expected_num_patches, _run_check, test_processor_override
  - `vllm/transformers_utils/configs/nemotron.py` modified +1/-1 (2 lines); hunks: -202,4 +202,4 @@ def _rope_scaling_validation(self):; symbols: _rope_scaling_validation
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` added +505/-0; `vllm/transformers_utils/configs/nemotron.py` modified +1/-1
  - tests: `tests/models/multimodal/processing/test_nemotron_vl.py` added +134/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22349 - [Model] NemotronH Support

- Link: https://github.com/vllm-project/vllm/pull/22349
- Status/date: merged / 2025-08-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; associated commits `14a5d903ab82`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +23/-7, 91 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] NemotronH Support"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; technical summary: Covers "[Model] NemotronH Support"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +21/-5 (26 lines); hunks: -64,20 +64,32 @@ class NemotronHMLP(nn.Module):; -110,6 +122,7 @@ def __init__(; symbols: NemotronHMLP, __init__, touching `NemotronHMLP, __init__`; `vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2 (4 lines); hunks: -151,7 +151,7 @@ def __init__(; -194,7 +194,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +21/-5 (26 lines); hunks: -64,20 +64,32 @@ class NemotronHMLP(nn.Module):; -110,6 +122,7 @@ def __init__(; symbols: NemotronHMLP, __init__
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2 (4 lines); hunks: -151,7 +151,7 @@ def __init__(; -194,7 +194,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +21/-5; `vllm/transformers_utils/configs/nemotron_h.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22739 - [Bugfix] Fix Nemotron VL image processing

- Link: https://github.com/vllm-project/vllm/pull/22739
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`; associated commits `a01e0018b50f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +190/-4, 234 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Nemotron VL image processing"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`; technical summary: Covers "[Bugfix] Fix Nemotron VL image processing"; the main implementation surface is `vllm/model_executor/models/nemotron_vl.py`, `tests/models/multimodal/processing/test_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_vl.py` modified +186/-0 (186 lines); hunks: -13,6 +13,7; -27,6 +28,7; symbols: build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets, dynamic_preprocess_nemotron_vl, touching `build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets`; `tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -23,15 +23,15 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches, touching `_get_expected_num_patches`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_vl.py` modified +186/-0 (186 lines); hunks: -13,6 +13,7; -27,6 +28,7; symbols: build_transform, find_closest_aspect_ratio, calculate_nemotron_vl_targets, dynamic_preprocess_nemotron_vl
  - `tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -23,15 +23,15 @@ def _get_expected_num_patches(; symbols: _get_expected_num_patches
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +186/-0
  - tests: `tests/models/multimodal/processing/test_nemotron_vl.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_nemotron_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22023 - Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/22023
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_vl.py`; associated commits `e75f34226161`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-23, 43 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nemotron_vl.py`; technical summary: Covers "Migrate InternVLImagePixelInputs (in nemotron_vl.py) to TensorSchema"; the main implementation surface is `vllm/model_executor/models/nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_vl.py` modified +5/-23 (28 lines); hunks: -458,27 +458,6 @@ def extract_feature(self, pixel_values: torch.Tensor) -> to...; -516,9 +495,12 @@ def _parse_and_validate_image_input(; symbols: extract_feature, _validate_pixel_values, _validate_shape, _parse_and_validate_image_input, touching `extract_feature, _validate_pixel_values, _validate_shape`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_vl.py` modified +5/-23 (28 lines); hunks: -458,27 +458,6 @@ def extract_feature(self, pixel_values: torch.Tensor) -> to...; -516,9 +495,12 @@ def _parse_and_validate_image_input(; symbols: extract_feature, _validate_pixel_values, _validate_shape, _parse_and_validate_image_input
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +5/-23
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #23644 - Support for NemotronH Nano VLM

- Link: https://github.com/vllm-project/vllm/pull/23644
- Status/date: merged / 2025-09-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `72d30108a0fe`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +1400/-1, 1423 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support for NemotronH Nano VLM"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Support for NemotronH Nano VLM"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0 (1395 lines); hunks: -0,0 +1,1395; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, touching `NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0 (1395 lines); hunks: -0,0 +1,1395; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImageEmbeddinInputs, NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` added +1395/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #25708 - [Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2

- Link: https://github.com/vllm-project/vllm/pull/25708
- Status/date: merged / 2025-09-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `57329a8c013c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +6/-6, 47 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "[Model] rename NemotronH_Nano_VL -> NemotronH_Nano_VL_V2"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -869,8 +869,8 @@ def get_dummy_mm_data(; -1249,7 +1249,7 @@ def print_architecture(self,; symbols: get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2, get_placeholder_str, touching `get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4 (8 lines); hunks: -869,8 +869,8 @@ def get_dummy_mm_data(; -1249,7 +1249,7 @@ def print_architecture(self,; symbols: get_dummy_mm_data, NemotronH_Nano_VL, NemotronH_Nano_VL_V2, get_placeholder_str
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22980 - EVS Support (Video tokens pruning)

- Link: https://github.com/vllm-project/vllm/pull/22980
- Status/date: merged / 2025-09-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +783/-39, 1076 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "EVS Support (Video tokens pruning)"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/multimodal/evs.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`; technical summary: Covers "EVS Support (Video tokens pruning)"; the main implementation surface is `vllm/multimodal/evs.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `tests/models/multimodal/generation/test_qwen2_5_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/multimodal/evs.py` added +273/-0 (273 lines); hunks: -0,0 +1,273; symbols: compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media, recompute_mrope_positions, touching `compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media`; `vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12 (238 lines); hunks: -25,9 +25,9; -58,15 +58,22; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs, touching `Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs`; `tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos, touching `qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos`; `vllm/model_executor/models/interfaces.py` modified +55/-0 (55 lines); hunks: -115,6 +115,42 @@ def get_input_embeddings(; -142,6 +178,25 @@ def supports_multimodal_encoder_tp_data(; symbols: get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions, supports_multimodal, touching `get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions`.
- Code diff details:
  - `vllm/multimodal/evs.py` added +273/-0 (273 lines); hunks: -0,0 +1,273; symbols: compute_retained_tokens_count, compute_retention_mask, compute_mrope_for_media, recompute_mrope_positions
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12 (238 lines); hunks: -25,9 +25,9; -58,15 +58,22; symbols: Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs
  - `tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0 (132 lines); hunks: -0,0 +1,132; symbols: qwen2_5_vl_chat_template, test_qwen2_5_vl_evs_functionality, test_qwen2_5_vl_evs_batched_videos
  - `vllm/model_executor/models/interfaces.py` modified +55/-0 (55 lines); hunks: -115,6 +115,42 @@ def get_input_embeddings(; -142,6 +178,25 @@ def supports_multimodal_encoder_tp_data(; symbols: get_input_embeddings, SupportsMultiModalPruning, recompute_mrope_positions, supports_multimodal
  - `vllm/config/multimodal.py` modified +9/-0 (9 lines); hunks: -78,6 +78,11 @@ class MultiModalConfig:; -118,3 +123,7 @@ def merge_mm_processor_kwargs(; symbols: MultiModalConfig, compute_hash, merge_mm_processor_kwargs, is_multimodal_pruning_enabled
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/multimodal/evs.py` added +273/-0; `vllm/model_executor/models/qwen2_5_vl.py` modified +226/-12; `vllm/model_executor/models/interfaces.py` modified +55/-0; `vllm/config/multimodal.py` modified +9/-0; `vllm/v1/worker/gpu_model_runner.py` modified +67/-16; `vllm/config/model.py` modified +16/-11
  - tests: `tests/models/multimodal/generation/test_qwen2_5_vl.py` added +132/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_qwen2_5_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #26186 - Fix issue of using only the part of video frame [Nemotron Nano]

- Link: https://github.com/vllm-project/vllm/pull/26186
- Status/date: merged / 2025-10-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `5a05f2660370`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-1, 9 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix issue of using only the part of video frame [Nemotron Nano]"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Fix issue of using only the part of video frame [Nemotron Nano]"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def video_to_pixel_values(; symbols: video_to_pixel_values, touching `video_to_pixel_values`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1 (2 lines); hunks: -208,7 +208,7 @@ def video_to_pixel_values(; symbols: video_to_pixel_values
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -208,7 +208,7 @@ def video_to_pixel_values(
-        frames_tensors.append(pil_frame[0])
+        frames_tensors.append(pil_frame[-1])
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26269 - [Model] EVS support for nano_nemotron_vl

- Link: https://github.com/vllm-project/vllm/pull/26269
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `b8f603cebe39`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +224/-31, 447 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] EVS support for nano_nemotron_vl"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "[Model] EVS support for nano_nemotron_vl"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19 (226 lines); hunks: -30,6 +30,7; -44,6 +45,10; symbols: __init__, supports_video, _preprocess_video, get_image_repl, touching `__init__, supports_video, _preprocess_video`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19 (226 lines); hunks: -30,6 +30,7; -44,6 +45,10; symbols: __init__, supports_video, _preprocess_video, get_image_repl
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +207/-19
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/multimodal/evs.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27107 - Nemotron Nano V2 VL + EVS Video Support

- Link: https://github.com/vllm-project/vllm/pull/27107
- Status/date: merged / 2025-10-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `e93ff6c8b92b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +316/-105, 771 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Nemotron Nano V2 VL + EVS Video Support"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Nemotron Nano V2 VL + EVS Video Support"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49 (315 lines); hunks: -14,14 +14,15; -53,12 +54,14; symbols: NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values, input_conditioner, touching `NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49 (315 lines); hunks: -14,14 +14,15; -53,12 +54,14; symbols: NanoNemotronVLVideoPixelInputs, NanoNemotronVLVideoEmbeddingInputs, video_to_pixel_values, input_conditioner
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +266/-49
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`, `vllm/multimodal/profiling.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25863 - [Model] Add MoE support for NemotronH

- Link: https://github.com/vllm-project/vllm/pull/25863
- Status/date: merged / 2025-10-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; associated commits `61089465a610`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +413/-39, 765 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add MoE support for NemotronH"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`; technical summary: Covers "[Model] Add MoE support for NemotronH"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`, `vllm/transformers_utils/configs/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunks: -18,21 +18,27; -54,16 +60,19; symbols: NemotronHMLP, __init__, forward, NemotronHMoE, touching `NemotronHMLP, __init__, forward`; `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunks: -185,6 +185,15 @@ def __init__(; -241,6 +250,15 @@ def __init__(; symbols: __init__, layers_block_type, touching `__init__, layers_block_type`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +329/-27 (356 lines); hunks: -18,21 +18,27; -54,16 +60,19; symbols: NemotronHMLP, __init__, forward, NemotronHMoE
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0 (20 lines); hunks: -185,6 +185,15 @@ def __init__(; -241,6 +250,15 @@ def __init__(; symbols: __init__, layers_block_type
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +329/-27; `vllm/transformers_utils/configs/nemotron_h.py` modified +20/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/config.py`, `vllm/model_executor/layers/fused_moe/fused_moe.py`, `vllm/model_executor/layers/fused_moe/layer.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27968 - [Model][Bugfix] fix pipeline parallelism support for NemotronH

- Link: https://github.com/vllm-project/vllm/pull/27968
- Status/date: merged / 2025-11-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `77f8001f5330`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +13/-5, 67 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Bugfix] fix pipeline parallelism support for NemotronH"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[Model][Bugfix] fix pipeline parallelism support for NemotronH"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +13/-5 (18 lines); hunks: -20,6 +20,7; -549,7 +550,7 @@ def get_layer(prefix: str):; symbols: get_layer, forward, load_weights, touching `get_layer, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +13/-5 (18 lines); hunks: -20,6 +20,7; -549,7 +550,7 @@ def get_layer(prefix: str):; symbols: get_layer, forward, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +13/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30795 - Fix nemotron_nas intermediate_size computation

- Link: https://github.com/vllm-project/vllm/pull/30795
- Status/date: merged / 2025-12-17
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_nas.py`; associated commits `f5db6385a19b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-4, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix nemotron_nas intermediate_size computation"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_nas.py`; technical summary: Covers "Fix nemotron_nas intermediate_size computation"; the main implementation surface is `vllm/model_executor/models/nemotron_nas.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_nas.py` modified +7/-4 (11 lines); hunks: -169,10 +169,13 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_nas.py` modified +7/-4 (11 lines); hunks: -169,10 +169,13 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_nas.py` modified +7/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_nas.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31539 - Add get_expert_mapping to NemotronHModel (for LoRA support)

- Link: https://github.com/vllm-project/vllm/pull/31539
- Status/date: merged / 2025-12-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `108a2728f74d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-10, 38 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add get_expert_mapping to NemotronHModel (for LoRA support)"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Add get_expert_mapping to NemotronHModel (for LoRA support)"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +14/-10 (24 lines); hunks: -632,14 +632,7 @@ def forward(; -653,8 +646,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights, get_expert_mapping, touching `forward, load_weights, get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +14/-10 (24 lines); hunks: -632,14 +632,7 @@ def forward(; -653,8 +646,19 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, load_weights, get_expert_mapping
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +14/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30864 - [Model] Nemotron Parse 1.1 Support

- Link: https://github.com/vllm-project/vllm/pull/30864
- Status/date: merged / 2026-01-05
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/nemotron_parse.py`; associated commits `ee212918250a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 13 files, +1117/-31, 1329 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Nemotron Parse 1.1 Support"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_parse.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "[Model] Nemotron Parse 1.1 Support"; the main implementation surface is `vllm/model_executor/models/nemotron_parse.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_parse.py` added +958/-0 (958 lines); hunks: -0,0 +1,958; symbols: BartScaledWordEmbedding, __init__, forward, BartParallelLMHead, touching `BartScaledWordEmbedding, __init__, forward`; `tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: run_test, test_models, touching `run_test, test_models`; `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7 (9 lines); hunks: -1220,7 +1220,7 @@ def extract_feature(self, pixel_values):; -1695,12 +1695,7 @@ def get_vit_model_from_radio_config(self, hf_config):; symbols: extract_feature, get_vit_model_from_radio_config, touching `extract_feature, get_vit_model_from_radio_config`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_parse.py` added +958/-0 (958 lines); hunks: -0,0 +1,958; symbols: BartScaledWordEmbedding, __init__, forward, BartParallelLMHead
  - `tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0 (89 lines); hunks: -0,0 +1,89; symbols: run_test, test_models
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7 (9 lines); hunks: -1220,7 +1220,7 @@ def extract_feature(self, pixel_values):; -1695,12 +1695,7 @@ def get_vit_model_from_radio_config(self, hf_config):; symbols: extract_feature, get_vit_model_from_radio_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_parse.py` added +958/-0; `vllm/model_executor/models/nano_nemotron_vl.py` modified +2/-7
  - tests: `tests/models/multimodal/generation/test_nemotron_parse.py` added +89/-0
- Risk and verification: The diff ships test coverage in `tests/conftest.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `tests/models/multimodal/pooling/test_radio.py`, `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #31807 - [NemotronH] Use ReplicatedLinear for fc1_latent_proj

- Link: https://github.com/vllm-project/vllm/pull/31807
- Status/date: merged / 2026-01-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `28c94770adfc`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-5, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NemotronH] Use ReplicatedLinear for fc1_latent_proj"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[NemotronH] Use ReplicatedLinear for fc1_latent_proj"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +1/-5 (6 lines); hunks: -210,16 +210,12 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-5 (6 lines); hunks: -210,16 +210,12 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #31898 - Enable quantized attention in NemotronH models

- Link: https://github.com/vllm-project/vllm/pull/31898
- Status/date: merged / 2026-01-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `bf184a66218b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +4/-0, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable quantized attention in NemotronH models"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Enable quantized attention in NemotronH models"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -483,6 +483,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-0 (1 lines); hunks: -483,6 +483,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -483,6 +483,7 @@ def __init__(
+            quant_config=quant_config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #30802 - Add support for LoRA adapters in Nemotron-H models

- Link: https://github.com/vllm-project/vllm/pull/30802
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `aa7f37ccfa16`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 10 files, +497/-27, 717 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add support for LoRA adapters in Nemotron-H models"; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Add support for LoRA adapters in Nemotron-H models"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -747,6 +747,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM, touching `NemotronHForCausalLM`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -747,6 +747,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -747,6 +747,9 @@ class NemotronHForCausalLM(
+    # Relevant only if self.has_moe is True
+    is_non_gated_moe: bool = True
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-0
- Risk and verification: The diff ships test coverage in `tests/lora/test_layers.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32121 - support dynamic resolution image encoding for Nemotron Nano VL

- Link: https://github.com/vllm-project/vllm/pull/32121
- Status/date: merged / 2026-01-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `cd3ac5b79703`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +755/-164, 1299 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "support dynamic resolution image encoding for Nemotron Nano VL"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "support dynamic resolution image encoding for Nemotron Nano VL"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141 (685 lines); hunks: -8,11 +8,15; -23,6 +27,7; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs, calculate_timestamps, touching `NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141 (685 lines); hunks: -8,11 +8,15; -23,6 +27,7; symbols: NanoNemotronVLImagePixelInputs, NanoNemotronVLImagePixelInputsDynamic, NanoNemotronVLImageEmbeddingInputs, calculate_timestamps
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +544/-141
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/intern_vit.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32682 - [Bugfix] Fix Nemotron-Nano-v2-vlm static resolution

- Link: https://github.com/vllm-project/vllm/pull/32682
- Status/date: merged / 2026-01-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `27ca95b3c9e6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Nemotron-Nano-v2-vlm static resolution"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "[Bugfix] Fix Nemotron-Nano-v2-vlm static resolution"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1678,7 +1678,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic, touching `_parse_and_validate_image_input, _process_image_input_dynamic`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1678,7 +1678,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1678,7 +1678,9 @@ def _parse_and_validate_image_input(
-            return NanoNemotronVLImagePixelInputs(**kwargs)
+            return NanoNemotronVLImagePixelInputs(
+                num_patches=kwargs.pop("image_num_patches"), **kwargs
+            )
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32265 - [LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models

- Link: https://github.com/vllm-project/vllm/pull/32265
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `f3a5ee705fa9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +31/-0, 106 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[LoRA][Spec Decode] Support LoRA for Nemotron-H MTP models"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -771,6 +771,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM, get_mamba_state_dtype_from_config, touching `NemotronHForCausalLM, get_mamba_state_dtype_from_config`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-0 (3 lines); hunks: -771,6 +771,9 @@ class NemotronHForCausalLM(; symbols: NemotronHForCausalLM, get_mamba_state_dtype_from_config
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -771,6 +771,9 @@ class NemotronHForCausalLM(
+    # Skip MTP (Multi-Token Prediction) layers during LoRA loading
+    lora_skip_prefixes = ["mtp."]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-0
- Risk and verification: Runtime changes concentrate in `vllm/lora/lora_model.py`, `vllm/lora/worker_manager.py`, `vllm/model_executor/models/interfaces.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32549 - Support heterogeneous NemotronHPuzzle model

- Link: https://github.com/vllm-project/vllm/pull/32549
- Status/date: merged / 2026-01-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `83fb2d09e8f6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +75/-5, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support heterogeneous NemotronHPuzzle model"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Support heterogeneous NemotronHPuzzle model"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +43/-3 (46 lines); hunks: -354,8 +354,12 @@ def __init__(; -479,6 +483,9 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +43/-3 (46 lines); hunks: -354,8 +354,12 @@ def __init__(; -479,6 +483,9 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +43/-3
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33189 - [Misc][Build] Lazy load cv2 in nemotron_parse.py

- Link: https://github.com/vllm-project/vllm/pull/33189
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_parse.py`; associated commits `9e138cb01d65`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 26 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][Build] Lazy load cv2 in nemotron_parse.py"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_parse.py`; technical summary: Covers "[Misc][Build] Lazy load cv2 in nemotron_parse.py"; the main implementation surface is `vllm/model_executor/models/nemotron_parse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_parse.py` modified +4/-1 (5 lines); hunks: -11,7 +11,6; -416,6 +415,8 @@ def _create_transforms(self):; symbols: _create_transforms, _resize_with_aspect_ratio, touching `_create_transforms, _resize_with_aspect_ratio`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_parse.py` modified +4/-1 (5 lines); hunks: -11,7 +11,6; -416,6 +415,8 @@ def _create_transforms(self):; symbols: _create_transforms, _resize_with_aspect_ratio
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_parse.py
@@ -11,7 +11,6 @@
-import cv2
@@ -416,6 +415,8 @@ def _create_transforms(self):
+        import cv2
@@ -457,6 +458,8 @@ def _resize_with_aspect_ratio(self, image: np.ndarray) -> np.ndarray:
+        import cv2
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_parse.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_parse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32669 - Bugfix: Pass router logits dtype in nemotron shared experts

- Link: https://github.com/vllm-project/vllm/pull/32669
- Status/date: merged / 2026-01-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `e01ff5c070f4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Bugfix: Pass router logits dtype in nemotron shared experts"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Bugfix: Pass router logits dtype in nemotron shared experts"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +3/-1 (4 lines); hunks: -145,11 +145,12 @@ def __init__(; -209,6 +210,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +3/-1 (4 lines); hunks: -145,11 +145,12 @@ def __init__(; -209,6 +210,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -145,11 +145,12 @@ def __init__(
+        router_logits_dtype = torch.float32
-            params_dtype=torch.float32,
+            params_dtype=router_logits_dtype,
@@ -209,6 +210,7 @@ def __init__(
+            router_logits_dtype=router_logits_dtype,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #32790 - [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)

- Link: https://github.com/vllm-project/vllm/pull/32790
- Status/date: merged / 2026-02-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `0aca8b8c628e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +303/-58, 499 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)"; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +30/-42 (72 lines); hunks: -188,10 +188,29 @@ def __init__(; -211,30 +230,9 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +30/-42 (72 lines); hunks: -188,10 +188,29 @@ def __init__(; -211,30 +230,9 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +30/-42
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #33506 - [Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4

- Link: https://github.com/vllm-project/vllm/pull/33506
- Status/date: merged / 2026-02-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +197/-45, 562 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4"; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`; technical summary: Covers "[Kernel] Support Flashinfer trtllm fused MoE non gated FP8 & NVFP4"; the main implementation surface is `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py`, `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5 (93 lines); hunks: -6,6 +6,7; -18,14 +19,28 @@ class FlashinferMoeBackend(Enum):; symbols: FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31, rotate_weights_for_fi_trtllm_fp8_per_tensor_moe, touching `FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31`; `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19 (70 lines); hunks: -15,6 +15,10; -50,8 +54,8 @@ def _supports_current_device() -> bool:; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation, touching `_supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme`; `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6 (14 lines); hunks: -35,8 +35,8 @@ def _supports_current_device() -> bool:; -52,8 +52,7 @@ def _supports_quant_scheme(; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation, touching `_supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme`; `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3 (7 lines); hunks: -937,10 +937,11 @@ def apply_monolithic(; symbols: apply_monolithic, touching `apply_monolithic`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5 (93 lines); hunks: -6,6 +6,7; -18,14 +19,28 @@ class FlashinferMoeBackend(Enum):; symbols: FlashinferMoeBackend, activation_to_flashinfer_int, swap_w13_to_w31, rotate_weights_for_fi_trtllm_fp8_per_tensor_moe
  - `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19 (70 lines); hunks: -15,6 +15,10; -50,8 +54,8 @@ def _supports_current_device() -> bool:; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6 (14 lines); hunks: -35,8 +35,8 @@ def _supports_current_device() -> bool:; -52,8 +52,7 @@ def _supports_quant_scheme(; symbols: _supports_current_device, _supports_no_act_and_mul, _supports_quant_scheme, _supports_activation
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3 (7 lines); hunks: -937,10 +937,11 @@ def apply_monolithic(; symbols: apply_monolithic
  - `tests/kernels/moe/test_flashinfer.py` modified +46/-12 (58 lines); hunks: -71,7 +71,8 @@ def quant_fp8_per_tensor_batches(a):; -81,6 +82,20 @@ def quant_fp8_per_tensor_batches(a):; symbols: quant_fp8_per_tensor_batches, check_accuracy, TestData, make_moe_tensors_8bit
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py` modified +88/-5; `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py` modified +51/-19; `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py` modified +8/-6; `vllm/model_executor/layers/quantization/modelopt.py` modified +4/-3
  - tests: `tests/kernels/moe/test_flashinfer.py` modified +46/-12
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_flashinfer.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34582 - [NemotronH] Do not force router to run in fp32

- Link: https://github.com/vllm-project/vllm/pull/34582
- Status/date: merged / 2026-02-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `3b30e6150777`, `3eff45d793da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +5/-4, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NemotronH] Do not force router to run in fp32"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[NemotronH] Do not force router to run in fp32"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +1/-4 (5 lines); hunks: -148,12 +148,10 @@ def __init__(; -232,7 +230,6 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +1/-4 (5 lines); hunks: -148,12 +148,10 @@ def __init__(; -232,7 +230,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +1/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/flashinfer_trtllm_moe.py`, `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34725 - [Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4

- Link: https://github.com/vllm-project/vllm/pull/34725
- Status/date: merged / 2026-02-18
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`; associated commits `caeb887bf633`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +20/-0, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4"; model line: Nemotron Super; category: bug fix; main diff: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`; technical summary: Covers "[Bugfix] Fix NVFP4 TRTLLM MoE non-gated support; add gsm8k for Nemotron-3-Nano FP8+NVFP4"; the main implementation surface is `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8; `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8.
- Code diff details:
  - `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
  - `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0 (8 lines); hunks: -0,0 +1,8
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml` added +8/-0; `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml` added +8/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-Fp8-ModelOpt-fi-trtllm.yaml`, `tests/evals/gsm8k/configs/moe-refactor/Nemotron-Nano-30B-NvFp4-ModelOpt-fi-cutlass.yaml`, `tests/evals/gsm8k/configs/moe-refactor/config-b200.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #34808 - Revert "[NemotronH] Do not force router to run in fp32 (#34582)"

- Link: https://github.com/vllm-project/vllm/pull/34808
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `3eff45d793da`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-1, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[NemotronH] Do not force router to run in fp32 (#34582)""; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Revert "[NemotronH] Do not force router to run in fp32 (#34582)""; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +4/-1 (5 lines); hunks: -148,10 +148,12 @@ def __init__(; -230,6 +232,7 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +4/-1 (5 lines); hunks: -148,10 +148,12 @@ def __init__(; -230,6 +232,7 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +4/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #33726 - [Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support

- Link: https://github.com/vllm-project/vllm/pull/33726
- Status/date: merged / 2026-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`, `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`; associated commits `f5972a872fa3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 19 files, +800/-158, 1473 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "[Model][Spec Decode] Nemotron-H MTP and Mamba Speculative Decoding Support"; the main implementation surface is `vllm/model_executor/models/nemotron_h_mtp.py`, `vllm/transformers_utils/configs/nemotron_h.py`, `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunks: -0,0 +1,503; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer, touching `NemotronHMTPAttentionDecoderLayer, __init__, forward`; `vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3 (9 lines); hunks: -51,6 +51,8 @@ class NemotronHConfig(PretrainedConfig):; -150,6 +152,7 @@ def __init__(; symbols: NemotronHConfig, __init__, touching `NemotronHConfig, __init__`; `vllm/model_executor/models/nemotron_h.py` modified +8/-0 (8 lines); hunks: -636,6 +636,9 @@ def forward(; -702,6 +705,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, is_spec_layer, _get_max_n_routed_experts, load_weights, touching `forward, is_spec_layer, _get_max_n_routed_experts`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0 (503 lines); hunks: -0,0 +1,503; symbols: NemotronHMTPAttentionDecoderLayer, __init__, forward, NemotronHMTPMoEDecoderLayer
  - `vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3 (9 lines); hunks: -51,6 +51,8 @@ class NemotronHConfig(PretrainedConfig):; -150,6 +152,7 @@ def __init__(; symbols: NemotronHConfig, __init__
  - `vllm/model_executor/models/nemotron_h.py` modified +8/-0 (8 lines); hunks: -636,6 +636,9 @@ def forward(; -702,6 +705,10 @@ def load_weights(self, weights: Iterable[tuple[str, torch.T...; symbols: forward, is_spec_layer, _get_max_n_routed_experts, load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h_mtp.py` added +503/-0; `vllm/transformers_utils/configs/nemotron_h.py` modified +6/-3; `vllm/model_executor/models/nemotron_h.py` modified +8/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/v1/attention/test_mamba_update_block_table.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35297 - [Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model

- Link: https://github.com/vllm-project/vllm/pull/35297
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `examples/pooling/embed/template/nemotron_embed_vl.jinja`, `vllm/model_executor/models/nemotron_vl.py`; associated commits `111d86906999`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +545/-31, 752 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/embed/template/nemotron_embed_vl.jinja`; technical summary: Covers "[Model] Add nvidia/llama-nemotron-embed-vl-1b-v2 multimodal embedding model"; the main implementation surface is `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/embed/template/nemotron_embed_vl.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_vl.py` modified +271/-31 (302 lines); hunks: -18,6 +18,7; -30,24 +31,28; symbols: build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor, __init__, touching `build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor`; `examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0 (20 lines); hunks: -0,0 +1,20.
- Code diff details:
  - `vllm/model_executor/models/nemotron_vl.py` modified +271/-31 (302 lines); hunks: -18,6 +18,7; -30,24 +31,28; symbols: build_transform, image_to_pixel_values_nemotron_vl, NemotronVLProcessor, __init__
  - `examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0 (20 lines); hunks: -0,0 +1,20
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +271/-31
  - docs: `examples/pooling/embed/template/nemotron_embed_vl.jinja` added +20/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_llama_nemotron_vl_embed.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35396 - Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models

- Link: https://github.com/vllm-project/vllm/pull/35396
- Status/date: merged / 2026-02-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h.py`; associated commits `832a780f3aed`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-0, 12 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models"; model line: Nemotron Super; category: docs/tests/CI; main diff: `vllm/model_executor/models/nemotron_h.py`; technical summary: Covers "Nemotron: use per-layer config in NemotronHMLPDecoderLayer for heterogeneous models"; the main implementation surface is `vllm/model_executor/models/nemotron_h.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h.py` modified +5/-0 (5 lines); hunks: -298,6 +298,11 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h.py` modified +5/-0 (5 lines); hunks: -298,6 +298,11 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nemotron_h.py
@@ -298,6 +298,11 @@ def __init__(
+        # Get per-layer config for heterogeneous models if exist
+        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
+        layer_config = get_layer_config(layer_idx) if get_layer_config else config
+        config = layer_config
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h.py` modified +5/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_h.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35100 - Support parakeet as audio encoder for nemotron-nano-vl

- Link: https://github.com/vllm-project/vllm/pull/35100
- Status/date: merged / 2026-02-27
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `c8aca0c9e1b3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +448/-20, 678 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support parakeet as audio encoder for nemotron-nano-vl"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Support parakeet as audio encoder for nemotron-nano-vl"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20 (274 lines); hunks: -44,6 +44,7; -55,12 +56,14; symbols: NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video, _preprocess_audio, touching `NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20 (274 lines); hunks: -44,6 +44,7; -55,12 +56,14; symbols: NanoNemotronVLAudioFeatureInputs, __init__, _preprocess_video, _preprocess_audio
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +254/-20
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/parakeet.py`, `vllm/transformers_utils/configs/parakeet.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35735 - [Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2

- Link: https://github.com/vllm-project/vllm/pull/35735
- Status/date: merged / 2026-03-03
- Trace source: `git log --name-only -- <model-files>` found it through `examples/pooling/score/template/nemotron-vl-rerank.jinja`, `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`; associated commits `c8b678e53e37`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 9 files, +503/-149, 723 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2"; model line: Nemotron Super; category: docs/tests/CI; main diff: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/score/template/nemotron-vl-rerank.jinja`; technical summary: Covers "[Model] Add support for nvidia/llama-nemotron-rerank-vl-1b-v2"; the main implementation surface is `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `vllm/model_executor/models/nemotron_vl.py`, `examples/pooling/score/template/nemotron-vl-rerank.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0 (355 lines); hunks: -0,0 +1,355; symbols: _run_test, test_models_text, test_models_image, _pil_to_data_uri, touching `_run_test, test_models_text, test_models_image`; `vllm/model_executor/models/nemotron_vl.py` modified +57/-0 (57 lines); hunks: -7,6 +7,7; -18,6 +19,7; symbols: load_weights, LlamaNemotronVLForSequenceClassification, __init__, touching `load_weights, LlamaNemotronVLForSequenceClassification, __init__`; `examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0 (15 lines); hunks: -0,0 +1,15.
- Code diff details:
  - `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0 (355 lines); hunks: -0,0 +1,355; symbols: _run_test, test_models_text, test_models_image, _pil_to_data_uri
  - `vllm/model_executor/models/nemotron_vl.py` modified +57/-0 (57 lines); hunks: -7,6 +7,7; -18,6 +19,7; symbols: load_weights, LlamaNemotronVLForSequenceClassification, __init__
  - `examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0 (15 lines); hunks: -0,0 +1,15
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` added +355/-0
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +57/-0
  - docs: `examples/pooling/score/template/nemotron-vl-rerank.jinja` added +15/-0
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`, `tests/models/multimodal/pooling/test_llama_nemotron_vl_embed.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35539 - Support Audio Extraction from MP4 Video for Nemotron Nano VL

- Link: https://github.com/vllm-project/vllm/pull/35539
- Status/date: merged / 2026-03-04
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `5d199ac8f25a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +225/-1, 293 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support Audio Extraction from MP4 Video for Nemotron Nano VL"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Support Audio Extraction from MP4 Video for Nemotron Nano VL"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1 (130 lines); hunks: -59,18 +59,25; -1381,6 +1388,127 @@ class NanoNemotronVLMultiModalProcessor(; symbols: NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply, _get_mm_fields_config, touching `NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1 (130 lines); hunks: -59,18 +59,25; -1381,6 +1388,127 @@ class NanoNemotronVLMultiModalProcessor(; symbols: NanoNemotronVLMultiModalProcessor, _extract_audio_from_videos, apply, _get_mm_fields_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +129/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/config.py`, `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/multimodal/media/audio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36192 - [Security] Respect user trust_remote_code setting in NemotronVL and KimiK25

- Link: https://github.com/vllm-project/vllm/pull/36192
- Status/date: merged / 2026-03-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_vl.py`; associated commits `00bd08edeee5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +7/-2, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nemotron_vl.py`; technical summary: Covers "[Security] Respect user trust_remote_code setting in NemotronVL and KimiK25"; the main implementation surface is `vllm/model_executor/models/nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_vl.py` modified +5/-1 (6 lines); hunks: -402,6 +402,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -456,7 +457,10 @@ def _init_vision_model(; symbols: __init__, _init_vision_model, _init_mlp1, touching `__init__, _init_vision_model, _init_mlp1`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_vl.py` modified +5/-1 (6 lines); hunks: -402,6 +402,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -456,7 +457,10 @@ def _init_vision_model(; symbols: __init__, _init_vision_model, _init_mlp1
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_vl.py` modified +5/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/kimi_k25.py`, `vllm/model_executor/models/nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35657 - [Model] Nano Nemotron VL - fast media preprocessing

- Link: https://github.com/vllm-project/vllm/pull/35657
- Status/date: merged / 2026-03-08
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `b7332b058c3b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +80/-61, 217 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Nano Nemotron VL - fast media preprocessing"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "[Model] Nano Nemotron VL - fast media preprocessing"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61 (141 lines); hunks: -17,11 +17,11; -214,7 +214,12 @@ class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):; symbols: NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values, video_to_pixel_values, touching `NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61 (141 lines); hunks: -17,11 +17,11; -214,7 +214,12 @@ class NanoNemotronVLVideoEmbeddingInputs(TensorSchema):; symbols: NanoNemotronVLVideoEmbeddingInputs, dynamic_preprocess, image_to_pixel_values, video_to_pixel_values
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +80/-61
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36393 - add nemotron v3 reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/36393
- Status/date: merged / 2026-03-09
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`; associated commits `203a7f27dac2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +186/-0, 195 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "add nemotron v3 reasoning parser"; model line: Nemotron Super; category: docs/tests/CI; main diff: `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`; technical summary: Covers "add nemotron v3 reasoning parser"; the main implementation surface is `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: ReasoningCase, FakeNemotronTokenizer, __init__, get_vocab, touching `ReasoningCase, FakeNemotronTokenizer, __init__`; `vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: NemotronV3ReasoningParser, extract_reasoning, touching `NemotronV3ReasoningParser, extract_reasoning`.
- Code diff details:
  - `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0 (150 lines); hunks: -0,0 +1,150; symbols: ReasoningCase, FakeNemotronTokenizer, __init__, get_vocab
  - `vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0 (32 lines); hunks: -0,0 +1,32; symbols: NemotronV3ReasoningParser, extract_reasoning
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` added +150/-0
  - runtime: `vllm/reasoning/nemotron_v3_reasoning_parser.py` added +32/-0
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_nemotron_v3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36635 - [NemotronH] Small fix reasoning parser

- Link: https://github.com/vllm-project/vllm/pull/36635
- Status/date: merged / 2026-03-11
- Trace source: `git log --name-only -- <model-files>` found it through `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`; associated commits `e661b9ee83d9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-1, 41 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NemotronH] Small fix reasoning parser"; model line: Nemotron Super; category: bug fix; main diff: `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`; technical summary: Covers "[NemotronH] Small fix reasoning parser"; the main implementation surface is `tests/reasoning/test_nemotron_v3_reasoning_parser.py`, `vllm/reasoning/nemotron_v3_reasoning_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0 (22 lines); hunks: -128,6 +128,28 @@ def test_nemotron_v3_without_thinking_returns_content(; symbols: test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning, touching `test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning`; `vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1 (5 lines); hunks: -24,7 +24,10 @@ def extract_reasoning(; symbols: extract_reasoning, touching `extract_reasoning`.
- Code diff details:
  - `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0 (22 lines); hunks: -128,6 +128,28 @@ def test_nemotron_v3_without_thinking_returns_content(; symbols: test_nemotron_v3_without_thinking_returns_content, test_nemotron_v3_force_nonempty_content_returns_content, test_nemotron_v3_with_thinking_keeps_truncated_reasoning
  - `vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1 (5 lines); hunks: -24,7 +24,10 @@ def extract_reasoning(; symbols: extract_reasoning
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/reasoning/test_nemotron_v3_reasoning_parser.py` modified +22/-0
  - runtime: `vllm/reasoning/nemotron_v3_reasoning_parser.py` modified +4/-1
- Risk and verification: The diff ships test coverage in `tests/reasoning/test_nemotron_v3_reasoning_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37456 - [Model] Remove unnecessary processor definition for Nemotron Parse

- Link: https://github.com/vllm-project/vllm/pull/37456
- Status/date: merged / 2026-03-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_parse.py`; associated commits `7476d148db99`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +0/-259, 288 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove unnecessary processor definition for Nemotron Parse"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/transformers_utils/processors/nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`; technical summary: Covers "[Model] Remove unnecessary processor definition for Nemotron Parse"; the main implementation surface is `vllm/transformers_utils/processors/nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: NemotronParseImageProcessor, __init__, _create_transforms, _resize_with_aspect_ratio, touching `NemotronParseImageProcessor, __init__, _create_transforms`; `vllm/model_executor/models/nemotron_parse.py` modified +0/-12 (12 lines); hunks: -55,7 +55,6; -367,17 +366,6 @@ class NemotronParsePixelInputs(TensorSchema):; symbols: NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config, get_hf_processor, touching `NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config`.
- Code diff details:
  - `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245 (245 lines); hunks: -1,245 +0,0; symbols: NemotronParseImageProcessor, __init__, _create_transforms, _resize_with_aspect_ratio
  - `vllm/model_executor/models/nemotron_parse.py` modified +0/-12 (12 lines); hunks: -55,7 +55,6; -367,17 +366,6 @@ class NemotronParsePixelInputs(TensorSchema):; symbols: NemotronParsePixelInputs, NemotronParseProcessingInfo, get_hf_config, get_hf_processor
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/nemotron_parse.py` removed +0/-245; `vllm/model_executor/models/nemotron_parse.py` modified +0/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nemotron_parse.py`, `vllm/transformers_utils/processors/__init__.py`, `vllm/transformers_utils/processors/nemotron_parse.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36808 - Support temporal compression for Nemotron-3-VL videos

- Link: https://github.com/vllm-project/vllm/pull/36808
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; associated commits `0b6d52629fe8`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +553/-130, 1189 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support temporal compression for Nemotron-3-VL videos"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Support temporal compression for Nemotron-3-VL videos"; the main implementation surface is `vllm/transformers_utils/processors/nano_nemotron_vl.py`, `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27 (232 lines); hunks: -11,6 +11,7; -43,6 +44,12; symbols: calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size, get_video_target_size_and_feature_size, touching `calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size`; `vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35 (210 lines); hunks: -8,6 +8,7; -77,6 +78,7; symbols: get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates, get_video_replacement_internvl, touching `get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates`.
- Code diff details:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27 (232 lines); hunks: -11,6 +11,7; -43,6 +44,12; symbols: calculate_timestamps, image_to_pixel_values, _compute_aspect_preserving_size, get_video_target_size_and_feature_size
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35 (210 lines); hunks: -8,6 +8,7; -77,6 +78,7; symbols: get_num_frames_with_most_features, get_hf_processor, _get_prompt_updates, get_video_replacement_internvl
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +205/-27; `vllm/model_executor/models/nano_nemotron_vl.py` modified +175/-35
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/model_executor/models/radio.py`, `vllm/transformers_utils/configs/radio.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37407 - [Bugfix] Fix Nemotron Parse loading

- Link: https://github.com/vllm-project/vllm/pull/37407
- Status/date: merged / 2026-03-19
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`; associated commits `765e4610651b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +49/-19, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Nemotron Parse loading"; model line: Nemotron Super; category: bug fix; main diff: `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`; technical summary: Covers "[Bugfix] Fix Nemotron Parse loading"; the main implementation surface is `tests/models/multimodal/generation/test_nemotron_parse.py`, `vllm/model_executor/models/nemotron_parse.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11 (55 lines); hunks: -1,21 +1,53; -44,6 +76,8 @@ def run_test(; symbols: DummyLogprobs, __init__, __repr__, mask_bbox_tokens, touching `DummyLogprobs, __init__, __repr__`; `vllm/model_executor/models/nemotron_parse.py` modified +3/-2 (5 lines); hunks: -319,8 +319,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11 (55 lines); hunks: -1,21 +1,53; -44,6 +76,8 @@ def run_test(; symbols: DummyLogprobs, __init__, __repr__, mask_bbox_tokens
  - `vllm/model_executor/models/nemotron_parse.py` modified +3/-2 (5 lines); hunks: -319,8 +319,9 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/generation/test_nemotron_parse.py` modified +44/-11
  - runtime: `vllm/model_executor/models/nemotron_parse.py` modified +3/-2
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_keye.py`, `tests/models/multimodal/generation/test_nemotron_parse.py`, `tests/models/test_terratorch.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37613 - [ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests

- Link: https://github.com/vllm-project/vllm/pull/37613
- Status/date: merged / 2026-03-20
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`; associated commits `fb4e8bf442c5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +8/-1, 40 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests"; model line: Nemotron Super; category: bug fix; main diff: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`; technical summary: Covers "[ROCm][CI] Fix accuracy for llama-nemotron-vl pooling tests"; the main implementation surface is `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1 (9 lines); hunks: -22,8 +22,10; -70,6 +72,7 @@ def _run_test(; symbols: _run_test, _run_vllm_reranker, _run_reranker_test, touching `_run_test, _run_vllm_reranker, _run_reranker_test`.
- Code diff details:
  - `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1 (9 lines); hunks: -22,8 +22,10; -70,6 +72,7 @@ def _run_test(; symbols: _run_test, _run_vllm_reranker, _run_reranker_test
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/models/multimodal/pooling/test_llama_nemotron_vl.py` modified +8/-1
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/pooling/test_llama_nemotron_vl.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37803 - Enable `NemotronHPuzzle` + `NemotronHMTP`

- Link: https://github.com/vllm-project/vllm/pull/37803
- Status/date: merged / 2026-03-22
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nemotron_h_mtp.py`; associated commits `e74c17e15331`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +6/-3, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable `NemotronHPuzzle` + `NemotronHMTP`"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/model_executor/models/nemotron_h_mtp.py`; technical summary: Covers "Enable `NemotronHPuzzle` + `NemotronHMTP`"; the main implementation surface is `vllm/model_executor/models/nemotron_h_mtp.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunks: -395,13 +395,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2 (7 lines); hunks: -395,13 +395,16 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nemotron_h_mtp.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/config/speculative.py`, `vllm/model_executor/models/nemotron_h_mtp.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36803 - [Test] E2E Nemotron-3-Super tests

- Link: https://github.com/vllm-project/vllm/pull/36803
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`; associated commits `56777b5c898d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +37/-0, 55 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] E2E Nemotron-3-Super tests"; model line: Nemotron Super; category: performance/backend optimization; main diff: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`; technical summary: Covers "[Test] E2E Nemotron-3-Super tests"; the main implementation surface is `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11.
- Code diff details:
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
  - `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0 (11 lines); hunks: -0,0 +1,11
- Key code excerpts:

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

- Reviewed files:
  - tests: `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml` added +11/-0; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml` added +11/-0; `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml` added +11/-0
- Risk and verification: The diff ships test coverage in `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-BF16.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-FP8.yaml`, `tests/evals/gsm8k/configs/Nemotron-3-Super-120B-A12B-NVFP4.yaml`, `tests/evals/gsm8k/configs/models-blackwell.txt`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #37903 - nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths

- Link: https://github.com/vllm-project/vllm/pull/37903
- Status/date: merged / 2026-03-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/processors/nano_nemotron_vl.py`; associated commits `a0d487b2e1d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +32/-44, 122 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`; technical summary: Covers "nano_nemotron_vl: suppress readonly torch.from_numpy() warning in image and video resize paths"; the main implementation surface is `vllm/transformers_utils/processors/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44 (76 lines); hunks: -8,6 +8,7; -66,6 +67,30 @@ def input_conditioner(x: torch.Tensor, norm_mean: torch.Tenso...; symbols: input_conditioner, _bicubic_from_ndarray, dynamic_preprocess, video_to_pixel_values, touching `input_conditioner, _bicubic_from_ndarray, dynamic_preprocess`.
- Code diff details:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44 (76 lines); hunks: -8,6 +8,7; -66,6 +67,30 @@ def input_conditioner(x: torch.Tensor, norm_mean: torch.Tenso...; symbols: input_conditioner, _bicubic_from_ndarray, dynamic_preprocess, video_to_pixel_values
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +32/-44
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/processors/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38018 - [Model] Use helper function to run MM processors with token inputs (where applicable)

- Link: https://github.com/vllm-project/vllm/pull/38018
- Status/date: merged / 2026-03-26
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `e812bf70bd66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +215/-145, 595 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Use helper function to run MM processors with token inputs (where applicable)"; model line: Nemotron Super; category: model support/runtime entry; main diff: `vllm/transformers_utils/processors/pixtral.py`, `vllm/transformers_utils/processors/voxtral.py`, `vllm/multimodal/processing/processor.py`; technical summary: Covers "[Model] Use helper function to run MM processors with token inputs (where applicable)"; the main implementation surface is `vllm/transformers_utils/processors/pixtral.py`, `vllm/transformers_utils/processors/voxtral.py`, `vllm/multimodal/processing/processor.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/pixtral.py` modified +9/-58 (67 lines); hunks: -5,10 +5,7; -55,62 +52,16 @@ class MistralCommonPixtralProcessor(ProcessorMixin):; symbols: MistralCommonPixtralProcessor, __init__, image_break_id, image_token_id, touching `MistralCommonPixtralProcessor, __init__, image_break_id`; `vllm/transformers_utils/processors/voxtral.py` modified +8/-54 (62 lines); hunks: -8,9 +8,6; -62,58 +59,15 @@ class MistralCommonVoxtralProcessor(ProcessorMixin):; symbols: MistralCommonVoxtralProcessor, __init__, audio_token_id, begin_audio_token_id, touching `MistralCommonVoxtralProcessor, __init__, audio_token_id`; `vllm/multimodal/processing/processor.py` modified +41/-10 (51 lines); hunks: -5,8 +5,15; -21,6 +28,7; symbols: _apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main, touching `_apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main`; `vllm/transformers_utils/processors/isaac.py` modified +32/-16 (48 lines); hunks: -1,16 +1,15; -308,15 +307,22 @@ def process_vision_for_patches(; symbols: process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs, IsaacProcessorKwargs, touching `process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs`.
- Code diff details:
  - `vllm/transformers_utils/processors/pixtral.py` modified +9/-58 (67 lines); hunks: -5,10 +5,7; -55,62 +52,16 @@ class MistralCommonPixtralProcessor(ProcessorMixin):; symbols: MistralCommonPixtralProcessor, __init__, image_break_id, image_token_id
  - `vllm/transformers_utils/processors/voxtral.py` modified +8/-54 (62 lines); hunks: -8,9 +8,6; -62,58 +59,15 @@ class MistralCommonVoxtralProcessor(ProcessorMixin):; symbols: MistralCommonVoxtralProcessor, __init__, audio_token_id, begin_audio_token_id
  - `vllm/multimodal/processing/processor.py` modified +41/-10 (51 lines); hunks: -5,8 +5,15; -21,6 +28,7; symbols: _apply_hf_processor_text_mm, _apply_hf_processor_mm_only, _apply_hf_processor_main
  - `vllm/transformers_utils/processors/isaac.py` modified +32/-16 (48 lines); hunks: -1,16 +1,15; -308,15 +307,22 @@ def process_vision_for_patches(; symbols: process_vision_for_patches, IsaacImageProcessorKwargs, IsaacImagesKwargs, IsaacProcessorKwargs
  - `vllm/model_executor/models/pixtral.py` modified +23/-1 (24 lines); hunks: -12,7 +12,7; -62,6 +62,7; symbols: _get_mm_fields_config, _call_hf_processor, _get_prompt_updates
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/pixtral.py` modified +9/-58; `vllm/transformers_utils/processors/voxtral.py` modified +8/-54; `vllm/multimodal/processing/processor.py` modified +41/-10; `vllm/transformers_utils/processors/isaac.py` modified +32/-16; `vllm/model_executor/models/pixtral.py` modified +23/-1; `vllm/model_executor/models/voxtral.py` modified +11/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/keye.py`, `vllm/model_executor/models/keye_vl1_5.py`, `vllm/model_executor/models/kimi_k25.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38567 - Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018

- Link: https://github.com/vllm-project/vllm/pull/38567
- Status/date: merged / 2026-03-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `e812bf70bd66`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +14/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Restore non-hf processor path for Nano-Nemotron-VL (bypass `call_hf_processor_mm_only`) - fixes #38018"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0 (14 lines); hunks: -307,6 +307,20 @@ def get_num_frames_with_most_features(; symbols: get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor, _get_image_fields_config, touching `get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0 (14 lines); hunks: -307,6 +307,20 @@ def get_num_frames_with_most_features(; symbols: get_num_frames_with_most_features, NanoNemotronVLMultiModalProcessor, _call_hf_processor, _get_image_fields_config
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +14/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38655 - Fix Nano Nemotron VL regressions

- Link: https://github.com/vllm-project/vllm/pull/38655
- Status/date: merged / 2026-04-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; associated commits `fa9e68022d29`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +84/-52, 331 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Nano Nemotron VL regressions"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; technical summary: Covers "Fix Nano Nemotron VL regressions"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23 (52 lines); hunks: -7,7 +7,6; -17,7 +16,7; symbols: get_hf_processor, is_dynamic_tiler, supports_video, supports_audio, touching `get_hf_processor, is_dynamic_tiler, supports_video`; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15 (32 lines); hunks: -356,15 +356,6 @@ def _images_to_pixel_values_lst(; -519,7 +510,6 @@ def compute_params(; symbols: _images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams, compute_params, touching `_images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23 (52 lines); hunks: -7,7 +7,6; -17,7 +16,7; symbols: get_hf_processor, is_dynamic_tiler, supports_video, supports_audio
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15 (32 lines); hunks: -356,15 +356,6 @@ def _images_to_pixel_values_lst(; -519,7 +510,6 @@ def compute_params(; symbols: _images_to_pixel_values_lst, get_cached_feature_size, DynamicResolutionParams, compute_params
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +29/-23; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +17/-15
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39029 - nano_nemotron_vl: fix tensor device mismatch exception when video profiling

- Link: https://github.com/vllm-project/vllm/pull/39029
- Status/date: merged / 2026-04-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `d56e95223917`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-2, 16 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nano_nemotron_vl: fix tensor device mismatch exception when video profiling"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "nano_nemotron_vl: fix tensor device mismatch exception when video profiling"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2 (5 lines); hunks: -1239,12 +1239,13 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings, touching `_create_final_video_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2 (5 lines); hunks: -1239,12 +1239,13 @@ def _create_final_video_embeddings(; symbols: _create_final_video_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1239,12 +1239,13 @@ def _create_final_video_embeddings(
+        device = video_embeddings.device
-        repl_token_ids = torch.tensor(video_repl.full)
+        repl_token_ids = torch.tensor(video_repl.full, device=device)
-        embed_token_ids = torch.tensor(self._img_context_token_ids)
+        embed_token_ids = torch.tensor(self._img_context_token_ids, device=device)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38727 - nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len

- Link: https://github.com/vllm-project/vllm/pull/38727
- Status/date: merged / 2026-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `a9a0e0551f03`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +52/-10, 84 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "nano-nemotron-vl: get_mm_max_tokens_per_item for audio, video, image == seq_len"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10 (62 lines); hunks: -288,6 +288,35 @@ def get_max_image_tokens(self) -> int:; -306,6 +335,26 @@ def get_num_frames_with_most_features(; symbols: get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features, get_mm_max_tokens_per_item, touching `get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10 (62 lines); hunks: -288,6 +288,35 @@ def get_max_image_tokens(self) -> int:; -306,6 +335,26 @@ def get_num_frames_with_most_features(; symbols: get_max_image_tokens, get_dummy_image_size_and_max_tokens, get_num_frames_with_most_features, get_mm_max_tokens_per_item
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +52/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #38538 - nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time

- Link: https://github.com/vllm-project/vllm/pull/38538
- Status/date: merged / 2026-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; associated commits `df2503e125f3`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +74/-18, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; technical summary: Covers "nemotron-nano-vl: Allow `use_audio_in_video` to be passed at `vllm serve` time"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18 (90 lines); hunks: -597,19 +597,26 @@ def _get_prompt_updates(; -618,7 +625,16 @@ def _extract_audio_from_videos(; symbols: _get_prompt_updates, _extract_audio_from_videos, apply, touching `_get_prompt_updates, _extract_audio_from_videos, apply`; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0 (2 lines); hunks: -771,6 +771,7 @@ def __init__(; -781,6 +782,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18 (90 lines); hunks: -597,19 +597,26 @@ def _get_prompt_updates(; -618,7 +625,16 @@ def _extract_audio_from_videos(; symbols: _get_prompt_updates, _extract_audio_from_videos, apply
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0 (2 lines); hunks: -771,6 +771,7 @@ def __init__(; -781,6 +782,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +72/-18; `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`, `vllm/transformers_utils/processors/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37580 - Nemotron Nano VL: Streamline pixel shuffle

- Link: https://github.com/vllm-project/vllm/pull/37580
- Status/date: merged / 2026-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `270e8a410254`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +11/-36, 73 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Nemotron Nano VL: Streamline pixel shuffle"; model line: Nemotron Super; category: model implementation change; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Nemotron Nano VL: Streamline pixel shuffle"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36 (47 lines); hunks: -1005,38 +1005,27 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1045,22 +1034,8 @@ def pixel_shuffle_dynamic_res(; symbols: __init__, pixel_shuffle, pixel_shuffle_dynamic_res, touching `__init__, pixel_shuffle, pixel_shuffle_dynamic_res`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36 (47 lines); hunks: -1005,38 +1005,27 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: s...; -1045,22 +1034,8 @@ def pixel_shuffle_dynamic_res(; symbols: __init__, pixel_shuffle, pixel_shuffle_dynamic_res
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +11/-36
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #39901 - FIX: support language_model.backbone naming in NemotronH Nano VL quantization config

- Link: https://github.com/vllm-project/vllm/pull/39901
- Status/date: merged / 2026-04-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `8b5531933a7b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "FIX: support language_model.backbone naming in NemotronH Nano VL quantization config"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "FIX: support language_model.backbone naming in NemotronH Nano VL quantization config"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -37,6 +37,7; -903,6 +904,12 @@ class NemotronH_Nano_VL_V2(; symbols: NemotronH_Nano_VL_V2, get_placeholder_str, touching `NemotronH_Nano_VL_V2, get_placeholder_str`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0 (7 lines); hunks: -37,6 +37,7; -903,6 +904,12 @@ class NemotronH_Nano_VL_V2(; symbols: NemotronH_Nano_VL_V2, get_placeholder_str
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40283 - Optimize nemotron VL image/video preprocessing

- Link: https://github.com/vllm-project/vllm/pull/40283
- Status/date: merged / 2026-04-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/transformers_utils/processors/nano_nemotron_vl.py`; associated commits `982beae809b1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +116/-98, 384 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimize nemotron VL image/video preprocessing"; model line: Nemotron Super; category: performance/backend optimization; main diff: `vllm/transformers_utils/processors/nano_nemotron_vl.py`; technical summary: Covers "Optimize nemotron VL image/video preprocessing"; the main implementation surface is `vllm/transformers_utils/processors/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98 (214 lines); hunks: -8,7 +8,6; -26,7 +25,7; symbols: calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize, _bicubic_from_ndarray, touching `calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize`.
- Code diff details:
  - `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98 (214 lines); hunks: -8,7 +8,6; -26,7 +25,7; symbols: calculate_timestamps, input_conditioner, _bicubic_resize_and_normalize, _bicubic_from_ndarray
- Key code excerpts:

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

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/nano_nemotron_vl.py` modified +116/-98
- Risk and verification: Runtime changes concentrate in `vllm/transformers_utils/processors/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40724 - Fix Nano Nemotron VL static image inputs

- Link: https://github.com/vllm-project/vllm/pull/40724
- Status/date: merged / 2026-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/nano_nemotron_vl.py`; associated commits `9ad5abe7722b`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Nano Nemotron VL static image inputs"; model line: Nemotron Super; category: bug fix; main diff: `vllm/model_executor/models/nano_nemotron_vl.py`; technical summary: Covers "Fix Nano Nemotron VL static image inputs"; the main implementation surface is `vllm/model_executor/models/nano_nemotron_vl.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1124,7 +1124,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic, touching `_parse_and_validate_image_input, _process_image_input_dynamic`.
- Code diff details:
  - `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1 (4 lines); hunks: -1124,7 +1124,9 @@ def _parse_and_validate_image_input(; symbols: _parse_and_validate_image_input, _process_image_input_dynamic
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/nano_nemotron_vl.py
@@ -1124,7 +1124,9 @@ def _parse_and_validate_image_input(
-                num_patches=kwargs.pop("image_num_patches"), **kwargs
+                pixel_values_flat=pixel_values_flat,
+                num_patches=kwargs.pop("image_num_patches"),
+                **kwargs,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/nano_nemotron_vl.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/nano_nemotron_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
