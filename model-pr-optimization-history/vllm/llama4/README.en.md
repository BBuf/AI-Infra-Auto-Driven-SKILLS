# vllm Llama 4 Model PR Optimization History

## Implementation File Coverage

| File | Git-traced PRs |
| --- | --- |
| `examples/tool_chat_template_llama4_json.jinja` | [#16428](https://github.com/vllm-project/vllm/pull/16428) |
| `examples/tool_chat_template_llama4_pythonic.jinja` | [#16463](https://github.com/vllm-project/vllm/pull/16463), [#17917](https://github.com/vllm-project/vllm/pull/17917) |
| `tests/models/multimodal/processing/test_llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113) |
| `tests/models/multimodal/processing/test_mllama4.py` | no direct PR-number commit |
| `tests/tool_parsers/test_llama4_pythonic_tool_parser.py` | no direct PR-number commit |
| `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` | [#25145](https://github.com/vllm-project/vllm/pull/25145), [#25889](https://github.com/vllm-project/vllm/pull/25889), [#26790](https://github.com/vllm-project/vllm/pull/26790), [#30709](https://github.com/vllm-project/vllm/pull/30709) |
| `vllm/model_executor/models/llama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16311](https://github.com/vllm-project/vllm/pull/16311), [#16439](https://github.com/vllm-project/vllm/pull/16439), [#16512](https://github.com/vllm-project/vllm/pull/16512), [#16801](https://github.com/vllm-project/vllm/pull/16801), [#17315](https://github.com/vllm-project/vllm/pull/17315), [#19997](https://github.com/vllm-project/vllm/pull/19997), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#21499](https://github.com/vllm-project/vllm/pull/21499), [#22691](https://github.com/vllm-project/vllm/pull/22691), [#22701](https://github.com/vllm-project/vllm/pull/22701), ... (20 total) |
| `vllm/model_executor/models/llama4_eagle.py` | [#20591](https://github.com/vllm-project/vllm/pull/20591), [#20788](https://github.com/vllm-project/vllm/pull/20788), [#27136](https://github.com/vllm-project/vllm/pull/27136), [#29926](https://github.com/vllm-project/vllm/pull/29926) |
| `vllm/model_executor/models/mllama4.py` | [#16113](https://github.com/vllm-project/vllm/pull/16113), [#16201](https://github.com/vllm-project/vllm/pull/16201), [#16365](https://github.com/vllm-project/vllm/pull/16365), [#16746](https://github.com/vllm-project/vllm/pull/16746), [#18368](https://github.com/vllm-project/vllm/pull/18368), [#20419](https://github.com/vllm-project/vllm/pull/20419), [#22021](https://github.com/vllm-project/vllm/pull/22021), [#22107](https://github.com/vllm-project/vllm/pull/22107), [#25961](https://github.com/vllm-project/vllm/pull/25961), [#28602](https://github.com/vllm-project/vllm/pull/28602), [#30709](https://github.com/vllm-project/vllm/pull/30709), [#35147](https://github.com/vllm-project/vllm/pull/35147), ... (15 total) |
| `vllm/tool_parsers/llama4_pythonic_tool_parser.py` | no direct PR-number commit |

## PR Coverage Summary

- Git-traced PRs: 40
- Extra PRs preserved from existing docs: 20
- Total PRs in this document: 60
- File trace command: `git log --name-only -- <model-files>`
- Diff audit source: GitHub Pull Request files API

## Timeline

| Date | PR | State | Title | Main files |
| --- | --- | --- | --- | --- |
| 2025-04-06 | [#16104](https://github.com/vllm-project/vllm/pull/16104) | merged | [Model] Support Llama4 in vLLM | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` |
| 2025-04-07 | [#16113](https://github.com/vllm-project/vllm/pull/16113) | merged | Upstream Llama4 Support to Main | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py` |
| 2025-04-07 | [#16201](https://github.com/vllm-project/vllm/pull/16201) | merged | [Misc] Move Llama 4 projector call into encoder execution | `vllm/model_executor/models/mllama4.py` |
| 2025-04-09 | [#16311](https://github.com/vllm-project/vllm/pull/16311) | merged | [BugFix] llama4 qknorm should be not shared across head | `vllm/model_executor/models/llama4.py` |
| 2025-04-10 | [#16365](https://github.com/vllm-project/vllm/pull/16365) | merged | [Model] Remove image mm limit for LLaMa4 | `vllm/model_executor/models/mllama4.py` |
| 2025-04-11 | [#16439](https://github.com/vllm-project/vllm/pull/16439) | merged | [Llama4] Enable attention temperature tuning by default for long context (>32k) | `vllm/model_executor/models/llama4.py` |
| 2025-04-11 | [#16463](https://github.com/vllm-project/vllm/pull/16463) | merged | [Frontend] Added chat templates for LLaMa4 pythonic tool calling | `examples/tool_chat_template_llama4_pythonic.jinja` |
| 2025-04-12 | [#16512](https://github.com/vllm-project/vllm/pull/16512) | merged | Optimized topk for topk=1 (Llama-4) | `vllm/model_executor/models/llama4.py` |
| 2025-04-18 | [#16801](https://github.com/vllm-project/vllm/pull/16801) | merged | [BugFix] Accuracy fix for llama4 int4 - improperly casted scales | `vllm/model_executor/models/llama4.py` |
| 2025-04-18 | [#16746](https://github.com/vllm-project/vllm/pull/16746) | merged | [Bugfix] fix pp for llama4 | `vllm/model_executor/models/mllama4.py` |
| 2025-04-24 | [#16428](https://github.com/vllm-project/vllm/pull/16428) | merged | Add chat template for Llama 4 models | `examples/tool_chat_template_llama4_json.jinja` |
| 2025-04-29 | [#17315](https://github.com/vllm-project/vllm/pull/17315) | merged | [model] make llama4 compatible with pure dense layers | `vllm/model_executor/models/llama4.py` |
| 2025-05-22 | [#17917](https://github.com/vllm-project/vllm/pull/17917) | merged | [Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser | `examples/tool_chat_template_llama4_pythonic.jinja` |
| 2025-06-02 | [#18368](https://github.com/vllm-project/vllm/pull/18368) | merged | [Model] enable data parallel for Llama4 vision encoder | `vllm/model_executor/models/mllama4.py` |
| 2025-06-25 | [#19997](https://github.com/vllm-project/vllm/pull/19997) | merged | [Llama4] Update `attn_temperature_tuning` | `vllm/model_executor/models/llama4.py` |
| 2025-07-12 | [#20419](https://github.com/vllm-project/vllm/pull/20419) | merged | Enable ModelOpt Llama4 fp8 checkpoint deployment | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2025-07-16 | [#20591](https://github.com/vllm-project/vllm/pull/20591) | merged | [Meta] Llama4 EAGLE Support | `vllm/model_executor/models/llama4_eagle.py` |
| 2025-07-30 | [#21499](https://github.com/vllm-project/vllm/pull/21499) | merged | [NVIDIA] Fix Llama4 Scout FP4 functionality issues | `vllm/model_executor/models/llama4.py` |
| 2025-07-31 | [#20788](https://github.com/vllm-project/vllm/pull/20788) | merged | [Meta] Official Eagle mm support, first enablement on llama4 | `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py` |
| 2025-08-03 | [#22107](https://github.com/vllm-project/vllm/pull/22107) | merged | [Fix] Fix llama4 modelopt weight loading error | `vllm/model_executor/models/mllama4.py` |
| 2025-08-12 | [#22511](https://github.com/vllm-project/vllm/pull/22511) | merged | Fix Llama4 FlashInfer FP4 MoE issues | `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` |
| 2025-08-13 | [#22701](https://github.com/vllm-project/vllm/pull/22701) | merged | Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op | `vllm/model_executor/models/llama4.py` |
| 2025-08-19 | [#22691](https://github.com/vllm-project/vllm/pull/22691) | merged | [bug fix] Fix llama4 spec decoding | `vllm/model_executor/models/llama4.py` |
| 2025-08-28 | [#22021](https://github.com/vllm-project/vllm/pull/22021) | merged | Migrate Llama4ImagePatchInputs to TensorSchema | `vllm/model_executor/models/mllama4.py` |
| 2025-09-11 | [#24444](https://github.com/vllm-project/vllm/pull/24444) | merged | [Bugfix] Fix platform-specific routing in CustomOp implementations | `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` |
| 2025-09-30 | [#25889](https://github.com/vllm-project/vllm/pull/25889) | merged | [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding` | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-10-06 | [#25961](https://github.com/vllm-project/vllm/pull/25961) | merged | Support llama3 eagle3 head with llama4 verifier | `vllm/model_executor/models/mllama4.py` |
| 2025-10-14 | [#26790](https://github.com/vllm-project/vllm/pull/26790) | merged | llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-10-21 | [#27136](https://github.com/vllm-project/vllm/pull/27136) | merged | [Fix][Spec Decode] Fix llama4 draft loading with different quantization | `vllm/model_executor/models/llama4_eagle.py` |
| 2025-10-30 | [#25145](https://github.com/vllm-project/vllm/pull/25145) | merged | [XPU][bugfix] fix rope for llama4 and deepseek | `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2025-11-14 | [#28602](https://github.com/vllm-project/vllm/pull/28602) | merged | LLaMA4 LoRA Adapter Enablement | `vllm/model_executor/models/mllama4.py` |
| 2025-11-20 | [#28577](https://github.com/vllm-project/vllm/pull/28577) | merged | [BugFix] Fix Llama4 Pipeline Parallelism Assert Error | `vllm/model_executor/models/llama4.py` |
| 2025-12-05 | [#29926](https://github.com/vllm-project/vllm/pull/29926) | merged | [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute | `vllm/model_executor/models/llama4_eagle.py` |
| 2026-01-10 | [#30709](https://github.com/vllm-project/vllm/pull/30709) | merged | [Misc][LLaMa4] Compile LLaMa Vision Encoder | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` |
| 2026-01-23 | [#32886](https://github.com/vllm-project/vllm/pull/32886) | merged | [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4 | `vllm/model_executor/models/llama4.py` |
| 2026-02-11 | [#34243](https://github.com/vllm-project/vllm/pull/34243) | merged | [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8) | `vllm/model_executor/models/llama4.py` |
| 2026-02-19 | [#34471](https://github.com/vllm-project/vllm/pull/34471) | merged | [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers | `vllm/model_executor/models/llama4.py` |
| 2026-02-21 | [#34997](https://github.com/vllm-project/vllm/pull/34997) | merged | Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers " | `vllm/model_executor/models/llama4.py` |
| 2026-02-23 | [#35033](https://github.com/vllm-project/vllm/pull/35033) | merged | [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests | `vllm/model_executor/models/llama4.py` |
| 2026-02-24 | [#35147](https://github.com/vllm-project/vllm/pull/35147) | merged | [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4) | `vllm/model_executor/models/mllama4.py` |
| 2026-03-09 | [#36436](https://github.com/vllm-project/vllm/pull/36436) | merged | [Misc] Refactored 5 duplicate helper functions that were copied-pasted across multiple parsers | `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/olmo3_tool_parser.py`, `vllm/tool_parsers/pythonic_tool_parser.py` |
| 2026-03-09 | [#36281](https://github.com/vllm-project/vllm/pull/36281) | merged | [BE] Rename `should_torch_compile_mm_vit` to `should_torch_compile_mm_encoder` | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `vllm/model_executor/models/mllama4.py` |
| 2026-03-11 | [#36770](https://github.com/vllm-project/vllm/pull/36770) | merged | [Misc] Clean up renderers | `vllm/transformers_utils/processors/kimi_audio.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/kimi_audio.py` |
| 2026-03-13 | [#36063](https://github.com/vllm-project/vllm/pull/36063) | merged | [Refactor] Consolidate SupportsEagle | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/hunyuan_v1.py` |
| 2026-03-16 | [#36288](https://github.com/vllm-project/vllm/pull/36288) | merged | [torch.compile][BE] Modify cudagraph callable to check for is_forward_context_set | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/mllama4.py`, `docs/design/torch_compile_multimodal.md` |
| 2026-03-19 | [#37345](https://github.com/vllm-project/vllm/pull/37345) | merged | [torch.compile][BE][Multimodal] Remove requirement to set_model_tag to avoid cache conflict | `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `docs/design/torch_compile_multimodal.md` |
| 2026-03-23 | [#37834](https://github.com/vllm-project/vllm/pull/37834) | merged | [Test] Consolidate tool parser unit tests to tests/tool_parsers | `tests/tool_parsers/test_hermes_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py`, `tests/tool_parsers/test_granite4_tool_parser.py` |
| 2026-03-25 | [#35182](https://github.com/vllm-project/vllm/pull/35182) | merged | [Misc] Reorganize inputs | `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py` |
| 2026-03-26 | [#38029](https://github.com/vllm-project/vllm/pull/38029) | merged | [Tool Parser][1/3] Pass tools to ToolParser constructor | `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py` |
| 2026-04-20 | [#35949](https://github.com/vllm-project/vllm/pull/35949) | merged | [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase | `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py` |
| 2026-04-21 | [#35782](https://github.com/vllm-project/vllm/pull/35782) | merged | [MoE Refactor] Remove SharedFusedMoE class | `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py` |
| 2026-04-23 | [#40671](https://github.com/vllm-project/vllm/pull/40671) | merged | [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping | `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py` |
| 2026-05-11 | [#42280](https://github.com/vllm-project/vllm/pull/42280) | merged | [Model] Fix missing `maybe_prefix` | `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py` |
| 2026-06-05 | [#43167](https://github.com/vllm-project/vllm/pull/43167) | merged | Remove KV cache scale boilerplate from model weight loading methods | `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py` |
| 2026-06-10 | [#39419](https://github.com/vllm-project/vllm/pull/39419) | merged | [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding | `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py` |
| 2026-06-10 | [#45047](https://github.com/vllm-project/vllm/pull/45047) | merged | [Bugfix] Fix Llama4 weight loading | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2026-06-12 | [#40660](https://github.com/vllm-project/vllm/pull/40660) | merged | [MM][Perf][CG] Support ViT full cudagraphs for mllama4 | `vllm/model_executor/models/mllama4.py` |
| 2026-06-12 | [#39612](https://github.com/vllm-project/vllm/pull/39612) | merged | [Migration] Migrate GGUF quantization support to plugin | `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`, `tests/models/test_gguf_download.py` |
| 2026-06-15 | [#44645](https://github.com/vllm-project/vllm/pull/44645) | merged | [Bugfix] Stream Llama4 weight loading to avoid host-OOM with copy-returning loaders | `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py` |
| 2026-06-16 | [#43586](https://github.com/vllm-project/vllm/pull/43586) | merged | [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR | `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py` |

## Per-PR Diff Audit Cards

### PR #16104 - [Model] Support Llama4 in vLLM

- Link: https://github.com/vllm-project/vllm/pull/16104
- Status/date: merged / 2025-04-06
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 35 files, +2369/-142, 3141 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Support Llama4 in vLLM"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json`; technical summary: Covers "[Model] Support Llama4 in vLLM"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`; `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward, touching `Llama4MoE, custom_routing_function, __init__`; `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200; `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` added +886/-0 (886 lines); hunks: -0,0 +1,886; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +530/-0 (530 lines); hunks: -0,0 +1,530; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0 (200 lines); hunks: -0,0 +1,200
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
  - `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0 (68 lines); hunks: -851,6 +851,70 @@ def _compute_inv_freq(self, base: Union[int, float]) -> tor...; -1130,6 +1194,10 @@ def get_rope(; symbols: _compute_inv_freq, Llama4VisionRotaryEmbedding, __init__, _compute_cos_sin_cache
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -0,0 +1,886 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/models/llama4.py
@@ -0,0 +1,530 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json
@@ -0,0 +1,200 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` added +886/-0; `vllm/model_executor/models/llama4.py` added +530/-0; `vllm/model_executor/layers/fused_moe/configs/E=16,N=1024,device_name=AMD_Instinct_MI300X.json` added +200/-0; `vllm/model_executor/layers/rotary_embedding.py` modified +68/-0; `vllm/model_executor/layers/fused_moe/layer.py` modified +41/-24; `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` modified +24/-14
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/multimodal/processing/test_common.py`, `tests/models/multimodal/processing/test_llama4.py`, `tests/models/registry.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16113 - Upstream Llama4 Support to Main

- Link: https://github.com/vllm-project/vllm/pull/16113
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `tests/models/multimodal/processing/test_llama4.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `55dcce91df15`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 43 files, +2436/-155, 3350 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Upstream Llama4 Support to Main"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py`; technical summary: Covers "Upstream Llama4 Support to Main"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`, `tests/models/multimodal/processing/test_llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`; `vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward, touching `Llama4MoE, custom_routing_function, __init__`; `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override, touching `test_processor_override`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` added +895/-0 (895 lines); hunks: -0,0 +1,895; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, forward
  - `vllm/model_executor/models/llama4.py` added +531/-0 (531 lines); hunks: -0,0 +1,531; symbols: Llama4MoE, custom_routing_function, __init__, forward
  - `tests/models/multimodal/processing/test_llama4.py` added +99/-0 (99 lines); hunks: -0,0 +1,99; symbols: test_processor_override
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -0,0 +1,895 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- vllm/model_executor/models/llama4.py
@@ -0,0 +1,531 @@
+# SPDX-License-Identifier: Apache-2.0
+#
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
diff -- tests/models/multimodal/processing/test_llama4.py
@@ -0,0 +1,99 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` added +895/-0; `vllm/model_executor/models/llama4.py` added +531/-0
  - tests: `tests/models/multimodal/processing/test_llama4.py` added +99/-0
- Risk and verification: The diff ships test coverage in `tests/models/decoder_only/audio_language/test_ultravox.py`, `tests/models/decoder_only/vision_language/test_models.py`, `tests/models/decoder_only/vision_language/test_phi3v.py`, `tests/models/decoder_only/vision_language/test_pixtral.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16201 - [Misc] Move Llama 4 projector call into encoder execution

- Link: https://github.com/vllm-project/vllm/pull/16201
- Status/date: merged / 2025-04-07
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `ed636d99caa0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-3, 22 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Move Llama 4 projector call into encoder execution"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Misc] Move Llama 4 projector call into encoder execution"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings, touching `_process_image_input, get_multimodal_embeddings, get_input_embeddings`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +4/-3 (7 lines); hunks: -760,6 +760,8 @@ def _process_image_input(; -791,10 +793,9 @@ def get_input_embeddings(; symbols: _process_image_input, get_multimodal_embeddings, get_input_embeddings
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -760,6 +760,8 @@ def _process_image_input(
+        vision_embeddings_flat = self.multi_modal_projector(
+            vision_embeddings_flat)
@@ -791,10 +793,9 @@ def get_input_embeddings(
-            multimodal_embeddings = torch.cat(multimodal_embeddings)
-            mm_embeddings = self.multi_modal_projector(multimodal_embeddings)
-                input_ids, inputs_embeds, select_patch_features(mm_embeddings),
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +4/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16311 - [BugFix] llama4 qknorm should be not shared across head

- Link: https://github.com/vllm-project/vllm/pull/16311
- Status/date: merged / 2025-04-09
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ec7da6fcf32f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-12, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] llama4 qknorm should be not shared across head"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[BugFix] llama4 qknorm should be not shared across head"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward, touching `__init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +7/-12 (19 lines); hunks: -155,14 +155,8 @@ def __init__(self,; -226,10 +220,11 @@ def forward(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -155,14 +155,8 @@ def __init__(self,
-        self.q_norm = RMSNorm(
-            hidden_size=self.q_size,
-            eps=config.rms_norm_eps,
-            has_weight=False,
-            dtype=torch.float32,
-        ) if self.use_qk_norm else None
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-12
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16365 - [Model] Remove image mm limit for LLaMa4

- Link: https://github.com/vllm-project/vllm/pull/16365
- Status/date: merged / 2025-04-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `61de3ef74b9c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +26/-7, 84 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Remove image mm limit for LLaMa4"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Model] Remove image mm limit for LLaMa4"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk, touching `get_hf_processor, get_supported_mm_limits, get_patch_per_chunk`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +3/-1 (4 lines); hunks: -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Proces...; symbols: get_hf_processor, get_supported_mm_limits, get_patch_per_chunk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -477,7 +477,9 @@ def get_hf_processor(self, **kwargs: object) -> Llama4Processor:
-        return {"image": 10}
+        # Although vLLM can support more images from an infra capability
+        # perspective, we do not recommend using >10 images in practice.
+        return {"image": None}
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16439 - [Llama4] Enable attention temperature tuning by default for long context (>32k)

- Link: https://github.com/vllm-project/vllm/pull/16439
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `99ef59cf7f93`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, 18 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] Enable attention temperature tuning by default for long context (>32k)"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Llama4] Enable attention temperature tuning by default for long context (>32k)"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__, touching `Llama4ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):; symbols: Llama4ForCausalLM, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -467,11 +467,15 @@ class Llama4ForCausalLM(LlamaForCausalLM):
-        # Update temperature tuning config from generation config
+        # update temperature tuning config from generation config
+        # enable temperature tuning by default when max_model_len > 32K
+        default_attn_temperature_tuning = \
+            vllm_config.model_config.max_model_len > 32768
-            = gen_config.get("attn_temperature_tuning", False)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16463 - [Frontend] Added chat templates for LLaMa4 pythonic tool calling

- Link: https://github.com/vllm-project/vllm/pull/16463
- Status/date: merged / 2025-04-11
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_pythonic.jinja`; associated commits `16eda8c43a9d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +182/-2, 223 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend] Added chat templates for LLaMa4 pythonic tool calling"; model line: Llama 4; category: model implementation change; main diff: `examples/tool_chat_template_llama4_pythonic.jinja`; technical summary: Covers "[Frontend] Added chat templates for LLaMa4 pythonic tool calling"; the main implementation surface is `examples/tool_chat_template_llama4_pythonic.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139.
- Code diff details:
  - `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0 (139 lines); hunks: -0,0 +1,139
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_llama4_pythonic.jinja
@@ -0,0 +1,139 @@
+{{- bos_token }}
+{%- if custom_tools is defined %}
+    {%- set tools = custom_tools %}
+{%- endif %}
+{%- if not tools_in_user_message is defined %}
+    {%- set tools_in_user_message = false %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` added +139/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/conftest.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #16512 - Optimized topk for topk=1 (Llama-4)

- Link: https://github.com/vllm-project/vllm/pull/16512
- Status/date: merged / 2025-04-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `bd6028d6b0bb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +11/-2, 31 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Optimized topk for topk=1 (Llama-4)"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "Optimized topk for topk=1 (Llama-4)"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function, touching `custom_routing_function`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -37,7 +37,7; -50,7 +50,7 @@ def custom_routing_function(; symbols: custom_routing_function
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -37,7 +37,7 @@
-from .utils import (AutoWeightsLoader, extract_layer_index,
+from .utils import (AutoWeightsLoader, extract_layer_index, fast_topk,
@@ -50,7 +50,7 @@ def custom_routing_function(
-        router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
+        router_scores, router_indices = fast_topk(gating_output, topk, dim=-1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16801 - [BugFix] Accuracy fix for llama4 int4 - improperly casted scales

- Link: https://github.com/vllm-project/vllm/pull/16801
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `7eb42556281d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +6/-9, 58 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Accuracy fix for llama4 int4 - improperly casted scales"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[BugFix] Accuracy fix for llama4 int4 - improperly casted scales"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__, touching `custom_routing_function, __init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -51,8 +51,8 @@ def custom_routing_function(; symbols: custom_routing_function, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -51,8 +51,8 @@ def custom_routing_function(
-        router_scores = torch.sigmoid(router_scores.float()).to(
-            hidden_states.dtype)
+        # psuedo-standard is that the router scores are floats
+        router_scores = torch.sigmoid(router_scores.float())
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16746 - [Bugfix] fix pp for llama4

- Link: https://github.com/vllm-project/vllm/pull/16746
- Status/date: merged / 2025-04-18
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `e31045f95ca0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-3, 21 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] fix pp for llama4"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Bugfix] fix pp for llama4"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights, touching `__init__, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +3/-3 (6 lines); hunks: -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,; symbols: __init__, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -672,9 +672,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config.with_hf_config(config.text_config),
+            vllm_config=vllm_config.with_hf_config(config.text_config,
+                                                   ["LlamaForCausalLM"]),
@@ -824,7 +824,7 @@ def load_weights(self, weights: Iterable[Tuple[str,
-            weights, prefix="language_model.model.")
+            weights, prefix="language_model.")
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +3/-3
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #16428 - Add chat template for Llama 4 models

- Link: https://github.com/vllm-project/vllm/pull/16428
- Status/date: merged / 2025-04-24
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_json.jinja`; associated commits `05e1fbfc52ca`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +139/-1, 172 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Add chat template for Llama 4 models"; model line: Llama 4; category: model support/runtime entry; main diff: `examples/tool_chat_template_llama4_json.jinja`; technical summary: Covers "Add chat template for Llama 4 models"; the main implementation surface is `examples/tool_chat_template_llama4_json.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116.
- Code diff details:
  - `examples/tool_chat_template_llama4_json.jinja` added +116/-0 (116 lines); hunks: -0,0 +1,116
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_llama4_json.jinja
@@ -0,0 +1,116 @@
+{%- macro is_array_of_type_objects(var) -%}
+    {%- if var is iterable and var is not string -%}
+        {%- set valid = true -%}
+        {%- for item in var -%}
+            {%- if 'type' not in item -%}
+                {%- set valid = false -%}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_json.jinja` added +116/-0
- Risk and verification: The diff ships test coverage in `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #17315 - [model] make llama4 compatible with pure dense layers

- Link: https://github.com/vllm-project/vllm/pull/17315
- Status/date: merged / 2025-04-29
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `b4ac4fa04da1`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +2/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[model] make llama4 compatible with pure dense layers"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[model] make llama4 compatible with pure dense layers"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +2/-2 (4 lines); hunks: -273,8 +273,8 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -273,8 +273,8 @@ def __init__(
-        is_moe_layer = (self.layer_idx +
-                        1) % config.interleave_moe_layer_step == 0
+        is_moe_layer = config.interleave_moe_layer_step > 0 and (
+            self.layer_idx + 1) % config.interleave_moe_layer_step == 0
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #17917 - [Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser

- Link: https://github.com/vllm-project/vllm/pull/17917
- Status/date: merged / 2025-05-22
- Trace source: `git log --name-only -- <model-files>` found it through `examples/tool_chat_template_llama4_pythonic.jinja`; associated commits `c91fe7b1b9c4`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +541/-72, 720 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser"; model line: Llama 4; category: bug fix; main diff: `examples/tool_chat_template_llama4_pythonic.jinja`; technical summary: Covers "[Frontend][Bug Fix] Update llama4 pythonic jinja template and llama4_pythonic parser"; the main implementation surface is `examples/tool_chat_template_llama4_pythonic.jinja`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12.
- Code diff details:
  - `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64 (100 lines); hunks: -1,85 +1,51; -91,10 +57,12
- Key code excerpts:

```diff
diff -- examples/tool_chat_template_llama4_pythonic.jinja
@@ -1,85 +1,51 @@
-{%- if custom_tools is defined %}
+{%- if custom_tools is defined and custom_tools%}
-{%- if not tools_in_user_message is defined %}
-    {%- set tools_in_user_message = false %}
-{%- endif %}
-{%- if not tools is defined %}
```

- Reviewed files:
  - docs: `examples/tool_chat_template_llama4_pythonic.jinja` modified +36/-64
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/tool_parsers/test_llama4_pythonic_tool_parser.py`, `tests/tool_use/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #18368 - [Model] enable data parallel for Llama4 vision encoder

- Link: https://github.com/vllm-project/vllm/pull/18368
- Status/date: merged / 2025-06-02
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `ebb1ec931871`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +214/-68, 496 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] enable data parallel for Llama4 vision encoder"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Model] enable data parallel for Llama4 vision encoder"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle, touching `Llama4ImagePatchInputs, Llama4VisionMLP, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +167/-68 (235 lines); hunks: -34,6 +34,7; -49,6 +50,7; symbols: Llama4ImagePatchInputs, Llama4VisionMLP, __init__, pixel_shuffle
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -34,6 +34,7 @@
+                                               ReplicatedLinear,
@@ -49,6 +50,7 @@
+from vllm.multimodal.utils import run_dp_sharded_vision_model
@@ -84,23 +86,29 @@ class Llama4ImagePatchInputs(TypedDict):
-    def __init__(self,
-                 input_size: int,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +167/-68
- Risk and verification: Runtime changes concentrate in `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #19997 - [Llama4] Update `attn_temperature_tuning`

- Link: https://github.com/vllm-project/vllm/pull/19997
- Status/date: merged / 2025-06-25
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1afa9948f593`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +1/-2, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] Update `attn_temperature_tuning`"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Llama4] Update `attn_temperature_tuning`"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +1/-2 (3 lines); hunks: -148,9 +148,8 @@ def __init__(self,; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -148,9 +148,8 @@ def __init__(self,
-        # TODO: attn_temperature_tuning should be a bool in huggingface
-            config.attn_temperature_tuning > 0
+            config.attn_temperature_tuning
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +1/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20419 - Enable ModelOpt Llama4 fp8 checkpoint deployment

- Link: https://github.com/vllm-project/vllm/pull/20419
- Status/date: merged / 2025-07-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `4afe687a8291`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +501/-35, 693 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Enable ModelOpt Llama4 fp8 checkpoint deployment"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "Enable ModelOpt Llama4 fp8 checkpoint deployment"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint, touching `Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights`; `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +144/-20 (164 lines); hunks: -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMult...; -902,32 +903,109 @@ def _consolidate_qkv_weights(; symbols: Llama4ForConditionalGeneration, _consolidate_qkv_weights, load_weights, _rename_weight_for_modelopt_checkpoint
  - `vllm/model_executor/models/llama4.py` modified +55/-4 (59 lines); hunks: -35,7 +35,8; -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -717,6 +717,7 @@ class Llama4ForConditionalGeneration(nn.Module, SupportsMultiModal,
+        "gate_up_proj": ["gate_proj", "up_proj"],
@@ -902,32 +903,109 @@ def _consolidate_qkv_weights(
-    def load_weights(self, weights: Iterable[tuple[str,
-                                                   torch.Tensor]]) -> set[str]:
+    def _rename_weight_for_modelopt_checkpoint(self, name: str) -> str:
+        """Rename weights from ModelOpt llama4 fp8 checkpoints to vLLM
diff -- vllm/model_executor/models/llama4.py
@@ -35,7 +35,8 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader, maybe_remap_kv_scale_name)
@@ -432,12 +433,24 @@ def load_weights(self, weights: Iterable[tuple[str,
-                name = name.replace(weight_name, param_name)
+                # This check is for ModelOpt ckpts with kv cache quant enabled
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +144/-20; `vllm/model_executor/models/llama4.py` modified +55/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/model_loader/weight_utils.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20591 - [Meta] Llama4 EAGLE Support

- Link: https://github.com/vllm-project/vllm/pull/20591
- Status/date: merged / 2025-07-16
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `c11013db8b76`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +258/-18, 363 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Meta] Llama4 EAGLE Support"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/llama4_eagle.py`; technical summary: Covers "[Meta] Llama4 EAGLE Support"; the main implementation surface is `vllm/model_executor/models/llama4_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights, touching `LlamaModel, __init__, forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` added +214/-0 (214 lines); hunks: -0,0 +1,214; symbols: LlamaModel, __init__, forward, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -0,0 +1,214 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright 2025 the LLAMA4, Meta Inc., vLLM, and HuggingFace Inc. team.
+# All rights reserved.
+#
+#
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` added +214/-0
- Risk and verification: The diff ships test coverage in `tests/models/registry.py`, `tests/models/test_initialization.py`, `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #21499 - [NVIDIA] Fix Llama4 Scout FP4 functionality issues

- Link: https://github.com/vllm-project/vllm/pull/21499
- Status/date: merged / 2025-07-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ff08e51940a7`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +219/-70, 432 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[NVIDIA] Fix Llama4 Scout FP4 functionality issues"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[NVIDIA] Fix Llama4 Scout FP4 functionality issues"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +205/-67 (272 lines); hunks: -342,34 +342,94 @@ def load_moe_expert_weights(; -382,6 +442,9 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -342,34 +342,94 @@ def load_moe_expert_weights(
+        """
+        Load MoE expert weights.
+        Args:
+            name: The name of the weight to load.
+            loaded_weight: The weight to load.
+            params_dict: The dictionary of module parameters.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +205/-67
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #20788 - [Meta] Official Eagle mm support, first enablement on llama4

- Link: https://github.com/vllm-project/vllm/pull/20788
- Status/date: merged / 2025-07-31
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/llama4_eagle.py`; associated commits `9e0726e5bfd2`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +206/-37, 487 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Meta] Official Eagle mm support, first enablement on llama4"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Meta] Official Eagle mm support, first enablement on llama4"; the main implementation surface is `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights, touching `__init__, get_input_embeddings, forward`; `vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +31/-4 (35 lines); hunks: -37,8 +37,9; -78,15 +79,23 @@ def __init__(; symbols: __init__, get_input_embeddings, forward, load_weights
  - `vllm/model_executor/models/llama4.py` modified +1/-0 (1 lines); hunks: -256,6 +256,7 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -37,8 +37,9 @@
+from vllm.multimodal.inputs import NestedTensors
-from .utils import AutoWeightsLoader, maybe_prefix
+from .utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings
@@ -78,15 +79,23 @@ def __init__(
+    def get_input_embeddings(
+        self,
diff -- vllm/model_executor/models/llama4.py
@@ -256,6 +256,7 @@ def __init__(
+        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +31/-4; `vllm/model_executor/models/llama4.py` modified +1/-0
- Risk and verification: The diff ships test coverage in `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #22107 - [Fix] Fix llama4 modelopt weight loading error

- Link: https://github.com/vllm-project/vllm/pull/22107
- Status/date: merged / 2025-08-03
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `337eb23bcca6`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +5/-4, 33 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix] Fix llama4 modelopt weight loading error"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Fix] Fix llama4 modelopt weight loading error"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint, touching `_consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +5/-4 (9 lines); hunks: -906,11 +906,13 @@ def _consolidate_qkv_weights(; -929,15 +931,14 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str...; symbols: _consolidate_qkv_weights, _rename_weight_for_modelopt_checkpoint
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -906,11 +906,13 @@ def _consolidate_qkv_weights(
-        if name.startswith("model."):
+        if name.startswith("model.") or name.startswith(
+                "language_model.model."):
+            renamed = name.replace("model.", "language_model.model.",
+                                   1) if name.startswith("model.") else name
-                renamed = name.replace("model.", "language_model.model.", 1)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +5/-4
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22511 - Fix Llama4 FlashInfer FP4 MoE issues

- Link: https://github.com/vllm-project/vllm/pull/22511
- Status/date: merged / 2025-08-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 3 files, +9/-5, 35 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix Llama4 FlashInfer FP4 MoE issues"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`; technical summary: Covers "Fix Llama4 FlashInfer FP4 MoE issues"; the main implementation surface is `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare, touching `prepare`; `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply, touching `apply`; `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply, touching `apply`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1 (7 lines); hunks: -60,7 +60,12 @@ def prepare(; symbols: prepare
  - `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2 (5 lines); hunks: -1299,8 +1299,9 @@ def apply(; symbols: apply
  - `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2 (2 lines); hunks: -170,8 +170,6 @@ def apply(; symbols: apply
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py
@@ -60,7 +60,12 @@ def prepare(
-        assert not apply_router_weight_on_input
+        if apply_router_weight_on_input:
+            topk = topk_ids.size(1)
+            # TODO: this only works for topK=1, will need to update for topK>1
+            assert topk == 1, \
+                "apply_router_weight_on_input is only implemented for topk=1"
diff -- vllm/model_executor/layers/quantization/modelopt.py
@@ -1299,8 +1299,9 @@ def apply(
-                n_group=num_expert_group,
-                topk_group=topk_group,
+                n_group=num_expert_group
+                if num_expert_group is not None else 0,
+                topk_group=topk_group if topk_group is not None else 0,
diff -- vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py
@@ -170,8 +170,6 @@ def apply(
-        assert not apply_router_weight_on_input
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py` modified +6/-1; `vllm/model_executor/layers/quantization/modelopt.py` modified +3/-2; `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` modified +0/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py`, `vllm/model_executor/layers/fused_moe/flashinfer_cutlass_prepare_finalize.py`, `vllm/model_executor/layers/quantization/modelopt.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22701 - Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op

- Link: https://github.com/vllm-project/vllm/pull/22701
- Status/date: merged / 2025-08-13
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `4f0f844b1675`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +6/-2, 17 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "Fix cuda illegal mem access with Llama4 TP8 + rms_norm custom op"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward, touching `forward`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -224,10 +224,14 @@ def forward(; symbols: forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -224,10 +224,14 @@ def forward(
-            q = q.reshape(-1, self.num_heads, self.head_dim)
+            # Normalization is applied on the head_dim dimension. The rest of
+            # the dimensions are collapsed into a single dimension to support
+            # custom rms_norm cuda kernel.
+            q = q.reshape(-1, self.head_dim)
-            k = k.reshape(-1, self.num_kv_heads, self.head_dim)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22691 - [bug fix] Fix llama4 spec decoding

- Link: https://github.com/vllm-project/vllm/pull/22691
- Status/date: merged / 2025-08-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `5bfe0dea7a34`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +4/-2, 20 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[bug fix] Fix llama4 spec decoding"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[bug fix] Fix llama4 spec decoding"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale, touching `__init__, _get_attn_scale`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +4/-2 (6 lines); hunks: -195,7 +195,9 @@ def __init__(self,; -206,7 +208,7 @@ def __init__(self,; symbols: __init__, _get_attn_scale
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -195,7 +195,9 @@ def __init__(self,
-        attn_cls = Attention if self.nope else ChunkedLocalAttention
+        use_chunked_local_attn = not self.nope and config.attention_chunk_size
+        attn_cls = (ChunkedLocalAttention
+                    if use_chunked_local_attn else Attention)
@@ -206,7 +208,7 @@ def __init__(self,
-            } if not self.nope else {}))
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +4/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #22021 - Migrate Llama4ImagePatchInputs to TensorSchema

- Link: https://github.com/vllm-project/vllm/pull/22021
- Status/date: merged / 2025-08-28
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `f32a5bc5058a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-18, 87 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Migrate Llama4ImagePatchInputs to TensorSchema"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "Migrate Llama4ImagePatchInputs to TensorSchema"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input, touching `Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +23/-18 (41 lines); hunks: -19,7 +19,7; -53,35 +53,42; symbols: Llama4ImagePatchInputs, _call_hf_processor, _parse_and_validate_image_input
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -19,7 +19,7 @@
-from typing import Literal, Optional, TypedDict, Union
+from typing import Annotated, Literal, Optional, Union
@@ -53,35 +53,42 @@
+from vllm.utils.tensor_schema import TensorSchema, TensorShape
-class Llama4ImagePatchInputs(TypedDict):
-    type: Literal["pixel_values"]
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-18
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #24444 - [Bugfix] Fix platform-specific routing in CustomOp implementations

- Link: https://github.com/vllm-project/vllm/pull/24444
- Status/date: merged / 2025-09-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 8 files, +53/-30, 187 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix platform-specific routing in CustomOp implementations"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`; technical summary: Covers "[Bugfix] Fix platform-specific routing in CustomOp implementations"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/mrope.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23 (23 lines); hunks: -8,7 +8,6; -202,28 +201,6 @@ def __init__(; symbols: __init__, forward, forward_native, touching `__init__, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1 (11 lines); hunks: -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -129,3 +129,12 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda, touching `_compute_cos_sin_cache, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1 (11 lines); hunks: -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -161,6 +161,15 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda, touching `_compute_cos_sin_cache, forward, forward_native`; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1 (10 lines); hunks: -12,7 +12,7; -70,3 +70,11 @@ def forward(; symbols: Ernie4_5_VLRotaryEmbedding, forward, forward_native, forward_cuda, touching `Ernie4_5_VLRotaryEmbedding, forward, forward_native`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23 (23 lines); hunks: -8,7 +8,6; -202,28 +201,6 @@ def __init__(; symbols: __init__, forward, forward_native
  - `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1 (11 lines); hunks: -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -129,3 +129,12 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1 (11 lines); hunks: -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:; -161,6 +161,15 @@ def forward(; symbols: _compute_cos_sin_cache, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1 (10 lines); hunks: -12,7 +12,7; -70,3 +70,11 @@ def forward(; symbols: Ernie4_5_VLRotaryEmbedding, forward, forward_native, forward_cuda
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +8/-1 (9 lines); hunks: -1593,7 +1593,7 @@ def maybe_all_reduce_tensor_model_parallel(; -1627,6 +1627,13 @@ def forward(; symbols: maybe_all_reduce_tensor_model_parallel, forward, forward_native, forward_cuda
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/mrope.py
@@ -8,7 +8,6 @@
-from vllm.platforms import current_platform
@@ -202,28 +201,6 @@ def __init__(
-        self.use_triton = current_platform.is_cuda_alike()
-    def forward(
-        self,
-        positions: torch.Tensor,
diff -- vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
@@ -88,7 +88,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
-    def forward(
+    def forward_native(
@@ -129,3 +129,12 @@ def forward(
+    def forward_cuda(
+        self,
+        positions: torch.Tensor,
diff -- vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py
@@ -111,7 +111,7 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/mrope.py` modified +0/-23; `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` modified +10/-1; `vllm/model_executor/layers/rotary_embedding/dual_chunk_rope.py` modified +10/-1; `vllm/model_executor/layers/rotary_embedding/ernie45_vl_rope.py` modified +9/-1; `vllm/model_executor/layers/fused_moe/layer.py` modified +8/-1; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +8/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/activation.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25889 - [Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`

- Link: https://github.com/vllm-project/vllm/pull/25889
- Status/date: merged / 2025-09-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `43b752c325d5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +3/-1, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; technical summary: Covers "[Llama4] [multimodal] Fix misplaced dtype cast of `cos_sin_cache` in `Llama4VisionRotaryEmbedding`"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native, touching `forward_native`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1 (4 lines); hunks: -59,7 +59,9 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -59,7 +59,9 @@ def forward_native(  # type: ignore[override]
-        self._match_cos_sin_cache_dtype(query)
+        # self.cos_sin_cache here is complex tensor so we cannot cast into
+        # query's dtype directly with self._match_cos_sin_cache_dtype
+        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +3/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25961 - Support llama3 eagle3 head with llama4 verifier

- Link: https://github.com/vllm-project/vllm/pull/25961
- Status/date: merged / 2025-10-06
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `05f6846ede18`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +83/-8, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Support llama3 eagle3 head with llama4 verifier"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "Support llama3 eagle3 head with llama4 verifier"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers, touching `get_dummy_mm_data, Llama4ForConditionalGeneration, __init__`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +25/-2 (27 lines); hunks: -64,7 +64,12; -717,7 +722,9 @@ def get_dummy_mm_data(; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, __init__, set_aux_hidden_state_layers
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -64,7 +64,12 @@
-from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
+from .interfaces import (
+    MultiModalEmbeddings,
+    SupportsEagle3,
+    SupportsMultiModal,
+    SupportsPP,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +25/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama.py`, `vllm/model_executor/models/llama_eagle3.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #26790 - llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch

- Link: https://github.com/vllm-project/vllm/pull/26790
- Status/date: merged / 2025-10-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `87efc681dbd5`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 11 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; technical summary: Covers "llama4_vision_rope: add HIP override to accept (q, k) and avoid (positions, q, k) mismatch"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip, touching `forward_cuda, forward_hip`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0 (7 lines); hunks: -78,3 +78,10 @@ def forward_cuda( # type: ignore[override]; symbols: forward_cuda, forward_hip
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -78,3 +78,10 @@ def forward_cuda(  # type: ignore[override]
+    def forward_hip(  # type: ignore[override]
+        self,
+        query: torch.Tensor,
+        key: torch.Tensor | None = None,
+    ) -> tuple[torch.Tensor, torch.Tensor | None]:
+        return self.forward_native(query, key)
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #27136 - [Fix][Spec Decode] Fix llama4 draft loading with different quantization

- Link: https://github.com/vllm-project/vllm/pull/27136
- Status/date: merged / 2025-10-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `be4445072c4e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +17/-10, 34 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Fix][Spec Decode] Fix llama4 draft loading with different quantization"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4_eagle.py`; technical summary: Covers "[Fix][Spec Decode] Fix llama4 draft loading with different quantization"; the main implementation surface is `vllm/model_executor/models/llama4_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +17/-10 (27 lines); hunks: -60,16 +60,23 @@ def __init__(; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -60,16 +60,23 @@ def __init__(
-        self.layers = nn.ModuleList(
-            [
-                Llama4DecoderLayer(
-                    vllm_config=vllm_config,
-                    prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
-                    config=self.config,
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +17/-10
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4_eagle.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #25145 - [XPU][bugfix] fix rope for llama4 and deepseek

- Link: https://github.com/vllm-project/vllm/pull/25145
- Status/date: merged / 2025-10-30
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; associated commits `b798e39f931a`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +22/-32, 116 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[XPU][bugfix] fix rope for llama4 and deepseek"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; technical summary: Covers "[XPU][bugfix] fix rope for llama4 and deepseek"; the main implementation surface is `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip, touching `Llama4VisionRotaryEmbedding, __init__, forward_cuda`.
- Code diff details:
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9 (11 lines); hunks: -5,10 +5,10; -78,10 +78,3 @@ def forward_cuda( # type: ignore[override]; symbols: Llama4VisionRotaryEmbedding, __init__, forward_cuda, forward_hip
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -5,10 +5,10 @@
-from .base import RotaryEmbedding
+from .base import RotaryEmbeddingBase
-class Llama4VisionRotaryEmbedding(RotaryEmbedding):
+class Llama4VisionRotaryEmbedding(RotaryEmbeddingBase):
@@ -78,10 +78,3 @@ def forward_cuda(  # type: ignore[override]
-    def forward_hip(  # type: ignore[override]
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +2/-9
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/rotary_embedding/base.py`, `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28602 - LLaMA4 LoRA Adapter Enablement

- Link: https://github.com/vllm-project/vllm/pull/28602
- Status/date: merged / 2025-11-14
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `964d65deedb9`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +34/-2, 79 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "LLaMA4 LoRA Adapter Enablement"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "LLaMA4 LoRA Adapter Enablement"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping, touching `get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +34/-2 (36 lines); hunks: -35,6 +35,7; -45,6 +46,7; symbols: get_dummy_mm_data, Llama4ForConditionalGeneration, _load_other_weights, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -35,6 +35,7 @@
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -45,6 +46,7 @@
+from vllm.model_executor.models.module_mapping import MultiModelKeys
@@ -68,11 +70,15 @@
+    SupportsLoRA,
-from .utils import AutoWeightsLoader, maybe_prefix
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +34/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #28577 - [BugFix] Fix Llama4 Pipeline Parallelism Assert Error

- Link: https://github.com/vllm-project/vllm/pull/28577
- Status/date: merged / 2025-11-20
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `dc45efc8ef7f`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +7/-0, 28 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BugFix] Fix Llama4 Pipeline Parallelism Assert Error"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[BugFix] Fix Llama4 Pipeline Parallelism Assert Error"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata, touching `set_moe_parameters, update_physical_experts_metadata`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +7/-0 (7 lines); hunks: -54,6 +54,7; -738,6 +739,9 @@ def set_moe_parameters(self):; symbols: set_moe_parameters, update_physical_experts_metadata
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -54,6 +54,7 @@
+    PPMissingLayer,
@@ -738,6 +739,9 @@ def set_moe_parameters(self):
+            if isinstance(layer, PPMissingLayer):
+                continue
@@ -774,6 +778,9 @@ def update_physical_experts_metadata(
+            if isinstance(layer, PPMissingLayer):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +7/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #29926 - [Bugfix][llama4_eagle] Fix missing 'lm_head' attribute

- Link: https://github.com/vllm-project/vllm/pull/29926
- Status/date: merged / 2025-12-05
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4_eagle.py`; associated commits `962d703818c0`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +16/-3, 46 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix][llama4_eagle] Fix missing 'lm_head' attribute"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4_eagle.py`; technical summary: Covers "[Bugfix][llama4_eagle] Fix missing 'lm_head' attribute"; the main implementation surface is `vllm/model_executor/models/llama4_eagle.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform, touching `__init__, transform`.
- Code diff details:
  - `vllm/model_executor/models/llama4_eagle.py` modified +11/-2 (13 lines); hunks: -28,7 +28,10; -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, transform
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -28,7 +28,10 @@
-from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
+from vllm.model_executor.layers.vocab_parallel_embedding import (
+    ParallelLMHead,
+    VocabParallelEmbedding,
+)
@@ -182,6 +185,12 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4_eagle.py` modified +11/-2
- Risk and verification: The diff ships test coverage in `tests/v1/e2e/test_spec_decode.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #30709 - [Misc][LLaMa4] Compile LLaMa Vision Encoder

- Link: https://github.com/vllm-project/vllm/pull/30709
- Status/date: merged / 2026-01-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`, `vllm/model_executor/models/mllama4.py`; associated commits `ea6d067a2aeb`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +85/-20, 202 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc][LLaMa4] Compile LLaMa Vision Encoder"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`; technical summary: Covers "[Misc][LLaMa4] Compile LLaMa Vision Encoder"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__, touching `forward, Llama4VisionModel, __init__`; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native, touching `forward_native`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +29/-9 (38 lines); hunks: -31,9 +31,11; -47,6 +49,7; symbols: forward, Llama4VisionModel, __init__
  - `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2 (7 lines); hunks: -60,14 +60,17 @@ def forward_native( # type: ignore[override]; symbols: forward_native
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -31,9 +31,11 @@
-from vllm.config import VllmConfig
+from vllm.compilation.decorators import support_torch_compile
+from vllm.config import VllmConfig, set_current_vllm_config
+from vllm.forward_context import set_forward_context
@@ -47,6 +49,7 @@
+from vllm.model_executor.models.vision import should_torch_compile_mm_vit
diff -- vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -60,14 +60,17 @@ def forward_native(  # type: ignore[override]
-        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
+        # NOTE: by not storing cos_sin_cache in self, we can avoid
+        # memory buffer update which is costly to runtime
+        cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
-        freqs_ci = self.cos_sin_cache.view(*broadcast_shape)
+        freqs_ci = cos_sin_cache.view(*broadcast_shape)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +29/-9; `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py` modified +5/-2
- Risk and verification: The diff ships test coverage in `tests/compile/fullgraph/test_multimodal_compile.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #32886 - [Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4

- Link: https://github.com/vllm-project/vllm/pull/32886
- Status/date: merged / 2026-01-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1fb648bf107e`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +21/-1, 36 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Bugfix] Fix FP8 MoE EP Weight Loading for ModelOpt Llama4"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, touching `load_moe_expert_weights`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +21/-1 (22 lines); hunks: -51,6 +51,8; -504,7 +506,25 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -51,6 +51,8 @@
+from vllm.platforms import current_platform
+from vllm.utils.torch_utils import is_torch_equal_or_newer
@@ -504,7 +506,25 @@ def load_moe_expert_weights(
-                    new_loaded_weight = new_loaded_weight[local_expert_indices]
+                    # Workaround for FP8 CPU indexing on older PyTorch:
+                    # https://github.com/vllm-project/vllm/issues/32862
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +21/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34243 - [Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)

- Link: https://github.com/vllm-project/vllm/pull/34243
- Status/date: merged / 2026-02-11
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `11c7ace34061`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-5, 76 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Bugfix] Enable attn quantization of Llama-4 by correctly permuting scales for rope (int8, fp8)"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-5 (34 lines); hunks: -44,6 +44,9; -829,11 +832,20 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,6 +44,9 @@
+from vllm.model_executor.layers.quantization.compressed_tensors import (
+    compressed_tensors as ct,
+)
@@ -829,11 +832,20 @@ def permute_qk_weight_for_rotary(
-        def permute(w: torch.Tensor, n_heads: int, is_weight_scale: bool):
+        def permute(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-5
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34471 - [Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers

- Link: https://github.com/vllm-project/vllm/pull/34471
- Status/date: merged / 2026-02-19
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `ee1d25f199ee`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +29/-68, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,9 +44,6 @@
-from vllm.model_executor.layers.quantization.compressed_tensors import (
-    compressed_tensors as ct,
-)
@@ -831,74 +828,38 @@ def permute_qk_weight_for_rotary(
-        # Helper function to permute the weight's channels
-        def permute(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #34997 - Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers "

- Link: https://github.com/vllm-project/vllm/pull/34997
- Status/date: merged / 2026-02-21
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `0e22cd618b5d`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +68/-29, 114 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers ""; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "Revert "[Llama4,Quantization] Simplify and generalize logic for Q/K permutations in quantized self-attn layers ""; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +68/-29 (97 lines); hunks: -44,6 +44,9; -828,38 +831,74 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,6 +44,9 @@
+from vllm.model_executor.layers.quantization.compressed_tensors import (
+    compressed_tensors as ct,
+)
@@ -828,38 +831,74 @@ def permute_qk_weight_for_rotary(
-        modules = name.split(".")
-        # Permute Q/K weights and corresponding scales for rotary embedding.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +68/-29
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35033 - [Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests

- Link: https://github.com/vllm-project/vllm/pull/35033
- Status/date: merged / 2026-02-23
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`; associated commits `1e8438a89a64`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +31/-70, 127 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Llama4,CI] Bring back Llama-4 bug fixes, and also fix Maverick tests"; the main implementation surface is `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute, touching `permute_qk_weight_for_rotary, permute`.
- Code diff details:
  - `vllm/model_executor/models/llama4.py` modified +29/-68 (97 lines); hunks: -44,9 +44,6; -831,74 +828,38 @@ def permute_qk_weight_for_rotary(; symbols: permute_qk_weight_for_rotary, permute
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/llama4.py
@@ -44,9 +44,6 @@
-from vllm.model_executor.layers.quantization.compressed_tensors import (
-    compressed_tensors as ct,
-)
@@ -831,74 +828,38 @@ def permute_qk_weight_for_rotary(
-        # Helper function to permute the weight's channels
-        def permute(
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/llama4.py` modified +29/-68
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_maverick.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35147 - [Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)

- Link: https://github.com/vllm-project/vllm/pull/35147
- Status/date: merged / 2026-02-24
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `012dee92331c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 1 files, +23/-1, 30 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[Feature] Add LoRA tower/connector support for Llama 4 Vision (mllama4)"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens, touching `get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +23/-1 (24 lines); hunks: -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:; symbols: get_mm_mapping, get_num_mm_encoder_tokens, get_num_mm_connector_tokens
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -1151,6 +1151,28 @@ def get_mm_mapping(self) -> MultiModelKeys:
-            connector="multi_modal_projector.",
+            connector=[
+                "multi_modal_projector.",
+                "vision_model.vision_adapter.",
+            ],
+    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +23/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36436 - [Misc] Refactored 5 duplicate helper functions that were copied-pasted across multiple parsers

- Link: https://github.com/vllm-project/vllm/pull/36436
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +247/-452, 911 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Refactored 5 duplicate helper functions that were copied-pasted across multiple parsers"; model line: Llama 4; category: model implementation change; main diff: `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/olmo3_tool_parser.py`, `vllm/tool_parsers/pythonic_tool_parser.py`; technical summary: Covers "[Misc] Refactored 5 duplicate helper functions that were copied-pasted across multiple parsers"; the main implementation surface is `vllm/tool_parsers/utils.py`, `vllm/tool_parsers/olmo3_tool_parser.py`, `vllm/tool_parsers/pythonic_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/utils.py` modified +209/-0 (209 lines); hunks: -1,6 +1,7; -17,6 +18,15; symbols: find_common_prefix, get_json_schema_from_tools, UnexpectedAstError, get_parameter_value, touching `find_common_prefix, get_json_schema_from_tools, UnexpectedAstError`; `vllm/tool_parsers/olmo3_tool_parser.py` modified +13/-156 (169 lines); hunks: -1,9 +1,8; -13,25 +12,23; symbols: _UnexpectedAstError, Olmo3PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming, touching `_UnexpectedAstError, Olmo3PythonicToolParser, extract_tool_calls`; `vllm/tool_parsers/pythonic_tool_parser.py` modified +12/-149 (161 lines); hunks: -2,9 +2,7; -14,25 +12,23; symbols: _UnexpectedAstError, PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming, touching `_UnexpectedAstError, PythonicToolParser, extract_tool_calls`; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +13/-147 (160 lines); hunks: -1,9 +1,8; -13,25 +12,23; symbols: _UnexpectedAstError, Llama4PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming, touching `_UnexpectedAstError, Llama4PythonicToolParser, extract_tool_calls`.
- Code diff details:
  - `vllm/tool_parsers/utils.py` modified +209/-0 (209 lines); hunks: -1,6 +1,7; -17,6 +18,15; symbols: find_common_prefix, get_json_schema_from_tools, UnexpectedAstError, get_parameter_value
  - `vllm/tool_parsers/olmo3_tool_parser.py` modified +13/-156 (169 lines); hunks: -1,9 +1,8; -13,25 +12,23; symbols: _UnexpectedAstError, Olmo3PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming
  - `vllm/tool_parsers/pythonic_tool_parser.py` modified +12/-149 (161 lines); hunks: -2,9 +2,7; -14,25 +12,23; symbols: _UnexpectedAstError, PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +13/-147 (160 lines); hunks: -1,9 +1,8; -13,25 +12,23; symbols: _UnexpectedAstError, Llama4PythonicToolParser, extract_tool_calls, extract_tool_calls_streaming
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/utils.py
@@ -1,6 +1,7 @@
+import ast
@@ -17,6 +18,15 @@
+from vllm.entrypoints.openai.engine.protocol import (
+    DeltaFunctionCall,
+    DeltaToolCall,
+    FunctionCall,
diff -- vllm/tool_parsers/olmo3_tool_parser.py
@@ -1,9 +1,8 @@
-import json
-from typing import Any
@@ -13,25 +12,23 @@
-    DeltaFunctionCall,
-    DeltaToolCall,
-    FunctionCall,
diff -- vllm/tool_parsers/pythonic_tool_parser.py
@@ -2,9 +2,7 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/utils.py` modified +209/-0; `vllm/tool_parsers/olmo3_tool_parser.py` modified +13/-156; `vllm/tool_parsers/pythonic_tool_parser.py` modified +12/-149; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +13/-147
- Risk and verification: Runtime changes concentrate in `vllm/tool_parsers/llama4_pythonic_tool_parser.py`, `vllm/tool_parsers/olmo3_tool_parser.py`, `vllm/tool_parsers/pythonic_tool_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36281 - [BE] Rename `should_torch_compile_mm_vit` to `should_torch_compile_mm_encoder`

- Link: https://github.com/vllm-project/vllm/pull/36281
- Status/date: merged / 2026-03-09
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 6 files, +23/-17, 138 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[BE] Rename `should_torch_compile_mm_vit` to `should_torch_compile_mm_encoder`"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[BE] Rename `should_torch_compile_mm_vit` to `should_torch_compile_mm_encoder`"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-5 (12 lines); hunks: -42,7 +42,10; -65,7 +68,6; symbols: forward, Qwen2_5_VisionBlock, __init__, Qwen2_5_VisionPatchEmbed, touching `forward, Qwen2_5_VisionBlock, __init__`; `vllm/model_executor/models/lfm2_siglip2.py` modified +5/-3 (8 lines); hunks: -10,7 +10,10; -25,7 +28,6; symbols: forward, Siglip2EncoderLayer, __init__, touching `forward, Siglip2EncoderLayer, __init__`; `vllm/model_executor/models/mllama4.py` modified +5/-3 (8 lines); hunks: -31,7 +31,10; -49,7 +52,6; symbols: forward, Llama4VisionModel, __init__, touching `forward, Llama4VisionModel, __init__`; `vllm/model_executor/models/vision.py` modified +0/-5 (5 lines); hunks: -143,11 +143,6 @@ def is_vit_use_data_parallel():; symbols: is_vit_use_data_parallel, should_torch_compile_mm_vit, touching `is_vit_use_data_parallel, should_torch_compile_mm_vit`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-5 (12 lines); hunks: -42,7 +42,10; -65,7 +68,6; symbols: forward, Qwen2_5_VisionBlock, __init__, Qwen2_5_VisionPatchEmbed
  - `vllm/model_executor/models/lfm2_siglip2.py` modified +5/-3 (8 lines); hunks: -10,7 +10,10; -25,7 +28,6; symbols: forward, Siglip2EncoderLayer, __init__
  - `vllm/model_executor/models/mllama4.py` modified +5/-3 (8 lines); hunks: -31,7 +31,10; -49,7 +52,6; symbols: forward, Llama4VisionModel, __init__
  - `vllm/model_executor/models/vision.py` modified +0/-5 (5 lines); hunks: -143,11 +143,6 @@ def is_vit_use_data_parallel():; symbols: is_vit_use_data_parallel, should_torch_compile_mm_vit
  - `docs/design/torch_compile_multimodal.md` modified +1/-1 (2 lines); hunks: -26,7 +26,7 @@ This feature is off by default, but can be enabled by setting...
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -42,7 +42,10 @@
-from vllm.compilation.decorators import support_torch_compile
+from vllm.compilation.decorators import (
+    should_torch_compile_mm_encoder,
+    support_torch_compile,
+)
@@ -65,7 +68,6 @@
diff -- vllm/model_executor/models/lfm2_siglip2.py
@@ -10,7 +10,10 @@
-from vllm.compilation.decorators import support_torch_compile
+from vllm.compilation.decorators import (
+    should_torch_compile_mm_encoder,
+    support_torch_compile,
+)
@@ -25,7 +28,6 @@
diff -- vllm/model_executor/models/mllama4.py
@@ -31,7 +31,10 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +7/-5; `vllm/model_executor/models/lfm2_siglip2.py` modified +5/-3; `vllm/model_executor/models/mllama4.py` modified +5/-3; `vllm/model_executor/models/vision.py` modified +0/-5; `vllm/compilation/decorators.py` modified +5/-0
  - docs: `docs/design/torch_compile_multimodal.md` modified +1/-1
- Risk and verification: Runtime changes concentrate in `vllm/compilation/decorators.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #36770 - [Misc] Clean up renderers

- Link: https://github.com/vllm-project/vllm/pull/36770
- Status/date: merged / 2026-03-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 12 files, +136/-220, 632 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Clean up renderers"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/transformers_utils/processors/kimi_audio.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/kimi_audio.py`; technical summary: Covers "[Misc] Clean up renderers"; the main implementation surface is `vllm/transformers_utils/processors/kimi_audio.py`, `tests/models/multimodal/processing/test_common.py`, `vllm/model_executor/models/kimi_audio.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/transformers_utils/processors/kimi_audio.py` modified +20/-78 (98 lines); hunks: -1,10 +1,8; -19,42 +17,13; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class, touching `_get_feat_extract_output_lengths, KimiAudioProcessor, __init__`; `tests/models/multimodal/processing/test_common.py` modified +25/-68 (93 lines); hunks: -6,9 +6,6; -21,7 +18,10; symbols: glmasr_patch_mm_data, get_text_token_prompts, test_processing_correctness, touching `glmasr_patch_mm_data, get_text_token_prompts, test_processing_correctness`; `vllm/model_executor/models/kimi_audio.py` modified +47/-35 (82 lines); hunks: -10,11 +10,13; -47,7 +49,10; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, __init__, KimiAudioProcessingInfo, touching `_get_whisper_local_path, _get_feat_extract_output_lengths, __init__`; `vllm/model_executor/models/pixtral.py` modified +10/-4 (14 lines); hunks: -172,12 +172,20 @@ def get_dummy_processor_inputs(; -192,8 +200,6 @@ def get_dummy_processor_inputs(; symbols: get_dummy_processor_inputs, touching `get_dummy_processor_inputs`.
- Code diff details:
  - `vllm/transformers_utils/processors/kimi_audio.py` modified +20/-78 (98 lines); hunks: -1,10 +1,8; -19,42 +17,13; symbols: _get_feat_extract_output_lengths, KimiAudioProcessor, __init__, check_argument_for_proper_class
  - `tests/models/multimodal/processing/test_common.py` modified +25/-68 (93 lines); hunks: -6,9 +6,6; -21,7 +18,10; symbols: glmasr_patch_mm_data, get_text_token_prompts, test_processing_correctness
  - `vllm/model_executor/models/kimi_audio.py` modified +47/-35 (82 lines); hunks: -10,11 +10,13; -47,7 +49,10; symbols: _get_whisper_local_path, _get_feat_extract_output_lengths, __init__, KimiAudioProcessingInfo
  - `vllm/model_executor/models/pixtral.py` modified +10/-4 (14 lines); hunks: -172,12 +172,20 @@ def get_dummy_processor_inputs(; -192,8 +200,6 @@ def get_dummy_processor_inputs(; symbols: get_dummy_processor_inputs
  - `vllm/model_executor/models/mllama4.py` modified +0/-12 (12 lines); hunks: -63,12 +63,10; -546,9 +544,6 @@ def forward(; symbols: forward, Mllama4ProcessingInfo, __init__, get_hf_config
- Key code excerpts:

```diff
diff -- vllm/transformers_utils/processors/kimi_audio.py
@@ -1,10 +1,8 @@
-# ruff: noqa
-# mypy: ignore-errors
-# coding=utf-8
-# Copyright 2026 The Moonshot AI team and the HuggingFace Inc. team. All rights reserved.
+# Copyright 2026 The Moonshot AI team and the HuggingFace Inc. team.
+# All rights reserved.
diff -- tests/models/multimodal/processing/test_common.py
@@ -6,9 +6,6 @@
-from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
-from mistral_common.protocol.instruct.messages import UserMessage
-from mistral_common.protocol.instruct.request import ChatCompletionRequest
@@ -21,7 +18,10 @@
-from vllm.multimodal.processing import BaseMultiModalProcessor, InputProcessingContext
+from vllm.multimodal.processing import (
diff -- vllm/model_executor/models/kimi_audio.py
@@ -10,11 +10,13 @@
```

- Reviewed files:
  - runtime: `vllm/transformers_utils/processors/kimi_audio.py` modified +20/-78; `vllm/model_executor/models/kimi_audio.py` modified +47/-35; `vllm/model_executor/models/pixtral.py` modified +10/-4; `vllm/model_executor/models/mllama4.py` modified +0/-12; `vllm/model_executor/models/voxtral.py` modified +10/-2; `vllm/tokenizers/registry.py` modified +0/-12
  - tests: `tests/models/multimodal/processing/test_common.py` modified +25/-68
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/processing/test_common.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36063 - [Refactor] Consolidate SupportsEagle

- Link: https://github.com/vllm-project/vllm/pull/36063
- Status/date: merged / 2026-03-13
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 24 files, +229/-235, 1149 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Refactor] Consolidate SupportsEagle"; model line: Llama 4; category: model support/runtime entry; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/hunyuan_v1.py`; technical summary: Covers "[Refactor] Consolidate SupportsEagle"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/qwen3_moe.py`, `vllm/model_executor/models/hunyuan_v1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +50/-7 (57 lines); hunks: -1273,6 +1273,25 @@ def supports_any_eagle(; -1320,24 +1339,48 @@ class SupportsEagle3(SupportsEagleBase, Protocol):; symbols: supports_any_eagle, EagleModelMixin, _set_aux_hidden_state_layers, _maybe_add_hidden_state, touching `supports_any_eagle, EagleModelMixin, _set_aux_hidden_state_layers`; `vllm/model_executor/models/qwen3_moe.py` modified +16/-19 (35 lines); hunks: -65,7 +65,14; -427,7 +434,7 @@ def forward(; symbols: forward, Qwen3MoeModel, __init__, touching `forward, Qwen3MoeModel, __init__`; `vllm/model_executor/models/hunyuan_v1.py` modified +17/-15 (32 lines); hunks: -66,7 +66,14; -586,7 +593,7 @@ def forward(; symbols: forward, HunYuanModel, __init__, touching `forward, HunYuanModel, __init__`; `vllm/model_executor/models/minicpm.py` modified +15/-17 (32 lines); hunks: -63,7 +63,13; -391,7 +397,7 @@ def forward(; symbols: forward, MiniCPMModel, __init__, touching `forward, MiniCPMModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +50/-7 (57 lines); hunks: -1273,6 +1273,25 @@ def supports_any_eagle(; -1320,24 +1339,48 @@ class SupportsEagle3(SupportsEagleBase, Protocol):; symbols: supports_any_eagle, EagleModelMixin, _set_aux_hidden_state_layers, _maybe_add_hidden_state
  - `vllm/model_executor/models/qwen3_moe.py` modified +16/-19 (35 lines); hunks: -65,7 +65,14; -427,7 +434,7 @@ def forward(; symbols: forward, Qwen3MoeModel, __init__
  - `vllm/model_executor/models/hunyuan_v1.py` modified +17/-15 (32 lines); hunks: -66,7 +66,14; -586,7 +593,7 @@ def forward(; symbols: forward, HunYuanModel, __init__
  - `vllm/model_executor/models/minicpm.py` modified +15/-17 (32 lines); hunks: -63,7 +63,13; -391,7 +397,7 @@ def forward(; symbols: forward, MiniCPMModel, __init__
  - `vllm/model_executor/models/apertus.py` modified +15/-15 (30 lines); hunks: -60,7 +60,13; -313,7 +319,7 @@ def forward(; symbols: forward, ApertusModel, __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1273,6 +1273,25 @@ def supports_any_eagle(
+class EagleModelMixin:
+    aux_hidden_state_layers: tuple[int, ...] = ()
+    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        self.aux_hidden_state_layers = layers
+    def _maybe_add_hidden_state(
+        self,
diff -- vllm/model_executor/models/qwen3_moe.py
@@ -65,7 +65,14 @@
-from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
+from .interfaces import (
+    EagleModelMixin,
+    MixtureOfExperts,
+    SupportsEagle,
+    SupportsEagle3,
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -66,7 +66,14 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +50/-7; `vllm/model_executor/models/qwen3_moe.py` modified +16/-19; `vllm/model_executor/models/hunyuan_v1.py` modified +17/-15; `vllm/model_executor/models/minicpm.py` modified +15/-17; `vllm/model_executor/models/apertus.py` modified +15/-15; `vllm/model_executor/models/qwen2.py` modified +15/-15
- Risk and verification: The diff ships test coverage in `tests/v1/kv_connector/extract_hidden_states_integration/predictable_llama.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #36288 - [torch.compile][BE] Modify cudagraph callable to check for is_forward_context_set

- Link: https://github.com/vllm-project/vllm/pull/36288
- Status/date: merged / 2026-03-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +27/-29, 112 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile][BE] Modify cudagraph callable to check for is_forward_context_set"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/mllama4.py`, `docs/design/torch_compile_multimodal.md`; technical summary: Covers "[torch.compile][BE] Modify cudagraph callable to check for is_forward_context_set"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/mllama4.py`, `docs/design/torch_compile_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +15/-20 (35 lines); hunks: -49,7 +49,6; -1207,13 +1206,12 @@ def _process_image_input(; symbols: _process_image_input, _process_video_input, touching `_process_image_input, _process_video_input`; `vllm/model_executor/models/mllama4.py` modified +1/-5 (6 lines); hunks: -38,7 +38,6; -872,10 +871,7 @@ def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:; symbols: embed_multimodal, forward, touching `embed_multimodal, forward`; `docs/design/torch_compile_multimodal.md` modified +0/-3 (3 lines); hunks: -34,9 +34,6 @@ relies on caching artifacts to reduce start time, we must prop...; `vllm/compilation/cuda_graph.py` modified +11/-1 (12 lines); hunks: -16,7 +16,11; -224,6 +228,12 @@ def clear_graphs(self) -> None:; symbols: clear_graphs, __call__, touching `clear_graphs, __call__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +15/-20 (35 lines); hunks: -49,7 +49,6; -1207,13 +1206,12 @@ def _process_image_input(; symbols: _process_image_input, _process_video_input
  - `vllm/model_executor/models/mllama4.py` modified +1/-5 (6 lines); hunks: -38,7 +38,6; -872,10 +871,7 @@ def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:; symbols: embed_multimodal, forward
  - `docs/design/torch_compile_multimodal.md` modified +0/-3 (3 lines); hunks: -34,9 +34,6 @@ relies on caching artifacts to reduce start time, we must prop...
  - `vllm/compilation/cuda_graph.py` modified +11/-1 (12 lines); hunks: -16,7 +16,11; -224,6 +228,12 @@ def clear_graphs(self) -> None:; symbols: clear_graphs, __call__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -49,7 +49,6 @@
-from vllm.forward_context import set_forward_context
@@ -1207,13 +1206,12 @@ def _process_image_input(
-            with set_forward_context(None, self.vllm_config):
-                if self.use_data_parallel:
-                    return run_dp_sharded_mrope_vision_model(
-                        self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
diff -- vllm/model_executor/models/mllama4.py
@@ -38,7 +38,6 @@
-from vllm.forward_context import set_forward_context
@@ -872,10 +871,7 @@ def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
-        with (
-            set_forward_context(None, self.vllm_config),
-        ):
-            return self._process_image_input(image_input)
diff -- docs/design/torch_compile_multimodal.md
@@ -34,9 +34,6 @@ relies on caching artifacts to reduce start time, we must properly propagate the
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +15/-20; `vllm/model_executor/models/mllama4.py` modified +1/-5; `vllm/compilation/cuda_graph.py` modified +11/-1
  - docs: `docs/design/torch_compile_multimodal.md` modified +0/-3
- Risk and verification: Runtime changes concentrate in `vllm/compilation/cuda_graph.py`, `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/qwen2_5_vl.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37345 - [torch.compile][BE][Multimodal] Remove requirement to set_model_tag to avoid cache conflict

- Link: https://github.com/vllm-project/vllm/pull/37345
- Status/date: merged / 2026-03-19
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +86/-69, 312 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[torch.compile][BE][Multimodal] Remove requirement to set_model_tag to avoid cache conflict"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `docs/design/torch_compile_multimodal.md`; technical summary: Covers "[torch.compile][BE][Multimodal] Remove requirement to set_model_tag to avoid cache conflict"; the main implementation surface is `vllm/model_executor/models/qwen2_5_vl.py`, `vllm/model_executor/models/lfm2_siglip2.py`, `docs/design/torch_compile_multimodal.md`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/qwen2_5_vl.py` modified +31/-36 (67 lines); hunks: -427,6 +427,7 @@ def forward(; -486,6 +487,7 @@ def forward(; symbols: forward, Qwen2_5_VisionBlock, __init__, Qwen2_5_VisionPatchEmbed, touching `forward, Qwen2_5_VisionBlock, __init__`; `vllm/model_executor/models/lfm2_siglip2.py` modified +7/-10 (17 lines); hunks: -272,6 +272,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -395,16 +396,12 @@ def __init__(; symbols: forward, Siglip2EncoderLayer, __init__, touching `forward, Siglip2EncoderLayer, __init__`; `docs/design/torch_compile_multimodal.md` modified +5/-6 (11 lines); hunks: -29,10 +29,9 @@ To compile a multimodal component such as an encoder, we foll...; -57,8 +56,8 @@ tradeoff; symbols: name, touching `name`; `vllm/model_executor/models/mllama4.py` modified +4/-7 (11 lines); hunks: -453,7 +453,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -754,12 +756,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, Llama4VisionModel, __init__, touching `forward, Llama4VisionModel, __init__`.
- Code diff details:
  - `vllm/model_executor/models/qwen2_5_vl.py` modified +31/-36 (67 lines); hunks: -427,6 +427,7 @@ def forward(; -486,6 +487,7 @@ def forward(; symbols: forward, Qwen2_5_VisionBlock, __init__, Qwen2_5_VisionPatchEmbed
  - `vllm/model_executor/models/lfm2_siglip2.py` modified +7/-10 (17 lines); hunks: -272,6 +272,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -395,16 +396,12 @@ def __init__(; symbols: forward, Siglip2EncoderLayer, __init__
  - `docs/design/torch_compile_multimodal.md` modified +5/-6 (11 lines); hunks: -29,10 +29,9 @@ To compile a multimodal component such as an encoder, we foll...; -57,8 +56,8 @@ tradeoff; symbols: name
  - `vllm/model_executor/models/mllama4.py` modified +4/-7 (11 lines); hunks: -453,7 +453,9 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:; -754,12 +756,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: forward, Llama4VisionModel, __init__
  - `vllm/compilation/wrapper.py` modified +15/-3 (18 lines); hunks: -75,8 +75,14 @@ def _call_with_optional_nvtx_range(; -87,7 +93,9 @@ def __init__(self) -> None:; symbols: _call_with_optional_nvtx_range, __init__, reset_compile_wrapper
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/qwen2_5_vl.py
@@ -427,6 +427,7 @@ def forward(
+    is_encoder=True,
@@ -486,6 +487,7 @@ def forward(
+    is_encoder=True,
@@ -521,6 +523,7 @@ def forward(self, x: torch.Tensor) -> torch.Tensor:
+    is_encoder=True,
@@ -592,18 +595,12 @@ def __init__(
diff -- vllm/model_executor/models/lfm2_siglip2.py
@@ -272,6 +272,7 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
+    is_encoder=True,
@@ -395,16 +396,12 @@ def __init__(
-        # Keep the import local to avoid circular dependencies during model init.
-        from vllm.compilation.backends import set_model_tag
-        with set_model_tag("Siglip2Encoder", is_encoder=True):
-            self.encoder = Siglip2Encoder(
diff -- docs/design/torch_compile_multimodal.md
@@ -29,10 +29,9 @@ To compile a multimodal component such as an encoder, we follow the same mechani
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/qwen2_5_vl.py` modified +31/-36; `vllm/model_executor/models/lfm2_siglip2.py` modified +7/-10; `vllm/model_executor/models/mllama4.py` modified +4/-7; `vllm/compilation/wrapper.py` modified +15/-3; `vllm/compilation/decorators.py` modified +14/-3; `vllm/config/compilation.py` modified +10/-4
  - docs: `docs/design/torch_compile_multimodal.md` modified +5/-6
- Risk and verification: Runtime changes concentrate in `vllm/compilation/decorators.py`, `vllm/compilation/wrapper.py`, `vllm/config/compilation.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #37834 - [Test] Consolidate tool parser unit tests to tests/tool_parsers

- Link: https://github.com/vllm-project/vllm/pull/37834
- Status/date: merged / 2026-03-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 11 files, +376/-353, 774 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Test] Consolidate tool parser unit tests to tests/tool_parsers"; model line: Llama 4; category: docs/tests/CI; main diff: `tests/tool_parsers/test_hermes_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py`, `tests/tool_parsers/test_granite4_tool_parser.py`; technical summary: Covers "[Test] Consolidate tool parser unit tests to tests/tool_parsers"; the main implementation surface is `tests/tool_parsers/test_hermes_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py`, `tests/tool_parsers/test_granite4_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/tool_parsers/test_hermes_tool_parser.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: qwen_tokenizer, hermes_parser, any_chat_request, test_hermes_parser_streaming_just_forward_text, touching `qwen_tokenizer, hermes_parser, any_chat_request`; `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py` modified +0/-201 (201 lines); hunks: -9,8 +9,6; -325,202 +323,3 @@ async def test_streaming_product_tool_call(; symbols: test_streaming_product_tool_call, qwen_tokenizer, hermes_parser, any_chat_request, touching `test_streaming_product_tool_call, qwen_tokenizer, hermes_parser`; `tests/tool_parsers/test_granite4_tool_parser.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: create_complex_input, random_chunks, tokenizer, test_tool_call_parser_complex, touching `create_complex_input, random_chunks, tokenizer`; `tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py` modified +0/-140 (140 lines); hunks: -1,18 +1,9; -38,137 +29,6 @@ def server():; symbols: server, create_complex_input, random_chunks, tokenizer, touching `server, create_complex_input, random_chunks`.
- Code diff details:
  - `tests/tool_parsers/test_hermes_tool_parser.py` added +220/-0 (220 lines); hunks: -0,0 +1,220; symbols: qwen_tokenizer, hermes_parser, any_chat_request, test_hermes_parser_streaming_just_forward_text
  - `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py` modified +0/-201 (201 lines); hunks: -9,8 +9,6; -325,202 +323,3 @@ async def test_streaming_product_tool_call(; symbols: test_streaming_product_tool_call, qwen_tokenizer, hermes_parser, any_chat_request
  - `tests/tool_parsers/test_granite4_tool_parser.py` added +147/-0 (147 lines); hunks: -0,0 +1,147; symbols: create_complex_input, random_chunks, tokenizer, test_tool_call_parser_complex
  - `tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py` modified +0/-140 (140 lines); hunks: -1,18 +1,9; -38,137 +29,6 @@ def server():; symbols: server, create_complex_input, random_chunks, tokenizer
  - `tests/entrypoints/openai/tool_parsers/conftest.py` removed +0/-12 (12 lines); hunks: -1,12 +0,0; symbols: default_tokenizer
- Key code excerpts:

```diff
diff -- tests/tool_parsers/test_hermes_tool_parser.py
@@ -0,0 +1,220 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import json
+import pytest
+from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.tokenizers import TokenizerLike
diff -- tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py
@@ -9,8 +9,6 @@
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
-from vllm.tokenizers import TokenizerLike
@@ -325,202 +323,3 @@ async def test_streaming_product_tool_call(
-@pytest.fixture
-def qwen_tokenizer() -> TokenizerLike:
-    from vllm.tokenizers import get_tokenizer
diff -- tests/tool_parsers/test_granite4_tool_parser.py
@@ -0,0 +1,147 @@
```

- Reviewed files:
  - tests: `tests/tool_parsers/test_hermes_tool_parser.py` added +220/-0; `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py` modified +0/-201; `tests/tool_parsers/test_granite4_tool_parser.py` added +147/-0; `tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py` modified +0/-140; `tests/entrypoints/openai/tool_parsers/conftest.py` removed +0/-12; `tests/tool_parsers/test_gigachat3_tool_parser.py` renamed +9/-0
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/tool_parsers/conftest.py`, `tests/entrypoints/openai/tool_parsers/test_granite4_tool_parser.py`, `tests/entrypoints/openai/tool_parsers/test_hermes_tool_parser.py`, `tests/tool_parsers/test_gigachat3_tool_parser.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35182 - [Misc] Reorganize inputs

- Link: https://github.com/vllm-project/vllm/pull/35182
- Status/date: merged / 2026-03-25
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 142 files, +1212/-1342, 6002 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Misc] Reorganize inputs"; model line: Llama 4; category: model implementation change; main diff: `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`; technical summary: Covers "[Misc] Reorganize inputs"; the main implementation surface is `vllm/multimodal/inputs.py`, `vllm/entrypoints/pooling/score/serving.py`, `vllm/entrypoints/serve/render/serving.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange, touching `VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins`; `vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item, touching `_embedding_score, _preprocess_late_interaction_item`; `vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat, touching `render_chat_request, render_chat`; `vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses, touching `__init__, _validate_generator_input, create_responses`.
- Code diff details:
  - `vllm/multimodal/inputs.py` modified +2/-162 (164 lines); hunks: -15,12 +15,11; -32,14 +31,9; symbols: VisionChunkImage, VisionChunkVideo, MultiModalDataBuiltins, PlaceholderRange
  - `vllm/entrypoints/pooling/score/serving.py` modified +36/-45 (81 lines); hunks: -35,7 +35,7; -110,12 +110,12 @@ async def _embedding_score(; symbols: _embedding_score, _preprocess_late_interaction_item
  - `vllm/entrypoints/serve/render/serving.py` modified +38/-37 (75 lines); hunks: -34,9 +34,15; -127,22 +133,22 @@ async def render_chat_request(; symbols: render_chat_request, render_chat
  - `vllm/entrypoints/openai/responses/serving.py` modified +22/-26 (48 lines); hunks: -110,7 +110,7; -269,10 +269,10 @@ def __init__(; symbols: __init__, _validate_generator_input, create_responses
  - `vllm/entrypoints/llm.py` modified +22/-22 (44 lines); hunks: -57,9 +57,9; -584,7 +584,7 @@ def wait_for_completion(; symbols: wait_for_completion, _resolve_mm_lora, beam_search
- Key code excerpts:

```diff
diff -- vllm/multimodal/inputs.py
@@ -15,12 +15,11 @@
-    final,
-from typing_extensions import NotRequired, TypeVar
+from typing_extensions import TypeVar
@@ -32,14 +31,9 @@
-    from vllm.inputs.data import _InputOptions
-    _InputOptions = dict
diff -- vllm/entrypoints/pooling/score/serving.py
@@ -35,7 +35,7 @@
-from vllm.inputs.data import ProcessorInputs, TokensPrompt, token_inputs
+from vllm.inputs import EngineInput, TokensPrompt, tokens_input
@@ -110,12 +110,12 @@ async def _embedding_score(
-        engine_prompts: list[ProcessorInputs] = []
+        engine_inputs: list[EngineInput] = []
-            engine_prompts.append(
diff -- vllm/entrypoints/serve/render/serving.py
@@ -34,9 +34,15 @@
```

- Reviewed files:
  - runtime: `vllm/multimodal/inputs.py` modified +2/-162; `vllm/entrypoints/pooling/score/serving.py` modified +36/-45; `vllm/entrypoints/serve/render/serving.py` modified +38/-37; `vllm/entrypoints/openai/responses/serving.py` modified +22/-26; `vllm/entrypoints/llm.py` modified +22/-22; `vllm/entrypoints/pooling/embed/io_processor.py` modified +20/-20
- Risk and verification: The diff ships test coverage in `tests/entrypoints/openai/chat_completion/test_chat_error.py`, `tests/entrypoints/openai/chat_completion/test_serving_chat.py`, `tests/entrypoints/openai/responses/test_serving_responses.py`, `tests/entrypoints/serve/render/test_launch_render.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #38029 - [Tool Parser][1/3] Pass tools to ToolParser constructor

- Link: https://github.com/vllm-project/vllm/pull/38029
- Status/date: merged / 2026-03-26
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 38 files, +147/-92, 858 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Tool Parser][1/3] Pass tools to ToolParser constructor"; model line: Llama 4; category: model implementation change; main diff: `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`; technical summary: Covers "[Tool Parser][1/3] Pass tools to ToolParser constructor"; the main implementation surface is `vllm/tool_parsers/abstract_tool_parser.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/step3p5_tool_parser.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab, touching `ToolParser, __init__, vocab`; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config, touching `Qwen3CoderToolParser, __init__, _reset_streaming_state`; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call, touching `__init__, setup_parser, set_tools`.
- Code diff details:
  - `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2 (16 lines); hunks: -5,13 +5,18; -30,6 +35,8; symbols: ToolParser, __init__, vocab
  - `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7 (12 lines); hunks: -10,7 +10,6; -23,15 +22,16; symbols: Qwen3CoderToolParser, __init__, _reset_streaming_state, _get_arguments_config
  - `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6 (11 lines); hunks: -11,7 +11,6; -23,7 +22,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5 (10 lines); hunks: -11,7 +11,6; -24,6 +23,7; symbols: __init__, setup_parser, set_tools, _reset_xml_parser_after_tool_call
  - `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2 (9 lines); hunks: -17,6 +17,7; -47,8 +48,12 @@ class Llama4PythonicToolParser(ToolParser):; symbols: Llama4PythonicToolParser, __init__
- Key code excerpts:

```diff
diff -- vllm/tool_parsers/abstract_tool_parser.py
@@ -5,13 +5,18 @@
+from typing import TypeAlias
+from openai.types.responses.tool import Tool as ResponsesTool
-from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
+from vllm.entrypoints.openai.chat_completion.protocol import (
+    ChatCompletionRequest,
+    ChatCompletionToolsParam,
diff -- vllm/tool_parsers/qwen3coder_tool_parser.py
@@ -10,7 +10,6 @@
-    ChatCompletionToolsParam,
@@ -23,15 +22,16 @@
+    Tool,
-    def __init__(self, tokenizer: TokenizerLike):
-        super().__init__(tokenizer)
+    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
diff -- vllm/tool_parsers/step3p5_tool_parser.py
@@ -11,7 +11,6 @@
```

- Reviewed files:
  - runtime: `vllm/tool_parsers/abstract_tool_parser.py` modified +14/-2; `vllm/tool_parsers/qwen3coder_tool_parser.py` modified +5/-7; `vllm/tool_parsers/step3p5_tool_parser.py` modified +5/-6; `vllm/tool_parsers/qwen3xml_tool_parser.py` modified +5/-5; `vllm/tool_parsers/llama4_pythonic_tool_parser.py` modified +7/-2; `vllm/tool_parsers/llama_tool_parser.py` modified +7/-2
- Risk and verification: Runtime changes concentrate in `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/entrypoints/openai/engine/serving.py`, `vllm/entrypoints/openai/parser/responses_parser.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #35949 - [MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase

- Link: https://github.com/vllm-project/vllm/pull/35949
- Status/date: merged / 2026-04-20
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +325/-702, 2430 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`; technical summary: Covers "[MoE Refactor] Move the shared/fused expert output sum into MoERunnerBase"; the main implementation surface is `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/exaone_moe.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake, touching `_resolve_layer_name, _moe_forward, _moe_forward_shared`; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__, touching `FusedMoE, __init__`; `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward, touching `__init__, forward`; `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights, touching `__init__, forward, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86 (261 lines); hunks: -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:; -113,7 +115,7 @@ def _moe_forward_shared(; symbols: _resolve_layer_name, _moe_forward, _moe_forward_shared, _moe_forward_shared_fake
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32 (60 lines); hunks: -230,11 +230,18 @@ class FusedMoE(PluggableLayer):; -246,7 +253,6 @@ def __init__(; symbols: FusedMoE, __init__
  - `vllm/model_executor/models/exaone_moe.py` modified +18/-28 (46 lines); hunks: -31,6 +31,7; -116,12 +117,26 @@ def __init__(; symbols: __init__, forward
  - `vllm/model_executor/models/kimi_linear.py` modified +20/-26 (46 lines); hunks: -11,11 +11,10; -132,12 +131,25 @@ def __init__(; symbols: __init__, forward, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +5/-30 (35 lines); hunks: -100,7 +100,7 @@ def __init__(; -170,7 +170,6 @@ def __init__(; symbols: __init__, forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py
@@ -81,15 +81,17 @@ def _resolve_layer_name(layer_name: str | LayerName) -> str:
-# the runner's 'forward_dispatch' method.
+# the runner's '_forward_dispatch' method.
+# These functions should never be called directly since they do not
+# include all the functionality of the MoE layer.
-    return layer.runner.forward_dispatch(
+    return layer.runner._forward_dispatch(
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -230,11 +230,18 @@ class FusedMoE(PluggableLayer):
-        reduce_results: Whether to all_reduce on the output of the layer
+        routed_scaling_factor: A scaling factor that is applied to the topk_weights
+                               by the router or the output of the layer depending
+                               on the value of `apply_routed_scale_to_output`
+        apply_routed_scale_to_output: Determine whether or not `routed_scaling_factor`
+                                      is applied to the topk_weights or to the experts
diff -- vllm/model_executor/models/exaone_moe.py
@@ -31,6 +31,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/runner/moe_runner_base.py` modified +175/-86; `vllm/model_executor/layers/fused_moe/layer.py` modified +28/-32; `vllm/model_executor/models/exaone_moe.py` modified +18/-28; `vllm/model_executor/models/kimi_linear.py` modified +20/-26; `vllm/model_executor/models/AXK1.py` modified +5/-30; `vllm/model_executor/models/ernie45_vl_moe.py` modified +5/-30
- Risk and verification: The diff ships test coverage in `tests/compile/passes/test_vllm_fusion_pattern_matcher_pass.py`, `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #35782 - [MoE Refactor] Remove SharedFusedMoE class

- Link: https://github.com/vllm-project/vllm/pull/35782
- Status/date: merged / 2026-04-21
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 33 files, +112/-141, 926 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Remove SharedFusedMoE class"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[MoE Refactor] Remove SharedFusedMoE class"; the main implementation surface is `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`, `vllm/model_executor/models/afmoe.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward, touching `SharedFusedMoE, forward`; `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping, touching `__init__, make_empty_intermediate_tensors, get_expert_mapping`; `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights, touching `__init__, load_moe_expert_weights, load_weights`; `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights, touching `__init__, compute_logits, get_expert_mapping`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25 (25 lines); hunks: -1,25 +0,0; symbols: SharedFusedMoE, forward
  - `vllm/model_executor/models/afmoe.py` modified +5/-5 (10 lines); hunks: -18,7 +18,7; -124,8 +124,8 @@ def __init__(; symbols: __init__, make_empty_intermediate_tensors, get_expert_mapping
  - `vllm/model_executor/models/llama4.py` modified +5/-5 (10 lines); hunks: -36,7 +36,7; -127,7 +127,7 @@ def __init__(self, vllm_config: VllmConfig, prefix: str = ""):; symbols: __init__, load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/AXK1.py` modified +4/-4 (8 lines); hunks: -42,7 +42,7; -163,7 +163,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/deepseek_v2.py` modified +4/-4 (8 lines); hunks: -48,9 +48,9; -311,7 +311,7 @@ def __init__(; symbols: __init__, compute_logits, get_expert_mapping, load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -1,25 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import torch
-from vllm.model_executor.layers.fused_moe.layer import FusedMoE
-# TODO(bnell): Remove this entirely
-class SharedFusedMoE(FusedMoE):
diff -- vllm/model_executor/models/afmoe.py
@@ -18,7 +18,7 @@
-from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
+from vllm.model_executor.layers.fused_moe import FusedMoE
@@ -124,8 +124,8 @@ def __init__(
-        # Routed experts using SharedFusedMoE
-        self.experts = SharedFusedMoE(
+        # Routed experts using FusedMoE
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,7 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` removed +0/-25; `vllm/model_executor/models/afmoe.py` modified +5/-5; `vllm/model_executor/models/llama4.py` modified +5/-5; `vllm/model_executor/models/AXK1.py` modified +4/-4; `vllm/model_executor/models/deepseek_v2.py` modified +4/-4; `vllm/model_executor/models/ernie45_moe.py` modified +4/-4
- Risk and verification: The diff ships test coverage in `tests/kernels/moe/test_moe_layer.py`, `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #40671 - [MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping

- Link: https://github.com/vllm-project/vllm/pull/40671
- Status/date: merged / 2026-04-23
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 53 files, +254/-98, 1073 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`; technical summary: Covers "[MoE Refactor] Rename FusedMoE.make_expert_params_mapping to fused_moe_make_expert_params_mapping"; the main implementation surface is `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/glm4_moe_lite.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping, touching `extra_repr, fused_moe_make_expert_params_mapping`; `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights, touching `load_moe_expert_weights, load_weights`; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits, touching `make_empty_intermediate_tensors, get_expert_mapping, load_weights`; `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights, touching `compute_logits, get_expert_mapping, load_weights`.
- Code diff details:
  - `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0 (19 lines); hunks: -1618,6 +1618,25 @@ def extra_repr(self) -> str:; symbols: extra_repr, fused_moe_make_expert_params_mapping
  - `vllm/model_executor/models/llama4.py` modified +7/-4 (11 lines); hunks: -36,7 +36,10; -414,7 +417,7 @@ def load_moe_expert_weights(; symbols: load_moe_expert_weights, load_weights
  - `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4 (10 lines); hunks: -41,7 +41,9; -308,7 +310,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping, load_weights, compute_logits
  - `vllm/model_executor/models/AXK1.py` modified +6/-3 (9 lines); hunks: -42,7 +42,10; -916,7 +919,7 @@ def compute_logits(; symbols: compute_logits, get_expert_mapping, load_weights
  - `vllm/model_executor/models/afmoe.py` modified +5/-2 (7 lines); hunks: -18,7 +18,10; -479,7 +482,7 @@ def make_empty_intermediate_tensors(; symbols: make_empty_intermediate_tensors, get_expert_mapping
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/fused_moe/layer.py
@@ -1618,6 +1618,25 @@ def extra_repr(self) -> str:
+# This is a temporary forwarding method which will be removed/modified layer.
+def fused_moe_make_expert_params_mapping(
+    model: torch.nn.Module,
+    ckpt_gate_proj_name: str,
+    ckpt_down_proj_name: str,
+    ckpt_up_proj_name: str,
diff -- vllm/model_executor/models/llama4.py
@@ -36,7 +36,10 @@
-from vllm.model_executor.layers.fused_moe import FusedMoE
+from vllm.model_executor.layers.fused_moe import (
+    FusedMoE,
+    fused_moe_make_expert_params_mapping,
+)
@@ -414,7 +417,7 @@ def load_moe_expert_weights(
diff -- vllm/model_executor/models/glm4_moe_lite.py
@@ -41,7 +41,9 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/fused_moe/layer.py` modified +19/-0; `vllm/model_executor/models/llama4.py` modified +7/-4; `vllm/model_executor/models/glm4_moe_lite.py` modified +6/-4; `vllm/model_executor/models/AXK1.py` modified +6/-3; `vllm/model_executor/models/afmoe.py` modified +5/-2; `vllm/model_executor/models/bailing_moe.py` modified +5/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/layers/fused_moe/__init__.py`, `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/model_executor/models/AXK1.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #42280 - [Model] Fix missing `maybe_prefix`

- Link: https://github.com/vllm-project/vllm/pull/42280
- Status/date: merged / 2026-05-11
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 25 files, +49/-29, 302 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Model] Fix missing `maybe_prefix`"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`; technical summary: Covers "[Model] Fix missing `maybe_prefix`"; the main implementation surface is `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/cohere_asr.py`, `vllm/model_executor/models/hunyuan_v1.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__, touching `__init__`; `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__, touching `__init__`; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, touching `__init__`.
- Code diff details:
  - `vllm/model_executor/models/arcee.py` modified +6/-2 (8 lines); hunks: -45,6 +45,7; -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:; symbols: __init__
  - `vllm/model_executor/models/cohere_asr.py` modified +3/-2 (5 lines); hunks: -64,7 +64,7; -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1 (5 lines); hunks: -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1 (4 lines); hunks: -198,7 +198,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1 (4 lines); hunks: -318,7 +318,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/arcee.py
@@ -45,6 +45,7 @@
+    maybe_prefix,
@@ -367,7 +368,10 @@ def __init__(self, *, vllm_config, prefix: str = "") -> None:
-        self.model = ArceeModel(vllm_config=vllm_config, prefix=f"{prefix}.model")
+        self.model = ArceeModel(
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "model"),
diff -- vllm/model_executor/models/cohere_asr.py
@@ -64,7 +64,7 @@
-from .utils import AutoWeightsLoader, WeightsMapper, make_layers
+from .utils import AutoWeightsLoader, WeightsMapper, make_layers, maybe_prefix
@@ -1717,7 +1717,8 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
-            vllm_config=vllm_config, prefix=f"{prefix}.decoder"
+            vllm_config=vllm_config,
+            prefix=maybe_prefix(prefix, "decoder"),
diff -- vllm/model_executor/models/hunyuan_v1.py
@@ -930,7 +930,10 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/arcee.py` modified +6/-2; `vllm/model_executor/models/cohere_asr.py` modified +3/-2; `vllm/model_executor/models/hunyuan_v1.py` modified +4/-1; `vllm/model_executor/models/deepseek_eagle.py` modified +3/-1; `vllm/model_executor/models/deepseek_eagle3.py` modified +3/-1; `vllm/model_executor/models/granite_speech.py` modified +2/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/arcee.py`, `vllm/model_executor/models/aria.py`, `vllm/model_executor/models/blip2.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43167 - Remove KV cache scale boilerplate from model weight loading methods

- Link: https://github.com/vllm-project/vllm/pull/43167
- Status/date: merged / 2026-06-05
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 56 files, +88/-731, 1251 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "Remove KV cache scale boilerplate from model weight loading methods"; model line: Llama 4; category: docs/tests/CI; main diff: `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`; technical summary: Covers "Remove KV cache scale boilerplate from model weight loading methods"; the main implementation surface is `tests/model_executor/test_eagle_quantization.py`, `vllm/model_executor/models/gpt_oss.py`, `vllm/model_executor/layers/quantization/kv_cache.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name, touching `test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale`; `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader, touching `_get_moe_weight_dtype, kv_cache_scale_loader`; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod, touching `KVCacheScaleParameter, __new__, weight_loader`; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter, touching `get_quant_method, get_cache_scale, get_cache_scale_mapper`.
- Code diff details:
  - `tests/model_executor/test_eagle_quantization.py` modified +0/-56 (56 lines); hunks: -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, d...; -183,33 +157,3 @@ def test_eagle3_lm_head_receives_quant_config():; symbols: test_fc_layer_quant_config_usage, test_kv_cache_scale_name_handling, test_kv_cache_scale_name_no_scale, test_maybe_remap_kv_scale_name
  - `vllm/model_executor/models/gpt_oss.py` modified +0/-46 (46 lines); hunks: -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:; symbols: _get_moe_weight_dtype, kv_cache_scale_loader
  - `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4 (32 lines); hunks: -15,6 +15,30; -37,11 +61,11 @@ def create_weights(self, layer: torch.nn.Module):; symbols: KVCacheScaleParameter, __new__, weight_loader, BaseKVCacheMethod
  - `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19 (31 lines); hunks: -207,25 +207,18 @@ def get_quant_method(; symbols: get_quant_method, get_cache_scale, get_cache_scale_mapper, CopyNumelCounter
  - `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20 (30 lines); hunks: -646,26 +646,16 @@ def get_scheme(; symbols: get_scheme, get_cache_scale, get_cache_scale_mapper, QuarkLinearMethod
- Key code excerpts:

```diff
diff -- tests/model_executor/test_eagle_quantization.py
@@ -100,32 +100,6 @@ def test_fc_layer_quant_config_usage(default_vllm_config, dist_init, device) ->
-def test_kv_cache_scale_name_handling():
-    # Mock a quant config that supports cache scales
-    mock_quant_config = Mock()
-    mock_quant_config.get_cache_scale = Mock(return_value="layers.0.self_attn.kv_scale")
-    # Condition check in load_weights
-    name = "layers.0.self_attn.k_proj.weight"
diff -- vllm/model_executor/models/gpt_oss.py
@@ -635,52 +635,6 @@ def _get_moe_weight_dtype(layer_id: int = 0) -> str | None:
-            def kv_cache_scale_loader(
-                quant_config: QuantizationConfig,
-                name: str,
-                params_dict: dict[str, typing.Any],
-                weight: torch.Tensor,
-                default_weight_loader: Callable[..., None],
diff -- vllm/model_executor/layers/quantization/kv_cache.py
@@ -15,6 +15,30 @@
```

- Reviewed files:
  - tests: `tests/model_executor/test_eagle_quantization.py` modified +0/-56
  - runtime: `vllm/model_executor/models/gpt_oss.py` modified +0/-46; `vllm/model_executor/layers/quantization/kv_cache.py` modified +28/-4; `vllm/model_executor/layers/quantization/fp8.py` modified +12/-19; `vllm/model_executor/layers/quantization/quark/quark.py` modified +10/-20; `vllm/model_executor/models/llama4.py` modified +3/-18; `vllm/model_executor/models/glm_ocr_mtp.py` modified +4/-13
- Risk and verification: The diff ships test coverage in `tests/model_executor/test_eagle_quantization.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39419 - [SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding

- Link: https://github.com/vllm-project/vllm/pull/39419
- Status/date: merged / 2026-06-10
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 7 files, +53/-39, 169 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; model line: Llama 4; category: model implementation change; main diff: `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`; technical summary: Covers "[SpecDecode] Reduce TP communication for large-vocab draft models speculative decoding"; the main implementation surface is `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama4_eagle.py`, `vllm/model_executor/models/qwen3.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin, touching `supports_any_eagle, LocalArgmaxMixin, get_top_tokens`; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform, touching `forward, get_top_tokens, load_weights`; `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM, touching `__init__, Qwen3ForCausalLM`; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__, touching `load_weights, Eagle3DeepseekV2ForCausalLM, __init__`.
- Code diff details:
  - `vllm/model_executor/models/interfaces.py` modified +35/-0 (35 lines); hunks: -1282,6 +1282,41 @@ def supports_any_eagle(; symbols: supports_any_eagle, LocalArgmaxMixin, get_top_tokens, EagleModelMixin
  - `vllm/model_executor/models/llama4_eagle.py` modified +0/-17 (17 lines); hunks: -208,23 +208,6 @@ def forward(; symbols: forward, get_top_tokens, load_weights, transform
  - `vllm/model_executor/models/qwen3.py` modified +8/-2 (10 lines); hunks: -48,7 +48,13; -259,7 +265,7 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str =...; symbols: __init__, Qwen3ForCausalLM
  - `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1 (3 lines); hunks: -31,6 +31,7; -309,7 +310,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, Eagle3DeepseekV2ForCausalLM, __init__
  - `vllm/model_executor/models/llama.py` modified +2/-1 (3 lines); hunks: -62,6 +62,7; -487,7 +488,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, LlamaForCausalLM
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/interfaces.py
@@ -1282,6 +1282,41 @@ def supports_any_eagle(
+class LocalArgmaxMixin:
+    """Mixin for draft model heads in speculative decoding.
+    Provides a D2T-aware ``get_top_tokens`` that preserves the
+    local-argmax communication reduction even when the draft vocabulary
+    is smaller than the target vocabulary.
+    When ``draft_id_to_target_id`` is present (shape ``(draft_vocab_size,)``,
diff -- vllm/model_executor/models/llama4_eagle.py
@@ -208,23 +208,6 @@ def forward(
-    def get_top_tokens(
-        self,
-        hidden_states: torch.Tensor,
-    ) -> torch.Tensor:
-        """Vocab-parallel argmax without all-gathering full logits.
-        Falls back to full logits when draft_id_to_target_id remapping is
diff -- vllm/model_executor/models/qwen3.py
@@ -48,7 +48,13 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/interfaces.py` modified +35/-0; `vllm/model_executor/models/llama4_eagle.py` modified +0/-17; `vllm/model_executor/models/qwen3.py` modified +8/-2; `vllm/model_executor/models/deepseek_eagle3.py` modified +2/-1; `vllm/model_executor/models/llama.py` modified +2/-1; `vllm/model_executor/models/qwen3_5_mtp.py` modified +2/-1
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/deepseek_eagle3.py`, `vllm/model_executor/models/interfaces.py`, `vllm/model_executor/models/llama.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #45047 - [Bugfix] Fix Llama4 weight loading

- Link: https://github.com/vllm-project/vllm/pull/45047
- Status/date: merged / 2026-06-10
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `fa8c868a3c64`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 4 files, +18/-6, 92 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Fix Llama4 weight loading"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Bugfix] Fix Llama4 weight loading"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +5/-1 (6 lines); hunks: -52,7 +52,10; -1021,6 +1024,7 @@ def _handle_expert_scale_broadcasting(; symbols: _handle_expert_scale_broadcasting, touching `_handle_expert_scale_broadcasting`; `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -662,6 +663,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights, touching `load_weights`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +5/-1 (6 lines); hunks: -52,7 +52,10; -1021,6 +1024,7 @@ def _handle_expert_scale_broadcasting(; symbols: _handle_expert_scale_broadcasting
  - `vllm/model_executor/models/llama4.py` modified +2/-0 (2 lines); hunks: -51,6 +51,7; -662,6 +663,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Te...; symbols: load_weights
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -52,7 +52,10 @@
-from vllm.model_executor.model_loader.weight_utils import default_weight_loader
+from vllm.model_executor.model_loader.weight_utils import (
+    default_weight_loader,
+    maybe_remap_moe_expert_param_name,
+)
@@ -1021,6 +1024,7 @@ def _handle_expert_scale_broadcasting(
diff -- vllm/model_executor/models/llama4.py
@@ -51,6 +51,7 @@
+    maybe_remap_moe_expert_param_name,
@@ -662,6 +663,7 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
+                    name = maybe_remap_moe_expert_param_name(name, params_dict)
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +5/-1; `vllm/model_executor/models/llama4.py` modified +2/-0
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/model_loader/weight_utils.py`, `vllm/model_executor/models/lfm2_moe.py`, `vllm/model_executor/models/llama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #40660 - [MM][Perf][CG] Support ViT full cudagraphs for mllama4

- Link: https://github.com/vllm-project/vllm/pull/40660
- Status/date: merged / 2026-06-12
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/mllama4.py`; associated commits `39dee1114a2c`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 5 files, +193/-14, 291 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support ViT full cudagraphs for mllama4"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/mllama4.py`; technical summary: Covers "[MM][Perf][CG] Support ViT full cudagraphs for mllama4"; the main implementation surface is `vllm/model_executor/models/mllama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +160/-12 (172 lines); hunks: -19,7 +19,7; -78,6 +78,7; symbols: Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata, get_image_patches_per_chunk, touching `Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +160/-12 (172 lines); hunks: -19,7 +19,7; -78,6 +78,7; symbols: Llama4ImagePatchInputs, Llama4ForConditionalGeneration, update_physical_experts_metadata, get_image_patches_per_chunk
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -19,7 +19,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -78,6 +78,7 @@
+    SupportsEncoderCudaGraph,
@@ -105,7 +106,7 @@ class Llama4ImagePatchInputs(TensorSchema):
-    The number of total patches for each image in the batch.
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +160/-12
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/models/utils.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #39612 - [Migration] Migrate GGUF quantization support to plugin

- Link: https://github.com/vllm-project/vllm/pull/39612
- Status/date: merged / 2026-06-12
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 57 files, +71/-9047, 9824 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Migration] Migrate GGUF quantization support to plugin"; model line: Llama 4; category: docs/tests/CI; main diff: `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`, `tests/models/test_gguf_download.py`; technical summary: Covers "[Migration] Migrate GGUF quantization support to plugin"; the main implementation surface is `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/model_loader/gguf_loader.py`, `tests/models/test_gguf_download.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/layers/quantization/gguf.py` removed +0/-690 (690 lines); hunks: -1,690 +0,0; symbols: GGUFConfig, for, __init__, __repr__, touching `GGUFConfig, for, __init__`; `vllm/model_executor/model_loader/gguf_loader.py` removed +0/-453 (453 lines); hunks: -1,453 +0,0; symbols: GGUFModelLoader, __init__, _prepare_weights, _get_all_gguf_files, touching `GGUFModelLoader, __init__, _prepare_weights`; `tests/models/test_gguf_download.py` removed +0/-224 (224 lines); hunks: -1,224 +0,0; symbols: TestGGUFDownload, test_download_gguf_single_file, test_download_gguf_sharded_files, test_download_gguf_subdir, touching `TestGGUFDownload, test_download_gguf_single_file, test_download_gguf_sharded_files`; `vllm/model_executor/model_loader/weight_utils.py` modified +0/-167 (167 lines); hunks: -54,11 +54,6; -250,10 +245,6 @@ def get_quant_config(; symbols: get_quant_config, get_sparse_attention_config, download_gguf, download_weights_from_hf, touching `get_quant_config, get_sparse_attention_config, download_gguf`.
- Code diff details:
  - `vllm/model_executor/layers/quantization/gguf.py` removed +0/-690 (690 lines); hunks: -1,690 +0,0; symbols: GGUFConfig, for, __init__, __repr__
  - `vllm/model_executor/model_loader/gguf_loader.py` removed +0/-453 (453 lines); hunks: -1,453 +0,0; symbols: GGUFModelLoader, __init__, _prepare_weights, _get_all_gguf_files
  - `tests/models/test_gguf_download.py` removed +0/-224 (224 lines); hunks: -1,224 +0,0; symbols: TestGGUFDownload, test_download_gguf_single_file, test_download_gguf_sharded_files, test_download_gguf_subdir
  - `vllm/model_executor/model_loader/weight_utils.py` modified +0/-167 (167 lines); hunks: -54,11 +54,6; -250,10 +245,6 @@ def get_quant_config(; symbols: get_quant_config, get_sparse_attention_config, download_gguf, download_weights_from_hf
  - `vllm/model_executor/layers/linear.py` modified +1/-96 (97 lines); hunks: -5,7 +5,7; -360,19 +360,6 @@ def __init__(; symbols: __init__, weight_loader
- Key code excerpts:

```diff
diff -- vllm/model_executor/layers/quantization/gguf.py
@@ -1,690 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from collections.abc import Mapping
-from types import MappingProxyType
-from typing import TYPE_CHECKING, Any
-if TYPE_CHECKING:
diff -- vllm/model_executor/model_loader/gguf_loader.py
@@ -1,453 +0,0 @@
-# SPDX-License-Identifier: Apache-2.0
-# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-import os
-from collections.abc import Generator
-from typing import TYPE_CHECKING, cast
-import gguf
diff -- tests/models/test_gguf_download.py
@@ -1,224 +0,0 @@
```

- Reviewed files:
  - runtime: `vllm/model_executor/layers/quantization/gguf.py` removed +0/-690; `vllm/model_executor/model_loader/gguf_loader.py` removed +0/-453; `vllm/model_executor/model_loader/weight_utils.py` modified +0/-167; `vllm/model_executor/layers/linear.py` modified +1/-96; `vllm/model_executor/layers/vocab_parallel_embedding.py` modified +8/-18; `vllm/model_executor/models/siglip.py` modified +0/-26
  - tests: `tests/models/test_gguf_download.py` removed +0/-224; `tests/plugins_tests/gguf/test_gguf_plugin_multimodal.py` renamed +6/-19
- Risk and verification: The diff ships test coverage in `requirements/test/rocm.txt`, `tests/compile/fullgraph/test_full_graph.py`, `tests/kernels/quantization/test_ggml.py`, `tests/kernels/quantization/test_gguf.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

### PR #44645 - [Bugfix] Stream Llama4 weight loading to avoid host-OOM with copy-returning loaders

- Link: https://github.com/vllm-project/vllm/pull/44645
- Status/date: merged / 2026-06-15
- Trace source: `git log --name-only -- <model-files>` found it through `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; associated commits `1801fad0ba62`; preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 2 files, +64/-72, 162 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[Bugfix] Stream Llama4 weight loading to avoid host-OOM with copy-returning loaders"; model line: Llama 4; category: bug fix; main diff: `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`; technical summary: Covers "[Bugfix] Stream Llama4 weight loading to avoid host-OOM with copy-returning loaders"; the main implementation surface is `vllm/model_executor/models/mllama4.py`, `vllm/model_executor/models/llama4.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/mllama4.py` modified +58/-70 (128 lines); hunks: -1131,66 +1131,6 @@ def _rename_weight_for_modelopt_checkpoint(self, name: st...; -1251,19 +1191,67 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: _rename_weight_for_modelopt_checkpoint, _separate_and_rename_weights, _handle_expert_scale_broadcasting, _load_other_weights, touching `_rename_weight_for_modelopt_checkpoint, _separate_and_rename_weights, _handle_expert_scale_broadcasting`; `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -798,10 +798,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, permute_qk_weight_for_rotary, touching `load_weights, permute_qk_weight_for_rotary`.
- Code diff details:
  - `vllm/model_executor/models/mllama4.py` modified +58/-70 (128 lines); hunks: -1131,66 +1131,6 @@ def _rename_weight_for_modelopt_checkpoint(self, name: st...; -1251,19 +1191,67 @@ def load_weights(self, weights: Iterable[tuple[str, torc...; symbols: _rename_weight_for_modelopt_checkpoint, _separate_and_rename_weights, _handle_expert_scale_broadcasting, _load_other_weights
  - `vllm/model_executor/models/llama4.py` modified +6/-2 (8 lines); hunks: -798,10 +798,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch....; symbols: load_weights, permute_qk_weight_for_rotary
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/mllama4.py
@@ -1131,66 +1131,6 @@ def _rename_weight_for_modelopt_checkpoint(self, name: str) -> str:
-    def _separate_and_rename_weights(
-        self, weights: Iterable[tuple[str, torch.Tensor]]
-    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
-        """Rename weights and separate them into language_model and other
-        weights."""
-        language_model_weights = []
diff -- vllm/model_executor/models/llama4.py
@@ -798,10 +798,14 @@ def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
-        weights = [
+        # Use a generator (not a list comprehension) so the weights iterator is
+        # consumed lazily by AutoWeightsLoader. Materializing it here would hold
+        # the entire language-model checkpoint in host memory at once, which can
+        # OOM loaders that return private copies rather than mmap views.
+        weights = (
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/mllama4.py` modified +58/-70; `vllm/model_executor/models/llama4.py` modified +6/-2
- Risk and verification: Runtime changes concentrate in `vllm/model_executor/models/llama4.py`, `vllm/model_executor/models/mllama4.py`; regression risk is weight loading, parallel sharding, attention/MoE backend selection, and parser output.

### PR #43586 - [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR

- Link: https://github.com/vllm-project/vllm/pull/43586
- Status/date: merged / 2026-06-16
- Trace source: preserved from an explicit existing history/skill citation
- Diff scope read: GitHub Pull Request files API returned 16 files, +809/-69, 1559 readable patch lines; this card prioritizes model-related and high-change files.
- Motivation: Title: "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; model line: Llama 4; category: performance/backend optimization; main diff: `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`; technical summary: Covers "[MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR"; the main implementation surface is `vllm/model_executor/models/deepseek_ocr.py`, `docs/design/cuda_graphs_multimodal.md`, `tests/models/multimodal/generation/test_vit_cudagraph.py`. File-level evidence, code excerpts, and validation risks are preserved below.
- Key implementation: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features, touching `get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__`; `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata, touching `BudgetGraphMetadata`; `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template, touching `VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template`; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2, touching `run_tarsier2`.
- Code diff details:
  - `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5 (380 lines); hunks: -4,7 +4,7; -15,6 +15,7; symbols: get_replacement_deepseek_vl2, DeepseekOCRForCausalLM, __init__, _encode_local_features
  - `docs/design/cuda_graphs_multimodal.md` modified +63/-16 (79 lines); hunks: -2,6 +2,8; -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on...; symbols: BudgetGraphMetadata
  - `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15 (56 lines); hunks: -29,6 +29,7 @@ class VitCudagraphTestConfig:; -75,15 +76,16 @@ def step3_vl_chat_template(content: str) -> str:; symbols: VitCudagraphTestConfig, params_with_marks, step3_vl_chat_template
  - `examples/generate/multimodal/vision_language_offline.py` modified +3/-2 (5 lines); hunks: -2533,15 +2533,16 @@ def run_tarsier2(questions: list[str], modality: str) ->...; symbols: run_tarsier2
  - `vllm/model_executor/models/interfaces.py` modified +5/-0 (5 lines); hunks: -1623,6 +1623,7 @@ def postprocess_encoder_output(; -1643,6 +1644,7 @@ def prepare_encoder_cudagraph_capture_inputs(; symbols: postprocess_encoder_output, prepare_encoder_cudagraph_capture_inputs, prepare_encoder_cudagraph_replay_buffers, encoder_cudagraph_forward
- Key code excerpts:

```diff
diff -- vllm/model_executor/models/deepseek_ocr.py
@@ -4,7 +4,7 @@
-from typing import Annotated, Literal
+from typing import Annotated, Any, Literal
@@ -15,6 +15,7 @@
+    SupportsEncoderCudaGraph,
@@ -52,6 +53,7 @@
+    IMAGE_SIZE,
diff -- docs/design/cuda_graphs_multimodal.md
@@ -2,6 +2,8 @@
+For two-tower vision encoders (e.g., DeepSeek-OCR's SAM + CLIP with dynamic tiling), a **dual-path graph** mode captures two independent sets of CUDA graphs — one for the global i
@@ -11,6 +13,8 @@ Vision encoder inference incurs CUDA kernel launch overhead on the host side. Th
+For two-tower vision encoders such as DeepSeek-OCR (SAM + CLIP with dynamic tiling), the global image path and local patch path have independent token profiles (272 tokens per glo
@@ -37,17 +41,57 @@ class BudgetGraphMetadata:
+When `EncoderCudaGraphConfig.enable_dual_path_graph` is `True`, the manager generates two independent budget lists — `global_token_budgets` (multiples of `global_token_per_image`)
+For dual-path models, the manager routes to `_execute_local_dual_path()`, which constrains both global and local token budgets simultaneously during packing (see [Dual-Path graph
diff -- tests/models/multimodal/generation/test_vit_cudagraph.py
@@ -29,6 +29,7 @@ class VitCudagraphTestConfig:
```

- Reviewed files:
  - runtime: `vllm/model_executor/models/deepseek_ocr.py` modified +375/-5; `vllm/model_executor/models/interfaces.py` modified +5/-0; `vllm/model_executor/models/step3_vl.py` modified +5/-0; `vllm/model_executor/models/glm4_1v.py` modified +4/-0; `vllm/model_executor/models/internvl.py` modified +4/-0
  - docs: `docs/design/cuda_graphs_multimodal.md` modified +63/-16; `examples/generate/multimodal/vision_language_offline.py` modified +3/-2
  - tests: `tests/models/multimodal/generation/test_vit_cudagraph.py` modified +41/-15
- Risk and verification: The diff ships test coverage in `tests/models/multimodal/generation/test_vit_cudagraph.py`, `tests/v1/cudagraph/test_encoder_cudagraph.py`; future changes in this area should rerun those tests plus a minimal launch or accuracy smoke.

## Gap-Closure Notes

- Acceptance rule: every PR card must keep trace source, diff scope, implementation notes, code excerpts, reviewed files, and verification risk.
- If new model files fall outside the current filters, add the file filter first and rerun the same `git log --name-only -- <model-files>` trace.
