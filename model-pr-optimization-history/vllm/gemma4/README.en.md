# vLLM Gemma 4 Support and PR History

This note tracks the vLLM runtime, key PRs, and cookbook-facing touchpoints for Gemma 4.

- Status: supported on current mainline

## Key Conclusions

- Gemma 4 is a genuinely multi-surface family: text, MoE, multimodal, tool calling, and speculative decoding all matter.
- Fast prefill, quantized MoE, and tool-parser correctness are the main areas that changed rapidly after bring-up.

## Main Runtime Surfaces

- `vllm/vllm/model_executor/models/gemma4.py`
- `vllm/vllm/model_executor/models/gemma4_mm.py`

## Landed PRs

- [#38826](https://github.com/vllm-project/vllm/pull/38826) `Implement Google Gemma 4 architecture support`: Initial Gemma 4 text/MoE/multimodal landing.
- [#38879](https://github.com/vllm-project/vllm/pull/38879) `Enable Fast Prefill Optimization`: Added YOCO KV-sharing based fast prefill for Gemma4.
- [#39045](https://github.com/vllm-project/vllm/pull/39045) `Support quantized MoE`: Extended Gemma4 to quantized MoE checkpoints.
- [#38844](https://github.com/vllm-project/vllm/pull/38844) `Enable Gemma4ForCausalLM to load LoRA adapters correctly`: Fixed adapter naming/load behavior.
- [#39450](https://github.com/vllm-project/vllm/pull/39450) `Add Gemma4 Eagle3 support`: Enabled speculative decode for Gemma4.

## Matching Skill

- `skills/model-optimization/vllm/vllm-gemma4-optimization/SKILL.md`
- `skills/model-optimization/vllm/vllm-gemma4-optimization/references/pr-history.md`

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Gemma 4` against `vllm-project/vllm` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-04-02 | [#38826](https://github.com/vllm-project/vllm/pull/38826) | merged | feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use) | model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py` |
| 2026-04-02 | [#38844](https://github.com/vllm-project/vllm/pull/38844) | merged | [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly | model wrapper, scheduler/runtime, tests/benchmarks | `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py` |
| 2026-04-03 | [#38879](https://github.com/vllm-project/vllm/pull/38879) | merged | [Gemma4] Enable Fast Prefill Optimization | model wrapper, scheduler/runtime | `vllm/model_executor/models/gemma4.py` |
| 2026-04-05 | [#39045](https://github.com/vllm-project/vllm/pull/39045) | merged | [Gemma4] Support quantized MoE | model wrapper, scheduler/runtime | `vllm/model_executor/models/gemma4.py` |
| 2026-04-09 | [#39450](https://github.com/vllm-project/vllm/pull/39450) | merged | Add Gemma4 Eagle3 support | model wrapper, attention/backend, scheduler/runtime, docs/config | `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py` |

### File-level PR diff reading notes

### PR #38826 - feat(models): implement Google Gemma 4 architecture support (MoE, Multimodal, Reasoning, Tool-Use)

- Link: https://github.com/vllm-project/vllm/pull/38826
- Status/date: `merged`, created 2026-04-02, merged 2026-04-02; author `lucianommartins`.
- Diff scope read: `20` files, `+5051/-1`; areas: model wrapper, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: spec, config, attention, cache, expert, kv, lora, moe, processor, quant.
- Code diff details:
  - `vllm/model_executor/models/gemma4_mm.py` added +1341/-0 (1341 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: Gemma4ImagePixelInputs, Gemma4AudioInputs, Gemma4VideoInputs, Gemma4ProcessingInfo
  - `vllm/model_executor/models/gemma4.py` added +1239/-0 (1239 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _get_text_config, Gemma4MLP, __init__, forward
  - `vllm/tool_parsers/gemma4_tool_parser.py` added +724/-0 (724 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array, Gemma4ToolParser
  - `tests/tool_parsers/test_gemma4_tool_parser.py` added +504/-0 (504 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: mock_tokenizer, parser, mock_request, TestParseGemma4Args:
  - `vllm/model_executor/models/gemma4_utils.py` added +292/-0 (292 lines); hunks: +# SPDX-License-Identifier: Apache-2.0; symbols: parse_thinking_output, _strip_thought_label, _clean_answer, _parse_tool_arguments
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; keywords observed in patches: spec, config, attention, cache, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/gemma4_mm.py`, `vllm/model_executor/models/gemma4.py`, `vllm/tool_parsers/gemma4_tool_parser.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #38844 - [Gemma4][Bugfix]: Enable Gemma4ForCasualLM to load lora adapters correctly

- Link: https://github.com/vllm-project/vllm/pull/38844
- Status/date: `merged`, created 2026-04-02, merged 2026-04-11; author `ShubyM`.
- Diff scope read: `2` files, `+40/-0`; areas: model wrapper, scheduler/runtime, tests/benchmarks; keywords: expert, lora, moe, attention, eagle, kv, test.
- Code diff details:
  - `tests/lora/test_lora_checkpoints.py` modified +23/-0 (23 lines); hunks: from vllm.lora.lora_model import LoRAModel; def test_lora_weights_mapping(baichuan_lora_files):; symbols: test_lora_weights_mapping, test_gemma4_lora_weights_mapping, test_gemma4_moe_lora_weights_mapping
  - `vllm/model_executor/models/gemma4.py` modified +17/-0 (17 lines); hunks: ); def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; symbols: load_weights, Gemma4ForCausalLM
- Optimization/support interpretation: The concrete diff surface is `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py`; keywords observed in patches: expert, lora, moe, attention, eagle, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `tests/lora/test_lora_checkpoints.py`, `vllm/model_executor/models/gemma4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #38879 - [Gemma4] Enable Fast Prefill Optimization

- Link: https://github.com/vllm-project/vllm/pull/38879
- Status/date: `merged`, created 2026-04-03, merged 2026-04-06; author `LucasWilkinson`.
- Diff scope read: `1` files, `+369/-47`; areas: model wrapper, scheduler/runtime; keywords: attention, cache, config, cuda, expert, kv, lora, scheduler.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +369/-47 (416 lines); hunks: """Gemma 4 model implementation for vLLM."""; get_tensor_model_parallel_rank,; symbols: forward, _run_decoder_layers, Gemma4SelfDecoderLayers, __init__
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/gemma4.py`; keywords observed in patches: attention, cache, config, cuda, expert, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/gemma4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #39045 - [Gemma4] Support quantized MoE

- Link: https://github.com/vllm-project/vllm/pull/39045
- Status/date: `merged`, created 2026-04-05, merged 2026-04-09; author `dsikka`.
- Diff scope read: `1` files, `+34/-14`; areas: model wrapper, scheduler/runtime; keywords: config, expert, moe, quant.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +34/-14 (48 lines); hunks: def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:; def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str; symbols: load_weights, load_weights, load_weights, _weight_iterator
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/gemma4.py`; keywords observed in patches: config, expert, moe, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/gemma4.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #39450 - Add Gemma4 Eagle3 support

- Link: https://github.com/vllm-project/vllm/pull/39450
- Status/date: `merged`, created 2026-04-09, merged 2026-04-10; author `fynnsu`.
- Diff scope read: `5` files, `+43/-10`; areas: model wrapper, attention/backend, scheduler/runtime, docs/config; keywords: eagle, config, kv, spec, cache, attention, expert, flash, lora, moe.
- Code diff details:
  - `vllm/model_executor/models/gemma4.py` modified +20/-5 (25 lines); hunks: from vllm.sequence import IntermediateTensors; def forward(; symbols: forward, Gemma4Model, Gemma4Model, __init__
  - `vllm/model_executor/models/gemma4_mm.py` modified +12/-2 (14 lines); hunks: from vllm.sequence import IntermediateTensors; def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:; symbols: forward, Gemma4ForConditionalGeneration, Gemma4ForConditionalGeneration
  - `vllm/v1/core/single_type_kv_cache_manager.py` modified +9/-3 (12 lines); hunks: def find_longest_cache_hit(; symbols: find_longest_cache_hit, get_num_skipped_tokens
  - `vllm/config/speculative.py` modified +1/-0 (1 lines); hunks: def _verify_args(self) -> Self:; symbols: _verify_args
  - `vllm/v1/spec_decode/eagle.py` modified +1/-0 (1 lines); hunks: def load_model(self, target_model: nn.Module) -> None:; symbols: load_model
- Optimization/support interpretation: The concrete diff surface is `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py`; keywords observed in patches: eagle, config, kv, spec, cache, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `vllm/model_executor/models/gemma4.py`, `vllm/model_executor/models/gemma4_mm.py`, `vllm/v1/core/single_type_kv_cache_manager.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 5; open PRs: 0.
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
