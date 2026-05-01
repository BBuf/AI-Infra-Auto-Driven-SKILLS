# GLM-4.6/4.7 Optimization Playbook

Use this playbook after reading `references/pr-history.md`. Do not treat it as a replacement for PR diff review; it is the operating checklist for applying the reviewed PR history to new SGLang work.

## Model Split

| Model lane | Primary runtime | Parser contract | Main risks |
| --- | --- | --- | --- |
| GLM-4.6 | `glm4_moe.py` | `--tool-call-parser glm45`, `--reasoning-parser glm45` | shared-expert fusion, dual-stream overlap, GLM4 XML tool streaming |
| GLM-4.7 | `glm4_moe.py`, `glm4_moe_nextn.py` | `--tool-call-parser glm47`, `--reasoning-parser glm45` | GLM-4.7 no-newline tool format, NVFP4/FP8 MTP, NextN draft quant config |
| GLM-4.7-Flash | `glm4_moe_lite.py` | `--tool-call-parser glm47`, `--reasoning-parser glm45` | Lite config/import, Flash has no EAGLE implementation, compressed-tensors/AWQ, AMD/NPU backend divergence |
| GLM-4.6V / GLM-4.7V adjacent | `glm4v.py`, `glm4v_moe.py`, shared `glm4_moe.py` | usually GLM4 parser family | only include here when text MoE/shared-expert code is touched |

## Investigation Order

1. Identify the exact lane: full GLM4-MoE, NextN/MTP draft, or GLM4-MoE-Lite/Flash.
2. Reproduce BF16 or baseline quantized launch with parser flags before enabling speculative decoding.
3. Validate parser behavior:
   - GLM-4.6 uses newline form: `<tool_call>name\n...`
   - GLM-4.7 uses no-newline form: `<tool_call>name<arg_key>...`
   - reasoning still uses `glm45`
4. Validate loading:
   - `get_rope_config(config)` works for trust-remote-code configs.
   - `glm4_moe_lite` configs require Transformers >=5 behavior.
   - packed quantized modules may not expose dense `.weight`.
5. Validate draft/MTP:
   - `mtp.safetensors` is present or auto-added.
   - draft `quant_config` is preserved when checkpoint quantization is auto-detected.
   - average speculative accept length is checked, not just correctness.
6. Validate MoE optimization:
   - shared-expert fusion
   - dual-stream shared/routed expert overlap
   - A2A/FlashInfer/DeepEP dispatch
7. Validate hardware-specific paths:
   - NVIDIA CUDA graph capture and Blackwell FP4 backends
   - AMD AITER FP8 per-token path and `gfx95_quant_format`
   - NPU fused attention/QKNorm/RoPE and multi-stream paths

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| Tool arguments arrive only after full tool call | `glm4_moe_detector.py`, `glm47_moe_detector.py`, Rust parser | `#11951`, `#13989`, `#15333`, `#15520` |
| GLM-4.7 parser drops function name or arguments | `glm47_moe_detector.py` | `#15333`, `#15753`, `#15754`, `#20543` |
| Indentation in tool arguments disappears | GLM detectors | `#20543` |
| Startup fails after Transformers/rope change | `glm4_moe.py`, `glm4_moe_lite.py`, `hf_transformers_utils.py` | `#21135`, `#21851`, `#19040` |
| GLM-4.7-Flash import fails on NPU | `glm4_moe_lite.py`, `deepseek_v2.py` | `#22509`, `#17869`, `#22801` |
| CompressedTensors/AWQ Flash checkpoint lacks `.weight` | `glm4_moe_lite.py`, DeepSeek MLA loader | `#19106` |
| Shared-expert fusion gives wrong output | `glm4_moe.py`, `glm4_moe_lite.py` | `#13873`, `#14668`, `#21851`, `#19106` |
| GLM gate routing accuracy regresses | `Glm4MoeGate` | `#21660` |
| MTP accept length collapses to `1.0` | `glm4_moe_nextn.py`, loader quant config | `#17166`, `#19246`, `#22315`, `#22823` |
| AMD FP8 throughput flat after RMSNorm quant | `communicator.py`, `fp8_utils.py`, `glm4_moe.py` | `#21403`, `#21534`, `#22720` |
| NPU GLM-4.7 throughput poor | NPU backend, `glm4_moe.py`, `glm4_moe_lite.py` | `#19246`, `#17869`, `#22801` |

## PR Reading Rules For This Lane

- For every PR cited in produced documentation, open the GitHub PR metadata and diff.
- Record state, merge commit or open state, diff stats, motivation, key implementation, key code excerpts, reviewed files, and validation implications.
- If a PR is open, say so explicitly and distinguish it from merged current-main behavior.
- If a PR is superseded by a later merged fix, keep the earlier PR only as design/history context.
- Do not fill PR cards from title search, commit messages only, or generated summaries.

## Parser Validation

Minimum GLM parser suite:

- GLM-4.6 newline tool call with scalar arguments.
- GLM-4.6 streaming tool name followed by argument deltas.
- GLM-4.7 no-newline tool call.
- Complex JSON Schema with arrays, objects, nullable values, and enums.
- Empty function name and invalid tool name.
- Escaped JSON strings with literal `\\n`, quotes, and paths.
- Leading/trailing whitespace preservation in string values.
- `continue_final_message=true` with `--reasoning-parser glm45`.

## Loading and Quant Validation

Minimum GLM-4.7/Flash load matrix:

- GLM-4.7 BF16, parser flags only.
- GLM-4.7-FP8, TP8, no MTP.
- GLM-4.7-FP8 with EAGLE/NEXTN, checking accept length.
- GLM-4.7 NVFP4/modelopt FP4 on Blackwell, checking backend auto-selection.
- GLM-4.7-Flash BF16 or base quantized checkpoint.
- GLM-4.7-Flash compressed-tensors/AWQ or REAP checkpoint if the change touches packed quant modules.

Important checks:

- Do not assume packed quant modules have `.weight`.
- Do not enable shared-expert fusion when checkpoint quant config marks shared experts ignored/non-quantized.
- Do not route GLM-4.7-Flash through EAGLE unless a later PR explicitly implements it.
- Confirm `Glm4MoeForCausalLMNextN` sees the same quantization family as the target model unless the test intentionally covers NPU unquant draft.

## MoE Optimization Validation

Run these toggles independently before combining them:

- shared-expert fusion (`#13873`)
- CUDA dual-stream shared/routed overlap (`#13786`)
- FlashInfer A2A dispatcher (`#14668`)
- NPU dual-stream/fused QKNorm-RoPE (`#19246`, `#22801`)
- AMD fused RMSNorm + FP8 per-token quant (`#21403`)
- GLM gate FP32 projection (`#21660`)

For each toggle, capture:

- launch command and hardware
- quantization and TP/EP
- parser flags
- accuracy metric
- output throughput / ITL / TTFT as relevant
- profile or profiler summary if it is a performance claim

## Production Blog Toggles

The LMSYS / Novita production blog lists several optimizations that can be confused with each other:

- shared-expert fusion
- QK-Norm-RoPE fusion
- async transfer
- suffix decoding
- speculative EAGLE/NEXTN

When producing docs or debugging performance, keep these as separate rows in the benchmark matrix. A speedup that requires suffix decoding should not be described as a shared-expert-fusion speedup.

## Documentation Checklist

When updating model history docs:

- Include merged PR cards and open PR candidates separately.
- Mention `glm47` tool parser and `glm45` reasoning parser explicitly for GLM-4.7.
- Mention `glm4_moe_lite` and `Glm4MoeLiteForCausalLM` explicitly for Flash.
- Explain whether each code excerpt came from the full model, Lite model, NextN, parser, quant loader, or hardware backend.
- Add a validation sentence for every PR.
