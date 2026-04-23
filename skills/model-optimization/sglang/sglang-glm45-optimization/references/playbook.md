# GLM-4.5 Playbook

## Required PR Reading

Before changing GLM-4.5/GLM-4.5-Air model loading, MoE execution, quantization, parser, or deployment docs, read the diff-reviewed cards in `references/pr-history.md`:

- `#8224` for the original GLM-4.5 text MoE, NextN/MTP, detector, and parser registration.
- `#8456`, `#12497`, and `#19106` for compressed-tensors, NVFP4, WNA16, shared-expert-ignore, and TF>=5 quantized-load behavior.
- `#8647`, `#11665`, `#11692`, `#11800`, `#11847`, `#12572`, and `#14668` for EP/TP, reduce-scatter, A2A, PP, dispatcher, symmetric-memory, and FlashInfer A2A behavior.
- `#8729`, `#8804`, `#9223`, `#12162`, `#12834`, `#12957`, `#13786`, and `#13873` for router precision, shared-expert fusion, `TopKOutput`, routed-expert capture, KTransformers, dead-code cleanup, dual-stream overlap, and shared-GLM4-MoE fusion.
- `#12456`, `#13989`, `#15333`, `#15753`, `#15754`, `#17714`, `#20543`, `#20917`, and `#23067` for GLM45/GLM47 parser split, escaped values, streaming arguments, JSON Schema inference, empty/partial tool-call robustness, reasoning tool interruption, whitespace preservation, `enable_thinking`, and `continue_final_message`.
- `#13711` and `#19728` for open hardware/platform work that can affect GLM-4.5-Air/GLM-4.5V fused-MoE or ROCm FP8 paths.

## Symptom Map

| Symptom | First files | PR trail |
| --- | --- | --- |
| New GLM4-MoE checkpoint cannot be recognized | `model_config.py`, `glm4_moe.py`, `glm4_moe_nextn.py` | `#8224`, `#11017` |
| EP+TP launch fails around shared experts | `glm4_moe.py`, MoE EP world size | `#8647`, `#8804`, `#13873` |
| Reduce-scatter or communicator correctness issue | `glm4_moe.py`, `LayerCommunicator` | `#8883`, `#11665` |
| A2A backend fails or Mooncake/DeepEP path not initialized | `glm4_moe.py`, dispatcher files | `#11692`, `#11847`, `#14668` |
| NVFP4 scale shape assertion | modelopt/NVFP4 quant loader | `#12497` |
| CompressedTensors/WNA16/AWQ-4bit serving failure | `glm4_moe_lite.py`, `deepseek_weight_loader.py`, quant ignore mapping | `#8456`, `#19106` |
| Shared-expert fusion gives wrong weights | `determine_num_fused_shared_experts`, `load_weights` | `#8804`, `#13873`, `#19106` |
| Tool-call values double-escape or lose whitespace | `glm4_moe_detector.py` | `#12456`, `#20543` |
| Streaming tool arguments arrive only at end | `glm4_moe_detector.py` | `#13989`, `#15754` |
| GLM45 reasoning merges tool call into reasoning | `reasoning_parser.py` | `#17714` |
| `/v1/responses` ignores `enable_thinking=false` | `serving_responses.py` | `#20917` |
| `continue_final_message=true` returns HTTP 500 | `Glm45Detector.__init__` | `#23067` |
| RTX Pro 6000 fused-MoE perf gap | fused-MoE config JSON | `#13711` |
| ROCm FP8 startup fails with padding mismatch | `fused_moe.py`, `fp8_kernel.py` | `#19728` |

## Investigation Order

1. Identify whether the failure is text GLM-4.5, GLM-4.5-Air, GLM-4.5V, or a later GLM4-MoE model reusing the same code.
2. Reproduce BF16 with minimal TP before enabling quantization, MTP, EP, or A2A.
3. For quantized load failures, inspect quant method, ignored modules, packed fused-module mapping, and shared-expert fusion before touching kernels.
4. For MoE correctness or latency, check reduce-scatter state, shared-expert fusion count, `TopKOutput`, and A2A backend in that order.
5. For parser bugs, split reasoning parser (`reasoning_parser.py`) from tool-call parser (`glm4_moe_detector.py`) and test streaming plus non-streaming.
6. For VLM or ROCm reports, confirm whether the patch touches shared text MoE/fused-MoE/quant/parser code; otherwise move it to the VLM/OCR skill.
7. Only profile fused-MoE configs after startup, parser, and accuracy are stable.

## Validation Checklist

- BF16 GLM-4.5 and GLM-4.5-Air startup.
- FP8, compressed-tensors, and NVFP4 load, including warning scan for skipped or missing scale tensors.
- TP-only, EP+TP, DeepEP, Mooncake, and FlashInfer A2A modes as applicable.
- Shared-expert fusion on and off, with weight remapping checked.
- MTP/NextN startup and base model output parity.
- `--reasoning-parser glm45` with normal reasoning, truncated reasoning, and `<tool_call>` interruption.
- `--tool-call-parser glm45` with escaped JSON, array/object schemas, partial streaming chunks, no-arg tools, undefined tools, and exact whitespace.
- `/v1/chat/completions` and `/v1/responses` with `enable_thinking` unset, true, and false.
- `continue_final_message=true` with trailing assistant content if #23067 or equivalent logic is present.
- Hardware-specific fused-MoE configs only on the target device named in the PR.

## Change Rules

- Do not re-enable reduce-scatter for GLM45 without reading #8883 and #11665 and providing a correctness matrix.
- Do not disable shared-expert fusion globally just to fix a single quantized checkpoint; first check whether `quant_config.ignore` marks shared experts non-quantized as in #19106.
- Keep `glm45` and `glm47` parser behavior separate; GLM-4.5/4.6 use `glm45`, GLM-4.7/5 use `glm47` only where docs/runtime say so.
- Preserve exact argument whitespace in GLM tool-call parser.
- Treat open PR cards as hypotheses until merged; cite them as pending fixes, not current-main behavior.
