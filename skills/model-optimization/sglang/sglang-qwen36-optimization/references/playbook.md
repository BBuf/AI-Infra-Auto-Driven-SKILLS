# Qwen3.6 Playbook

## Required PR Reading

Before changing Qwen3.6 docs, launch commands, FP8 loading, MTP, or CPU offload behavior, read the diff-reviewed cards in `references/pr-history.md`:

- `#23034` for the initial Qwen3.6 cookbook, command generator, reasoning/tool parser flags, MTP, Mamba V2, and B200 attention backend.
- `#23467` for FP8 `modules_to_not_convert` dotted-boundary matching and fused-shard fallback.
- `#23486` for the 27B dense variant and model-size selector.
- `#23474` for the open hybrid linear-attn `--cpu-offload-gb` fix.

## Symptom Map

| Symptom | First files | Likely source |
| --- | --- | --- |
| Generated command is wrong | `qwen36-deployment.jsx`, `Qwen3.6.mdx` | docs/cookbook drift |
| MTP breaks hybrid state | `qwen36-deployment.jsx`, `qwen3_next.py`, speculative workers | Mamba V2/spec-v2 flag drift or shared Qwen3-Next issue |
| Multimodal input fails | Qwen VLM processors and docs examples | shared Qwen VLM issue |
| Tool calls parse incorrectly | `qwen3_coder_detector.py` | shared Qwen3-Coder parser |
| FP8 loads with `weight_scale_inv not found` | `quantization/utils.py` | `#23467` dotted-boundary/fused-shard skip logic |
| CPU offload fails | `utils/offloader.py`, hybrid linear-attention cached views | open `#23474` |

## Investigation Order

1. Check whether the issue reproduces on Qwen3-Next or Qwen3.5.
2. Check generated command flags from `qwen36-deployment.jsx`.
3. For FP8 load issues, inspect `modules_to_not_convert` and fused projection names before changing model code.
4. Run text-only reasoning and tool-call streaming.
5. Run multimodal image and video examples.
6. Enable MTP only after the baseline mode works, and confirm it selects Mamba V2.
7. Test `--cpu-offload-gb` only after checking the offloader alias/tied-parameter path.

## Validation Commands

Use the generated command from:

- `docs_new/src/snippets/autoregressive/qwen36-deployment.jsx`

Then test:

- text-only chat
- streaming reasoning
- streaming tool call
- image input
- video input
- MTP with `SGLANG_ENABLE_SPEC_V2=1`
- FP8 load without `weight_scale_inv not found`
- CPU offload with a hybrid model when `#23474` or equivalent logic is present

## Change Rules

- Do not introduce a new model class just because the docs mention Qwen3.6; prove config/runtime mismatch first.
- Keep Qwen3.6 docs, command generator, and support matrix together.
- For parser fixes, validate `--reasoning-parser qwen3` plus `--tool-call-parser qwen3_coder`.
