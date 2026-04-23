# Qwen3.6 Playbook

## Symptom Map

| Symptom | First files | Likely source |
| --- | --- | --- |
| Generated command is wrong | `qwen36-deployment.jsx`, `Qwen3.6.mdx` | docs/cookbook drift |
| MTP breaks hybrid state | `qwen3_next.py`, speculative workers | shared Qwen3-Next issue |
| Multimodal input fails | Qwen VLM processors and docs examples | shared Qwen VLM issue |
| Tool calls parse incorrectly | `qwen3_coder_detector.py` | shared Qwen3-Coder parser |
| CPU offload fails | hybrid linear-attention/offload code | open `#23474` |

## Investigation Order

1. Check whether the issue reproduces on Qwen3-Next or Qwen3.5.
2. Check generated command flags from `qwen36-deployment.jsx`.
3. Run text-only reasoning and tool-call streaming.
4. Run multimodal image and video examples.
5. Enable MTP only after the baseline mode works.

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

## Change Rules

- Do not introduce a new model class just because the docs mention Qwen3.6; prove config/runtime mismatch first.
- Keep Qwen3.6 docs, command generator, and support matrix together.
- For parser fixes, validate `--reasoning-parser qwen3` plus `--tool-call-parser qwen3_coder`.
