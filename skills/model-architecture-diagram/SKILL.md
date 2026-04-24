---
name: model-architecture-diagram
description: Return or generate model architecture diagrams for user-specified LLM, VLM, MoE, diffusion, OCR, and SGLang/sgl-cookbook model families. Use when the user asks for a model structure chart, architecture diagram, 模型结构图, 架构图, computation-flow diagram, or wants a rendered image link for a specific model such as DeepSeek, GLM, Qwen, Kimi, MiniMax, Step, Hunyuan, Llama, Gemma, GPT-OSS, Wan, FLUX, or Z-Image.
---

# Model Architecture Diagram

## Workflow

Always prefer an original existing diagram before generating a new one.

1. Run the bundled resolver:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "<model name>"
```

2. If the resolver returns `kind: existing`, return the raw image Markdown it prints. Preserve the source attribution line.
3. If the resolver returns `kind: generated`, return the local SVG image Markdown and the source-code notes it prints. The generated SVG is intentionally simple and readable.
4. If the user asks for internals beyond the first diagram, rerun with `--force-generate` and then read the reported SGLang source files before refining the diagram.

## Source Priority

Use sources in this order:

- `references/diagram-index.json`: original diagrams from `datawhalechina/self-llm` and `CalvinXKY/InfraTech`, stored as raw GitHub image links.
- Local SGLang source, especially `python/sglang/srt/models/` and `python/sglang/multimodal_gen/`.
- Local sgl-cookbook docs and model YAML, especially `docs/autoregressive`, `docs/diffusion`, and `data/models/src`.
- Generated fallback templates in `scripts/model_architecture_diagram.py`.

Do not copy remote image binaries into the skill. Return their raw GitHub URLs so the chat renderer can display the original image.

## Existing Diagram Rule

For a direct match, show the original image rather than redrawing it. Good direct matches include:

- DeepSeek V3/V3.2, GLM-5, Kimi K2/K2.5, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash from InfraTech.
- Hunyuan-A13B and Kimi-VL architecture/module diagrams from self-llm.

If multiple diagrams match, show all high-confidence matches up to the resolver's default limit. For example, DeepSeek V3 may return the full architecture plus MLA MHA/MQA diagrams.

## Generated Diagram Rule

When no original diagram matches:

- Generate a diagram in the same spirit as the reference repositories: top-down flow, grouped modality branches, color-coded attention/MLP/MoE blocks, and short labels.
- Mention that the image is generated from code/docs rather than an upstream original.
- Include the SGLang files and cookbook docs that should be inspected before making the diagram more exact.
- Prefer Mermaid plus SVG over hand-drawn prose. The resolver writes both.

## Useful Commands

List known original diagram aliases:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py --list-known
```

Force a generated diagram even when an original exists:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "Qwen3.5-397B-A17B" --force-generate
```

Emit JSON for automation:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "GLM-5" --format json
```

Generate a full sgl-cookbook audit gallery:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py \
  --batch-sgl-cookbook \
  --output-dir /tmp/sgl-cookbook-architecture-audit \
  --format json
```

## References

- `references/diagram-index.json`: original diagram link index and aliases.
- `references/source-notes.md`: audited source repositories, local repo paths, and style notes.
