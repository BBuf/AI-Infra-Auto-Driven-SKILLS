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

## Hosted sgl-cookbook Gallery

Do not commit the full `sgl-cookbook-model-architecture-images/` gallery into
the repository. The image set is large because it includes public original JPGs
plus generated PNG/SVG/Mermaid files, so it is hosted as a GitHub Release asset
and indexed by a GitHub issue instead.

Current hosted artifact:

- Issue index: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/issues/31
- Release page: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/tag/sgl-cookbook-architecture-images-2026-04-25-v3
- Zip download: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-04-25-v3/sgl-cookbook-model-architecture-images-2026-04-25-v3.zip
- Digest: `sha256:ac43b8f4cd7b7129288e7c4333883391a35220e6ae0e2cec5abd7d5783d5dce2`

The artifact contains 63 model directories: 62 discovered from sgl-cookbook plus
the manually audited DeepSeek-V4 support branch. It includes 11 public-original
matches, 52 generated diagram sets, `index.html`, `index.md`, `manifest.json`,
contact sheets, and `architecture-audit.md`.

The 2026-04-25-v3 artifact keeps the v2 generated-fallback refinement pass and
replaces the DeepSeek-V4 generated diagram with a dedicated visual correction:
the left spine now separates Attention RMSNorm and MoE RMSNorm lanes, the
compressed MQA panel separates Q, KV, output-projection, and sparse-compression
lanes, and the MTP draft plus node is spaced away from the V4 layer box.

How the gallery was produced:

- Each model document under local sgl-cookbook `docs/autoregressive`,
  `docs/diffusion`, and `docs/omni` was resolved with
  `scripts/model_architecture_diagram.py`.
- Direct public matches from `references/diagram-index.json` were downloaded
  unchanged from upstream raw GitHub image URLs.
- Generated diagrams were created as SVG and Mermaid by the resolver; PNG files
  are headless-Chrome screenshots of the SVGs at a 2048 x 1280 viewport.

To inspect the gallery locally:

```bash
curl -L -o /tmp/sgl-cookbook-model-architecture-images-2026-04-25-v3.zip \
  https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-04-25-v3/sgl-cookbook-model-architecture-images-2026-04-25-v3.zip
unzip -q /tmp/sgl-cookbook-model-architecture-images-2026-04-25-v3.zip -d /tmp
open /tmp/sgl-cookbook-model-architecture-images/index.html
```

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
