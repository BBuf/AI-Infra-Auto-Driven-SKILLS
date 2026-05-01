---
name: model-architecture-diagram
description: Return public original model architecture diagrams for user-specified LLM, VLM, MoE, diffusion, OCR, and SGLang/sgl-cookbook model families. Use when the user asks for a model structure chart, architecture diagram, 模型结构图, 架构图, or rendered image link for a specific model such as DeepSeek, GLM, Qwen, Kimi, MiniMax, Step, Hunyuan, or Qwen3-VL.
---

# Model Architecture Diagram

## Workflow

Return only public original diagrams indexed by this skill.

1. Run the bundled resolver:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "<model name>"
```

2. If the resolver returns `kind: existing`, return the raw image Markdown it prints and preserve the source attribution line.
3. If the resolver returns `kind: no_match`, tell the user that no public original architecture diagram is indexed for that model.

## Source Priority

Use `references/diagram-index.json` as the source of truth. It stores raw GitHub image URLs from:

- `datawhalechina/self-llm`
- `CalvinXKY/InfraTech`
- `Tongyi-MAI/Z-Image`
- `Wan-Video/Wan2.1`
- `Wan-Video/Wan2.2`
- `Tencent-Hunyuan/HunyuanVideo`
- `Tencent-Hunyuan/Hunyuan3D-2`
- `brayevalerien/Flux.1-Architecture-Diagram`

Prefer detailed implementation, cookbook, or architecture-card diagrams over
paper figures. Good sources show module boundaries, dataflow, MoE / attention /
cache paths, or model-specific runtime structure rather than only a high-level
paper overview. Official repository diagrams and curated implementation
diagrams are first choice; paper figures are fallback only when no more detailed
public original diagram is indexed.

Do not copy remote image binaries into the skill. Return the raw GitHub URLs so the chat renderer can display the original image.

## Existing Diagram Rule

For a direct match, show the original image. Good direct matches include:

- DeepSeek V3/V3.2/V4, GLM-5, Kimi K2/K2.5, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash from InfraTech.
- Hunyuan-A13B, Kimi-VL, Qwen3, Qwen3-VL detail flows, MiniMax M2, and Llama 4 architecture/module diagrams from self-llm.
- Z-Image, Wan2.1, Wan2.2, HunyuanVideo, Hunyuan3D 2.0, and FLUX.1 diffusion architecture/module diagrams from public GitHub sources.

If multiple diagrams match, show all high-confidence matches up to the resolver's default limit. For example, DeepSeek V3 may return the full architecture plus MLA MHA/MQA diagrams.

## Hosted Original Diagram Gallery

Do not commit the `sgl-cookbook-model-architecture-images/` gallery into the repository. The public-original image set is hosted as a GitHub Release asset and indexed by a GitHub issue.

Current hosted artifact:

- Issue index: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/issues/31
- Release page: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/tag/sgl-cookbook-architecture-images-2026-04-27
- Zip download: https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-04-27/sgl-cookbook-model-architecture-images-2026-04-27.zip
- Digest: `sha256:07d4989e4ee8e137013556efb79478028e77fd598ccc67d055fcbf902b5b0efc`

The artifact contains 44 public original diagram image files from the indexed upstream repositories, plus a lightweight `index.html`, `index.md`, `manifest.json`, HTML contact sheet, and `architecture-audit.md`.

To inspect the gallery locally:

```bash
curl -L -o /tmp/sgl-cookbook-model-architecture-images-2026-04-27.zip \
  https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/releases/download/sgl-cookbook-architecture-images-2026-04-27/sgl-cookbook-model-architecture-images-2026-04-27.zip
unzip -q /tmp/sgl-cookbook-model-architecture-images-2026-04-27.zip -d /tmp
open /tmp/sgl-cookbook-model-architecture-images/index.html
```

## Useful Commands

List known original diagram aliases:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py --list-known
```

Emit JSON for automation:

```bash
python3 skills/model-architecture-diagram/scripts/model_architecture_diagram.py "GLM-5" --format json
```

## References

- `references/diagram-index.json`: original diagram link index and aliases.
- `references/source-notes.md`: audited source repositories and local cache paths.
