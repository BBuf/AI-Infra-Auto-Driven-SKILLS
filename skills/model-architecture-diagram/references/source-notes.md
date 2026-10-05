# Model Architecture Diagram Source Notes

Audited on `2026-10-05`; hosted gallery remains its 2026-05-02 snapshot.

Selection rule added on `2026-05-01`: prefer detailed implementation,
cookbook, and architecture-card diagrams over direct paper figures whenever the
public sources provide both. Paper figures are acceptable fallback evidence only
when no more detailed public original diagram is indexed for the requested
model.

## Upstream Diagram Sources

- `datawhalechina/self-llm` at `e0fe14a35123f5cab6d32cb9e716b571bf0994cf` (`master`): broad model deployment/tutorial repository. Confirmed architecture-style diagrams include Hunyuan-A13B, Kimi-VL, Qwen3, Qwen3-VL detail flows, MiniMax M2, and Llama 4.
- `CalvinXKY/InfraTech` at `48e358a4b5119c24cf472adc64d0b4f86f894e99` (`main`): architecture-card repository with original diagrams for DeepSeek V3/V3.2/V4, GLM-5, Kimi K2/K2.5/K3, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash. The Kimi K3 original is `models/kimi_k_3/kimi_k_3_architecture.jpg` (10,850 × 12,619 pixels); the skill links to the public raw image rather than vendoring it.
- `Tongyi-MAI/Z-Image` at `26f23eda626f` (`main`): official Z-Image repository. Confirmed diagrams include S3-DiT architecture, training pipeline, Decoupled-DMD, and DMDR.
- `Wan-Video/Wan2.1` at `9737cba9c1c3` (`main`): official Wan2.1 repository. Confirmed architecture-style diagram includes the video DiT architecture.
- `Wan-Video/Wan2.2` at `42bf4cfaa384` (`main`): official Wan2.2 repository. Confirmed diagrams include MoE architecture, MoE transition schedule, and high-compression VAE.
- `Tencent-Hunyuan/HunyuanVideo` at `e260ed40c88d` (`main`): official HunyuanVideo repository. Confirmed diagrams include overall architecture, backbone, text encoder, and 3D VAE.
- `Tencent-Hunyuan/Hunyuan3D-2` at `f8db63096c82` (`main`): official Hunyuan3D 2.0 repository. Confirmed diagrams include system overview and two-stage architecture.
- `brayevalerien/Flux.1-Architecture-Diagram` at `5d6718e283d6` (`main`): public FLUX.1 architecture diagram repository that attributes the layout to Black Forest Labs source code and the original public diagram.

The skill stores original indexed URLs in `diagram-index.json`; it intentionally does not vendor binary images.

## Local Cache Paths

These paths are optional local mirrors of the upstream repositories:

- InfraTech: `/tmp/InfraTech`
- self-llm: `/tmp/self-llm`

The resolver returns the raw GitHub URL even when a local mirror exists.

New verified diagrams: Qwen3.8-Flash-Next official CDN (HF model-card revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`), MiMo V2.6 and V2.5 Pro pinned
HF images, GLM-5.2 at self-llm `e0fe14a35123f5cab6d32cb9e716b571bf0994cf`.
All four returned HTTP 200 and image/png on 2026-10-05 (HF follows a 302).
Vendor CDN files are mutable; pin the referring card and record the audit date.
HF links retain immutable resolve revisions rather than expiring signed CDN URLs.

InfraTech `48e358a4b5119c24cf472adc64d0b4f86f894e99` retains its 17 images.
DeepSeek V4.1, Hy4, GLM-5.3-Flash, Inkling, Step-3.7, dots3, MiniMax-M3,
Ling3, Gemma4 and GigaChat3.5 remain unindexed: inspected public images were
benchmarks/training figures or absent. A runtime family alias is not enough to
reuse a nearby architecture image.
