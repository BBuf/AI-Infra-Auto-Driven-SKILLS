# Model Architecture Diagram Source Notes

Audited on `2026-05-01`.

Selection rule added on `2026-05-01`: prefer detailed implementation,
cookbook, and architecture-card diagrams over direct paper figures whenever the
public sources provide both. Paper figures are acceptable fallback evidence only
when no more detailed public original diagram is indexed for the requested
model.

## Upstream Diagram Sources

- `datawhalechina/self-llm` at `a7cd4ef135b0` (`master`): broad model deployment/tutorial repository. Confirmed architecture-style diagrams include Hunyuan-A13B, Kimi-VL, Qwen3, Qwen3-VL detail flows, MiniMax M2, and Llama 4.
- `CalvinXKY/InfraTech` at `fc85c8e112a1` (`main`): architecture-card repository with original diagrams for DeepSeek V3/V3.2/V4, GLM-5, Kimi K2/K2.5, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash.
- `Tongyi-MAI/Z-Image` at `26f23eda626f` (`main`): official Z-Image repository. Confirmed diagrams include S3-DiT architecture, training pipeline, Decoupled-DMD, and DMDR.
- `Wan-Video/Wan2.1` at `9737cba9c1c3` (`main`): official Wan2.1 repository. Confirmed architecture-style diagram includes the video DiT architecture.
- `Wan-Video/Wan2.2` at `42bf4cfaa384` (`main`): official Wan2.2 repository. Confirmed diagrams include MoE architecture, MoE transition schedule, and high-compression VAE.
- `Tencent-Hunyuan/HunyuanVideo` at `e260ed40c88d` (`main`): official HunyuanVideo repository. Confirmed diagrams include overall architecture, backbone, text encoder, and 3D VAE.
- `Tencent-Hunyuan/Hunyuan3D-2` at `f8db63096c82` (`main`): official Hunyuan3D 2.0 repository. Confirmed diagrams include system overview and two-stage architecture.
- `brayevalerien/Flux.1-Architecture-Diagram` at `5d6718e283d6` (`main`): public FLUX.1 architecture diagram repository that attributes the layout to Black Forest Labs source code and the original public diagram.

The skill stores raw GitHub URLs in `diagram-index.json`; it intentionally does not vendor binary images.

## Local Cache Paths

These paths are optional local mirrors of the upstream repositories:

- InfraTech: `/tmp/InfraTech`
- self-llm: `/tmp/self-llm`

The resolver returns the raw GitHub URL even when a local mirror exists.
