# Model Architecture Diagram Source Notes

Audited on `2026-04-24`.

## Upstream Diagram Sources

- `datawhalechina/self-llm` at `a7cd4ef135b0` (`master`): broad model deployment/tutorial repository. Confirmed architecture-style diagrams include Hunyuan-A13B and Kimi-VL.
- `CalvinXKY/InfraTech` at `0fe7d3a57dce` (`main`): architecture-card repository with original diagrams for DeepSeek V3/V3.2, GLM-5, Kimi K2/K2.5, MiniMax M2.5, Qwen3.5, Qwen3-VL, and Step 3.5 Flash.

The skill stores raw GitHub URLs in `diagram-index.json`; it intentionally does not vendor binary images.

## Local Cache Paths

These paths are optional local mirrors of the upstream repositories:

- InfraTech: `/tmp/InfraTech`
- self-llm: `/tmp/self-llm`

The resolver returns the raw GitHub URL even when a local mirror exists.
