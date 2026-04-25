# SGLang Qwen-Image Support and Optimization PR History

This document covers Qwen-Image, Qwen-Image-Edit, Qwen-Image-Layered, CUDA graph, TeaCache, conditional batching, ModelOpt FP8, and AMD diffusion kernels. Every PR below was read through its GitHub diff and filled with motivation, implementation notes, and key code snippets.

Evidence snapshot:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Related skill: `skills/model-optimization/sglang/sglang-qwen-image-optimization`
- Full PR dossier: `skills/model-optimization/sglang/sglang-qwen-image-optimization/references/pr-history.md`

## Runtime Surfaces

- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`
- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`
- `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`
- `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`
- `docs/diffusion/quantization.md`

## PR Cards

### #18530 - AMD fused norm and RoPE

- Link: https://github.com/sgl-project/sglang/pull/18530
- State: open, `2` files, `+95/-35`
- Motivation: Qwen-Image cross-attention on AMD ROCm runs Q/K RMSNorm and RoPE separately. The PR fuses them with AITER and reports Qwen-Image-Edit latency improving from `84.48s` to `79.72s`.
- Key implementation: add `SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY`; under AITER/HIP/RMSNorm/image-RoPE conditions call `aiter.fused_rope_rms_2way` to produce joint query/key tensors.
- Code:

```python
use_fused_rope_rms = (
    _use_aiter
    and envs.SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY.get()
    and image_rotary_emb is not None
)
```

```python
aiter.fused_rope_rms_2way(
    txt_query.contiguous(),
    txt_key.contiguous(),
    img_query.contiguous(),
    img_key.contiguous(),
    ...
    joint_query,
    joint_key,
)
```

- Validation: AMD/AITER image quality and denoise latency are required.

### #19066 - Qwen2.5-VL ViT/text encoder optimization

- Link: https://github.com/sgl-project/sglang/pull/19066
- State: open, `7` files, `+874/-21`
- Motivation: Qwen-Image-adjacent Qwen2.5-VL encoding was slow and less compile-friendly; AMD SDPA with non-contiguous q/k/v slices could accumulate numerical error.
- Key implementation: add custom `Qwen2_5_VisionTransformer`, backend selection, SDPA `.contiguous()` fix, RoPE/window cache, fused ViT `gate_up_proj`, SGLang RMSNorm, and torch.compile mode plumbing.
- Code:

```python
def get_vit_attn_backend(attn_implementation: str | None = None):
    if attn_implementation == "sdpa":
        return AttentionBackendEnum.TORCH_SDPA
    try:
        import flash_attn
        return AttentionBackendEnum.FA
    except ImportError:
        return AttentionBackendEnum.TORCH_SDPA
```

```python
output_i = F.scaled_dot_product_attention(
    query[:, :, start_idx:end_idx, :].contiguous(),
    key[:, :, start_idx:end_idx, :].contiguous(),
    value[:, :, start_idx:end_idx, :].contiguous(),
)
```

- Validation: run custom ViT vs HF consistency tests and Qwen-Image/Edit e2e generation.

### #19516 - Initial CUDA graph support

- Link: https://github.com/sgl-project/sglang/pull/19516
- State: open, `3` files, `+315/-36`
- Motivation: Qwen-Image profiles showed about `30%` CPU overhead. Shapes depend on resolution and prompt length, so the first design captures smaller block subfunctions and pads text length.
- Key implementation: split image/text pre/post attention, capture them with `torch.cuda.make_graphed_callables`, pad text to next power of two, and add `--enable-cuda-graph`.
- Code:

```python
def pad_tensor_into_power_of_2(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    padded_length = next_power_of_2(tensor.shape[dim])
    ...
    return torch.cat([tensor, padding], dim=dim)
```

```python
torch.cuda.make_graphed_callables(
    self._txt_pre_attention_forward,
    (padded_encoder_hidden_states, temb_txt_silu),
    pool=self.GLOBAL_GRAPH_POOL_HANDLE,
)
```

- Validation: design history; prefer the reland in #20810 for current work.

### #19521 - Qwen diffusion model detectors

- Link: https://github.com/sgl-project/sglang/pull/19521
- State: open, `1` file, `+22/-1`
- Motivation: local directories whose names do not contain canonical HF IDs could not be resolved by the diffusion registry.
- Key implementation: add pipeline-class detectors for Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-Plus, Qwen-Image-Layered, and GLM-Image.
- Code:

```python
model_detectors=[
    lambda pipeline_class: "qwenimagepipeline" in pipeline_class.lower()
],
```

```python
model_detectors=[
    lambda pipeline_class: "qwenimageeditpluspipeline" in pipeline_class.lower()
],
```

- Validation: test local paths such as `/models/foobar`.

### #20429 - Fused LayerNorm + scale/shift/gate select01

- Link: https://github.com/sgl-project/sglang/pull/20429
- State: open, `2` files, `+350/-22`
- Motivation: `_modulate()` ran residual, LayerNorm, select01, and modulation as separate kernels.
- Key implementation: add Triton `_fused_modulate_kernel` that optionally computes residual, applies LayerNorm, selects modulation params by index, and writes output/gate.
- Code:

```python
@triton.jit
def _fused_modulate_kernel(..., HAS_RESIDUAL: tl.constexpr):
    residual_out = gate_x * x + residual
    mean = tl.sum(x_for_norm, axis=0) / C
    y = x_hat * (1.0 + scale) + shift
```

```python
if _FUSE_LN_SCALE_SHIFT_SELECT01:
    x, residual_out, gate_result = fused_modulate_kernel(...)
```

- Validation: fixed-seed fused/unfused image comparison plus fp16/residual-path checks.

### #20432 - Dual-stream forward

- Link: https://github.com/sgl-project/sglang/pull/20432
- State: open, `1` file, `+232/-26`
- Motivation: on B200, text qkv/feedforward can overlap with image qkv/feedforward. PR body reports `7.83s -> 7.63s`; H200 gains were trivial, so it is env-gated.
- Key implementation: add `QWEN_IMAGE_DUAL_STREAM_FORWARD`, high-priority stream, image/text QKV helpers, and post-attention image MLP overlap.
- Code:

```python
_DUAL_STREAM_FORWARD = os.environ.get("QWEN_IMAGE_DUAL_STREAM_FORWARD", "0") == "1"
```

```python
with self.device_module.stream(high_priority_stream):
    img_query, img_key, img_value = _get_qkv_projections_img(self, hidden_states)
...
main_stream.wait_stream(high_priority_stream)
```

- Validation: benchmark per GPU family; B200 gains do not automatically transfer to H200.

### #20447 - TeaCache

- Link: https://github.com/sgl-project/sglang/pull/20447
- State: open, `8` files, `+295/-105`
- Motivation: TeaCache skips denoise work by reusing residuals when timestep-conditioned changes are small. PR body reports Qwen-Image-2512 `156.12s -> 61.29s`.
- Key implementation: add Qwen/GLM/Flux TeaCache params, read `enable_teacache` from forward context, support CFG branch separation, and wrap Qwen-Image blocks with skip/finalize hooks.
- Code:

```python
@dataclass
class QwenImageTeaCacheParams(TeaCacheParams):
    teacache_thresh: float = 0.2
    coefficients: list[float] = field(default_factory=lambda: [...])
```

```python
skip, hs_or_orig = self.teacache_skip_or_prepare(hidden_states, temb)
if skip:
    hidden_states = hs_or_orig
else:
    ...
    self.teacache_finalize(hidden_states, hs_or_orig)
```

- Validation: save images and latency; TeaCache is a quality/speed tradeoff.

### #20810 - Reland Qwen-Image CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/20810
- State: open, `4` files, `+681/-47`
- Motivation: reland #19516 with safer graph cache, static input pools, text buckets, and replay signature checks.
- Key implementation: add `CudaGraphCallableCache`, `SharedStaticInputPool`, text bucket selection, graph wrappers around block pre/post attention, and mutual exclusion with torch.compile.
- Code:

```python
class CudaGraphCallableCache:
    def capture_or_replay(...):
        with torch.cuda.graph(graph, pool=self._get_pool_handle()):
            output = fn(*static_inputs)
        ...
        entry.graph.replay()
        return entry.output
```

```python
if self.enable_torch_compile and self.enable_cuda_graph:
    raise ValueError(
        "--enable-torch-compile and --enable-cuda-graph are mutually exclusive for diffusion runtime"
    )
```

- Validation: prefer this graph design; test bucket fallback, graph memory, and image equality.

### #21988 - Multi-output condition batch fix

- Link: https://github.com/sgl-project/sglang/pull/21988
- State: open, `1` file, `+45/-2`
- Motivation: `num_outputs_per_prompt > 1` expanded latent samples but not text condition batches, causing denoise shape mismatch.
- Key implementation: repeat-interleave prompt and negative-prompt embeddings to `batch.batch_size`.
- Code:

```python
repeat_factor = target_batch_size // current_batch_size
return tensor.repeat_interleave(repeat_factor, dim=0).contiguous()
```

- Validation: test `num_outputs_per_prompt=1/2/4/8` and separate denoise success from VAE OOM.

### #22362 - Qwen-Image-Layered serve fix

- Link: https://github.com/sgl-project/sglang/pull/22362
- State: open, `2` files, `+4/-2`
- Motivation: `/v1/images/edits` required `prompt` even when Qwen-Image-Layered did not need it; RGBA outputs failed when saved as JPEG.
- Key implementation: default prompt to `" "` and default output extension to PNG.
- Code:

```python
prompt: str = Form(" ")
```

```python
return "png"
```

- Validation: test CLI and server layered/RGBA outputs.

### #22397 - Weight-name mapping

- Link: https://github.com/sgl-project/sglang/pull/22397
- State: open, `1` file, `+20/-0`
- Motivation: checkpoints may save `attn.to_out.weight` while SGLang exposes `attn.to_out.0.weight`; added Q/K/V projections may be split.
- Key implementation: map flat `to_out` to indexed `to_out.0`, and merge `add_q/k/v_proj` into `to_added_qkv` with shard ids 0/1/2.
- Code:

```python
r"^(transformer_blocks\.[0-9]+\.attn\.to_out)\.(weight|bias)$": r"\1.0.\2",
r"^(transformer_blocks\.(\d+)\.attn)\.add_q_proj\.(.+)$": (
    r"\1.to_added_qkv.\3",
    0,
    3,
),
```

- Validation: loader tests should cover flat/indexed `to_out` and split/fused added QKV.

### #22953 - Avoid Qwen-Image RoPE illegal memory access

- Link: https://github.com/sgl-project/sglang/pull/22953
- State: merged at `2026-04-23T04:41:27Z`, `1` file, `+12/-0`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `32` lines; current-main source rechecked at `bca3dd958`.
- Motivation: too many input images or long prompts can make text sequence length exceed RoPE cache length, causing CUDA illegal memory access.
- Key implementation: check required text length against RoPE text cache before entering the kernel and raise a clear `ValueError`.
- Code:

```python
if max_txt_seq_len > txt_cache_len:
    overflow = max_txt_seq_len - txt_cache_len
    raise ValueError(
        "QwenImage RoPE text cache overflow before denoising: "
        f"required_txt_seq_len={max_txt_seq_len}, txt_cache_len={txt_cache_len}, "
        f"overflow={overflow}. "
    )
```

- Validation: long prompt / many-image requests should fail fast before corrupting CUDA context. This guard is present in current SGLang main.

### #23155 - ModelOpt FP8

- Link: https://github.com/sgl-project/sglang/pull/23155
- State: open, `4` files, `+210/-33`
- Motivation: Qwen-Image and Qwen-Image-Edit need ModelOpt FP8 support, but naive FP8 caused severe dark/blurred image regression.
- Key implementation: make Qwen-Image DiT projections quant-aware, replace Diffusers FeedForward with `QwenImageFeedForward`, add Qwen BF16 fallback patterns, canonicalize modulation names, and write BF16 fallback tensors before ModelOpt ignored-weight preservation.
- Code:

```python
self.to_q = ReplicatedLinear(
    dim,
    self.inner_dim,
    bias=True,
    quant_config=quant_config,
    prefix=f"{prefix}.to_q",
)
```

```python
DEFAULT_QWEN_IMAGE_KEEP_BF16_PATTERNS = [
    r"^img_in$",
    r"^txt_in$",
    r"^time_text_embed\.timestep_embedder\.linear_[12]$",
    r"^norm_out\.linear$",
    r"^proj_out$",
    r"^transformer_blocks\.\d+\.img_mlp\.net\.2$",
    r"^transformer_blocks\.\d+\.(img_mod|txt_mod)$",
]
```

```python
if name in fallback_tensors:
    shard_tensors[name] = fallback_tensors[name]
    continue
```

- Validation: BF16-vs-FP8 image comparison, benchmark JSON, and profiler artifacts are required for both Qwen-Image and Qwen-Image-Edit.

## Validation Matrix

- Fixed prompt/seed/resolution/steps BF16 text-to-image.
- Fixed image/prompt BF16 edit.
- CUDA graph on/off including text bucket fallback.
- TeaCache on/off with saved images and latency.
- `num_outputs_per_prompt=1/2/4/8`.
- Qwen-Image-Layered `/v1/images/edits` without prompt and with RGBA output.
- ModelOpt FP8 Qwen-Image and Qwen-Image-Edit.
- AMD AITER fused RoPE/RMSNorm and Triton fused modulation.

<!-- MODEL_PR_DIFF_AUDIT:START en -->

## PR Diff Audit Cards (2026-04-25 rebuild)

This section re-audits `Qwen-Image` against `sgl-project/sglang` Pull Request metadata and file-level patches. Acceptance rule: every PR needs status, code surface, file-level diff digest, support/optimization interpretation, and verification risk notes; if no public PR is found, keep an explicit no-match conclusion instead of inventing history.

### Timeline

| Created | PR | State | Title | Code surface | Main diff files |
| --- | ---: | --- | --- | --- | --- |
| 2026-02-10 | [#18530](https://github.com/sgl-project/sglang/pull/18530) | open | [Diffusion] [AMD] fuse norm & rope for qwen-image | model wrapper, multimodal/processor | `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py` |
| 2026-02-20 | [#19066](https://github.com/sgl-project/sglang/pull/19066) | open | [diffusion] model: optimize Qwen2.5-VL ViT and text encoder | model wrapper, multimodal/processor, tests/benchmarks | `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`, `test/unit/test_qwen25_vit.py`, `python/sglang/multimodal_gen/runtime/models/encoders/vision.py` |
| 2026-02-27 | [#19516](https://github.com/sgl-project/sglang/pull/19516) | open | [Diffusion] add cuda graph support for Qwen-Image | model wrapper, multimodal/processor | `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/runtime/server_args.py` |
| 2026-02-27 | [#19521](https://github.com/sgl-project/sglang/pull/19521) | open | [Feature] Adding detectors for Qwen diffusion models | multimodal/processor | `python/sglang/multimodal_gen/registry.py` |
| 2026-03-12 | [#20429](https://github.com/sgl-project/sglang/pull/20429) | open | [Diffusion][Qwen-Image] Kernel fusion on layernorm and fuse_scale_shift_gate_select01 | model wrapper, kernel, multimodal/processor | `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` |
| 2026-03-12 | [#20432](https://github.com/sgl-project/sglang/pull/20432) | open | [Diffusion][Qwen-Image] Dual stream forward for Qwen-Image | model wrapper, multimodal/processor | `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` |
| 2026-03-12 | [#20447](https://github.com/sgl-project/sglang/pull/20447) | open | [Diffusion] Feat: support teacache for glm-image, qwen-image et.al | model wrapper, multimodal/processor, scheduler/runtime, docs/config | `python/sglang/multimodal_gen/runtime/cache/teacache.py`, `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` |
| 2026-03-18 | [#20810](https://github.com/sgl-project/sglang/pull/20810) | open | [Diffusion] Reland qwen image cuda graph | model wrapper, kernel, multimodal/processor | `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`, `python/sglang/multimodal_gen/runtime/server_args.py` |
| 2026-04-03 | [#21988](https://github.com/sgl-project/sglang/pull/21988) | open | [diffusion] fix: align qwen-image cond batch for multi-output generation | multimodal/processor, docs/config | `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` |
| 2026-04-08 | [#22362](https://github.com/sgl-project/sglang/pull/22362) | open | [Fix] fix error in sglang serve for qwen-image-layer | multimodal/processor | `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`, `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py` |
| 2026-04-08 | [#22397](https://github.com/sgl-project/sglang/pull/22397) | open | [BugFix] qwenimage: fix weight-name mapping for to_out and added QKV projections | model wrapper, multimodal/processor, docs/config | `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py` |
| 2026-04-16 | [#22953](https://github.com/sgl-project/sglang/pull/22953) | merged | [diffusion][bugfix] avoid illegal memory access in qwen image | multimodal/processor, docs/config | `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` |
| 2026-04-19 | [#23155](https://github.com/sgl-project/sglang/pull/23155) | open | [Diffusion] Add Qwen Image ModelOpt FP8 support | model wrapper, quantization, multimodal/processor, docs/config | `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md` |

### File-level PR diff reading notes

### PR #18530 - [Diffusion] [AMD] fuse norm & rope for qwen-image

- Link: https://github.com/sgl-project/sglang/pull/18530
- Status/date: `open`, created 2026-02-10; author `qichu-yun`.
- Diff scope read: `2` files, `+95/-35`; areas: model wrapper, multimodal/processor; keywords: attention, cache, cuda, flash, kv, mla, quant.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +94/-35 (129 lines); hunks: import functools; ); symbols: _get_qkv_projections, forward
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunks: class Envs:; symbols: Envs:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py`; keywords observed in patches: attention, cache, cuda, flash, kv, mla. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19066 - [diffusion] model: optimize Qwen2.5-VL ViT and text encoder

- Link: https://github.com/sgl-project/sglang/pull/19066
- Status/date: `open`, created 2026-02-20; author `Jasen2201`.
- Diff scope read: `7` files, `+874/-21`; areas: model wrapper, multimodal/processor, tests/benchmarks; keywords: config, cuda, attention, flash, spec, vision, cache, kv, quant, test.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py` modified +451/-16 (467 lines); hunks: # Copied and adapted from: https://github.com/hao-ai-lab/FastVideo; from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig; symbols: Qwen2_5_VLVisionAttention, __init__, forward, Qwen2_5_VLVisionBlock
  - `test/unit/test_qwen25_vit.py` added +238/-0 (238 lines); hunks: +"""; symbols: _calculate_dimensions, _pixel_to_grid, _build_grid_cases, _ensure_tp_initialized
  - `python/sglang/multimodal_gen/runtime/models/encoders/vision.py` modified +140/-1 (141 lines); hunks: # Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/vision.py; def get_patch_grid_length(self) -> int:; symbols: get_patch_grid_length, get_vit_attn_backend, _vit_flash_attn_varlen, _vit_sdpa_varlen
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py` modified +37/-0 (37 lines); hunks: from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution; def __init__(; symbols: __init__, _maybe_compile_text_encoder, load_model
  - `python/sglang/multimodal_gen/envs.py` modified +5/-0 (5 lines); hunks: VERBOSE: bool = False; def _getter():; symbols: _getter
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`, `test/unit/test_qwen25_vit.py`, `python/sglang/multimodal_gen/runtime/models/encoders/vision.py`; keywords observed in patches: config, cuda, attention, flash, spec, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`, `test/unit/test_qwen25_vit.py`, `python/sglang/multimodal_gen/runtime/models/encoders/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19516 - [Diffusion] add cuda graph support for Qwen-Image

- Link: https://github.com/sgl-project/sglang/pull/19516
- Status/date: `open`, created 2026-02-27; author `zyksir`.
- Diff scope read: `3` files, `+315/-36`; areas: model wrapper, multimodal/processor; keywords: cuda, attention, scheduler, cache, config, kv, mla, processor, quant.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +288/-35 (323 lines); hunks: ); def forward(; symbols: forward, pad_tensor_into_power_of_2, QwenImageTransformerBlock, __init__
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +19/-1 (20 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, _maybe_enable_cuda_graph_capture, _maybe_enable_torch_compile
  - `python/sglang/multimodal_gen/runtime/server_args.py` modified +8/-0 (8 lines); hunks: class ServerArgs:; def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:; symbols: ServerArgs:, add_cli_args
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/runtime/server_args.py`; keywords observed in patches: cuda, attention, scheduler, cache, config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/runtime/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19521 - [Feature] Adding detectors for Qwen diffusion models

- Link: https://github.com/sgl-project/sglang/pull/19521
- Status/date: `open`, created 2026-02-27; author `shenoyvvarun`.
- Diff scope read: `1` files, `+22/-1`; areas: multimodal/processor; keywords: config.
- Code diff details:
  - `python/sglang/multimodal_gen/registry.py` modified +22/-1 (23 lines); hunks: def _register_configs():; symbols: _register_configs
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/registry.py`; keywords observed in patches: config. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/registry.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20429 - [Diffusion][Qwen-Image] Kernel fusion on layernorm and fuse_scale_shift_gate_select01

- Link: https://github.com/sgl-project/sglang/pull/20429
- Status/date: `open`, created 2026-03-12; author `SYChen123`.
- Diff scope read: `2` files, `+350/-22`; areas: model wrapper, kernel, multimodal/processor; keywords: triton, attention, config, cuda.
- Code diff details:
  - `python/sglang/jit_kernel/diffusion/triton/scale_shift.py` modified +301/-3 (304 lines); hunks: +from typing import Optional, Tuple; def fuse_scale_shift_gate_select01_kernel_blc_opt(; symbols: fuse_scale_shift_gate_select01_kernel_blc_opt, fuse_scale_shift_gate_select01_kernel_blc_opt, fuse_scale_shift_gate_select01_kernel, _fused_modulate_kernel
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +49/-19 (68 lines); hunks: from sglang.jit_kernel.diffusion.triton.scale_shift import (; logger = init_logger(__name__) # pylint: disable=invalid-name; symbols: _modulate
- Optimization/support interpretation: The concrete diff surface is `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; keywords observed in patches: triton, attention, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20432 - [Diffusion][Qwen-Image] Dual stream forward for Qwen-Image

- Link: https://github.com/sgl-project/sglang/pull/20432
- Status/date: `open`, created 2026-03-12; author `SYChen123`.
- Diff scope read: `1` files, `+232/-26`; areas: model wrapper, multimodal/processor; keywords: attention, cache, cuda, flash, kv, processor.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +232/-26 (258 lines); hunks: # SPDX-License-Identifier: Apache-2.0; ); symbols: _get_or_create_alt_stream, _get_qkv_projections_img, _get_qkv_projections_txt, _get_qkv_projections
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; keywords observed in patches: attention, cache, cuda, flash, kv, processor. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20447 - [Diffusion] Feat: support teacache for glm-image, qwen-image et.al

- Link: https://github.com/sgl-project/sglang/pull/20447
- Status/date: `open`, created 2026-03-12; author `RuixiangMa`.
- Diff scope read: `8` files, `+295/-105`; areas: model wrapper, multimodal/processor, scheduler/runtime, docs/config; keywords: cache, config, attention, processor, kv, quant, spec, triton.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/cache/teacache.py` modified +107/-19 (126 lines); hunks: def forward(self, hidden_states, timestep, ...):; def _init_teacache_state(self) -> None:; symbols: forward, _init_teacache_state, _init_teacache_state, _update_teacache_status
  - `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py` modified +38/-29 (67 lines); hunks: import torch; def forward(; symbols: forward, forward
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +34/-24 (58 lines); hunks: import functools; def forward(; symbols: forward, forward
  - `python/sglang/multimodal_gen/runtime/models/dits/flux.py` modified +27/-20 (47 lines); hunks: import torch; def __init__(; symbols: __init__, forward, forward
  - `python/sglang/multimodal_gen/configs/sample/glmimage.py` modified +30/-1 (31 lines); hunks: -from dataclasses import dataclass; class GlmImageSamplingParams(SamplingParams):; symbols: GlmImageTeaCacheParams, GlmImageSamplingParams
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/cache/teacache.py`, `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; keywords observed in patches: cache, config, attention, processor, kv, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/cache/teacache.py`, `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20810 - [Diffusion] Reland qwen image cuda graph

- Link: https://github.com/sgl-project/sglang/pull/20810
- Status/date: `open`, created 2026-03-18; author `BBuf`.
- Diff scope read: `4` files, `+681/-47`; areas: model wrapper, kernel, multimodal/processor; keywords: cuda, attention, cache, config, quant, scheduler, kv, mla, processor.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +368/-46 (414 lines); hunks: ); def forward(; symbols: forward, QwenImageTransformerBlock, _get_shared_cuda_graph_pool_handle, _get_shared_cuda_graph_input_pool
  - `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py` added +240/-0 (240 lines); hunks: +from __future__ import annotations; symbols: pad_tensor_along_dim, pad_tensor_to_next_power_of_2, replace_shape_dim, shape_with_next_power_of_2
  - `python/sglang/multimodal_gen/runtime/server_args.py` modified +46/-0 (46 lines); hunks: class ServerArgs:; def _adjust_parameters(self):; symbols: ServerArgs:, _adjust_parameters, _validate_parameters, _adjust_save_paths
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +27/-1 (28 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, _maybe_enable_cuda_graph_capture, _maybe_enable_torch_compile
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`, `python/sglang/multimodal_gen/runtime/server_args.py`; keywords observed in patches: cuda, attention, cache, config, quant, scheduler. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`, `python/sglang/multimodal_gen/runtime/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21988 - [diffusion] fix: align qwen-image cond batch for multi-output generation

- Link: https://github.com/sgl-project/sglang/pull/21988
- Status/date: `open`, created 2026-04-03; author `MikukuOvO`.
- Diff scope read: `1` files, `+45/-2`; areas: multimodal/processor, docs/config; keywords: cache, config.
- Code diff details:
  - `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` modified +45/-2 (47 lines); hunks: def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):; def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):; symbols: get_freqs_cis, _expand_cond_tensor_batch, _expand_cond_batch, get_pos_prompt_embeds
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`; keywords observed in patches: cache, config. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22362 - [Fix] fix error in sglang serve for qwen-image-layer

- Link: https://github.com/sgl-project/sglang/pull/22362
- Status/date: `open`, created 2026-04-08; author `XingyongCheng`.
- Diff scope read: `2` files, `+3/-2`; areas: multimodal/processor; keywords: n/a.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py` modified +2/-1 (3 lines); hunks: def choose_output_image_ext(; symbols: choose_output_image_ext, build_sampling_params
  - `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py` modified +1/-1 (2 lines); hunks: async def edits(; symbols: edits
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`, `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`; keywords observed in patches: n/a. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`, `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22397 - [BugFix] qwenimage: fix weight-name mapping for to_out and added QKV projections

- Link: https://github.com/sgl-project/sglang/pull/22397
- Status/date: `open`, created 2026-04-08; author `jy-song-hub`.
- Diff scope read: `1` files, `+20/-0`; areas: model wrapper, multimodal/processor, docs/config; keywords: attention, config, kv, lora, quant, spec.
- Code diff details:
  - `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py` modified +20/-0 (20 lines); hunks: class QwenImageArchConfig(DiTArchConfig):; symbols: QwenImageArchConfig
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`; keywords observed in patches: attention, config, kv, lora, quant, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22953 - [diffusion][bugfix] avoid illegal memory access in qwen image

- Link: https://github.com/sgl-project/sglang/pull/22953
- Status/date: `merged`, created 2026-04-16, merged 2026-04-23; author `IPostYellow`.
- Diff scope read: `1` files, `+12/-0`; areas: multimodal/processor, docs/config; keywords: cache, config, flash.
- Code diff details:
  - `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` modified +12/-0 (12 lines); hunks: def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):; symbols: get_freqs_cis
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`; keywords observed in patches: cache, config, flash. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #23155 - [Diffusion] Add Qwen Image ModelOpt FP8 support

- Link: https://github.com/sgl-project/sglang/pull/23155
- Status/date: `open`, created 2026-04-19; author `BBuf`.
- Diff scope read: `4` files, `+210/-33`; areas: model wrapper, quantization, multimodal/processor, docs/config; keywords: config, fp8, quant, doc, fp4, kv, attention, benchmark, flash, spec.
- Code diff details:
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +124/-20 (144 lines); hunks: import torch; def __init__(; symbols: __init__, __init__, forward, QwenImageGELU
  - `docs/diffusion/quantization.md` modified +36/-6 (42 lines); hunks: backend.; sglang generate \
  - `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md` modified +29/-2 (31 lines); hunks: This repo now contains:; For `FLUX.1-dev`, the validated fallback set currently keeps these modules in BF
  - `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py` modified +21/-5 (26 lines); hunks: r"^transformer_blocks\.(0\|43\|44\|45\|46\|47)\.(attn1\|attn2\|audio_attn1\|audio_attn2\|audio_to_video_attn\|video_to_audio_attn)\.to_out\.0$",; def _module_name_variant; symbols: _resolve_transformer_dir, _module_name_variants, get_default_keep_bf16_patterns, build_modelopt_fp8_transformer
- Optimization/support interpretation: The concrete diff surface is `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md`; keywords observed in patches: config, fp8, quant, doc, fp4, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


### Gap and optimization follow-up

- Covered PRs: 13; open PRs: 12.
- Open PRs to keep tracking: [#18530](https://github.com/sgl-project/sglang/pull/18530), [#19066](https://github.com/sgl-project/sglang/pull/19066), [#19516](https://github.com/sgl-project/sglang/pull/19516), [#19521](https://github.com/sgl-project/sglang/pull/19521), [#20429](https://github.com/sgl-project/sglang/pull/20429), [#20432](https://github.com/sgl-project/sglang/pull/20432), [#20447](https://github.com/sgl-project/sglang/pull/20447), [#20810](https://github.com/sgl-project/sglang/pull/20810), [#21988](https://github.com/sgl-project/sglang/pull/21988), [#22362](https://github.com/sgl-project/sglang/pull/22362), [#22397](https://github.com/sgl-project/sglang/pull/22397), [#23155](https://github.com/sgl-project/sglang/pull/23155)
- Any future PR must add both the timeline row and the file-level diff card; title-only summaries are not acceptable.

<!-- MODEL_PR_DIFF_AUDIT:END en -->
