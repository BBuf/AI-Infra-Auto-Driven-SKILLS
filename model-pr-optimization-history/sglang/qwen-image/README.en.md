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
