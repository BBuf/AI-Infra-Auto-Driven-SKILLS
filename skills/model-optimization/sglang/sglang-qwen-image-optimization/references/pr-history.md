# Qwen-Image PR History

Evidence sweep:

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Manual diff review date: `2026-04-23`
- Searched paths: Qwen-Image diffusion configs, DiT runtime, image API, CUDA graph utilities, TeaCache, ModelOpt converter, diffusion quantization docs, ComfyUI executor and pipeline tests.
- Searched PR terms: `Qwen-Image`, `Qwen Image`, `Qwen-Image-Edit`, `Qwen-Image-Layered`, `qwen_image`, `QwenImage`, `QwenImageTransformer2DModel`.

## Runtime and Docs Surfaces

- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`
- `python/sglang/multimodal_gen/configs/sample/qwenimage.py`
- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/vision.py`
- `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`
- `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`
- `docs/diffusion/quantization.md`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image.mdx`
- `docs_new/cookbook/diffusion/Qwen-Image/Qwen-Image-Edit.mdx`
- `docs_new/src/snippets/diffusion/qwen-image-deployment.jsx`
- `docs_new/src/snippets/diffusion/qwen-image-edit-deployment.jsx`

## Diff-Reviewed PR Cards

### PR #18530 - AMD fused norm and RoPE for Qwen-Image

- Link: https://github.com/sgl-project/sglang/pull/18530
- State: open
- Diff stats: `2` files, `+95/-35`
- Diff coverage: full diff reviewed.
- Motivation: Qwen-Image cross-attention on AMD ROCm spends time in separate RMSNorm/QK normalization and RoPE operations for text and image streams. The PR body reports Qwen-Image-Edit latency improving from `84.48s` to `79.72s` (`5.6%`) with AITER.
- Key implementation: add `_use_aiter` and a guarded `SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY` path. When enabled, concatenate text and image cos/sin caches and call `aiter.fused_rope_rms_2way` to produce joint query/key tensors directly. Fallback keeps the existing `apply_qk_norm` plus FlashInfer RoPE path.
- Key code excerpts:

```python
use_fused_rope_rms = (
    _use_aiter
    and envs.SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY.get()
    and image_rotary_emb is not None
    and isinstance(self.norm_q, RMSNorm)
    and isinstance(self.norm_k, RMSNorm)
    and isinstance(self.norm_added_q, RMSNorm)
    and isinstance(self.norm_added_k, RMSNorm)
)
```

```python
aiter.fused_rope_rms_2way(
    txt_query.contiguous(),
    txt_key.contiguous(),
    img_query.contiguous(),
    img_key.contiguous(),
    self.norm_added_q.weight,
    self.norm_added_k.weight,
    self.norm_q.weight,
    self.norm_k.weight,
    txt_cos_sin,
    img_cos_sin,
    ...
    joint_query,
    joint_key,
)
```

```python
SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY = EnvBool(False)
```

- Reviewed files: `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py`
- Validation implications: validate AMD-only quality and latency with `SGLANG_DIFFUSION_ATTENTION_BACKEND=AITER`, `SGLANG_USE_AITER`, and the new fusion env. Do not enable this path on CUDA or non-RMSNorm configs without a separate validation.

### PR #19066 - Qwen2.5-VL ViT and text encoder optimization

- Link: https://github.com/sgl-project/sglang/pull/19066
- State: open
- Diff stats: `7` files, `+874/-21`
- Diff coverage: Qwen2.5-VL encoder, shared ViT attention helper, text/image encoding stages, torch.compile env handling, and unit test reviewed.
- Motivation: Qwen-Image diffusion uses Qwen2.5-VL-style image/text encoding. The HF ViT path was slower, less torch.compile-friendly, and on AMD SDPA could accumulate numerical error because non-contiguous q/k/v slices entered `F.scaled_dot_product_attention`.
- Key implementation: replace HF `Qwen2_5_VisionTransformerPretrainedModel` with a custom `Qwen2_5_VisionTransformer`. It adds backend selection (`flash_attn` or SDPA), `.contiguous()` slices for SDPA, precomputed/cached rotary embeddings, LRU cache for `(t,h,w)` RoPE/window indices, fused `gate_up_proj` in ViT MLP, SGLang RMSNorm, and `torch.compile` mode plumbing for text encoder and denoise stages.
- Key code excerpts:

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
    dropout_p=dropout_p,
    is_causal=False,
)
```

```python
@lru_cache(maxsize=1024)
def _get_rope_by_thw(self, t: int, h: int, w: int):
    ...
    return (cos_thw, sin_thw), window_index_thw, cu_window_seqlens_thw, cu_seqlens_thw
```

- Reviewed files: `qwen2_5vl.py`, `vision.py`, `image_encoding.py`, `denoising.py`, `mova.py`, `envs.py`, `test/unit/test_qwen25_vit.py`
- Validation implications: use custom-ViT-vs-HF consistency tests, then run end-to-end Qwen-Image/Edit generation. AMD SDPA correctness should be tested separately from FlashAttention.

### PR #19516 - Initial CUDA graph support for Qwen-Image

- Link: https://github.com/sgl-project/sglang/pull/19516
- State: open
- Diff stats: `3` files, `+315/-36`
- Diff coverage: Qwen-Image DiT block split, denoising stage hook, and server args reviewed.
- Motivation: Qwen-Image profiles showed roughly `30%` CPU overhead. Diffusion input shapes depend on resolution and prompt length, so graph capture cannot simply prebuild every possible shape. The initial approach focuses on Qwen-Image and pads text length while leaving image resolution unpadded.
- Key implementation: split `QwenImageTransformerBlock.forward()` into image/text pre-attention and post-attention functions. Capture those small functions with `torch.cuda.make_graphed_callables`. Text tensors are padded to next power-of-two length; image path keys use exact shapes. Add `--enable-cuda-graph` to diffusion server args.
- Key code excerpts:

```python
def pad_tensor_into_power_of_2(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    seq_length = tensor.shape[dim]
    padded_length = next_power_of_2(seq_length)
    ...
    return torch.cat([tensor, padding], dim=dim)
```

```python
self._cuda_graphs[("txt_pre_attention_forward", padded_seq_len_txt)] = (
    torch.cuda.make_graphed_callables(
        self._txt_pre_attention_forward,
        (padded_encoder_hidden_states, temb_txt_silu),
        pool=self.GLOBAL_GRAPH_POOL_HANDLE,
    )
)
```

```python
parser.add_argument(
    "--enable-cuda-graph",
    action=StoreBoolean,
    default=ServerArgs.enable_cuda_graph,
)
```

- Reviewed files: `qwen_image.py`, `denoising.py`, `server_args.py`
- Validation implications: this is open and superseded in shape by `#20810`. Treat it as design history for graph segmentation and text padding, not as the final graph cache design.

### PR #19521 - Qwen diffusion model detectors

- Link: https://github.com/sgl-project/sglang/pull/19521
- State: open
- Diff stats: `1` file, `+22/-1`
- Diff coverage: full diff reviewed.
- Motivation: diffusion model registry lookup failed when a model directory did not match the canonical HF model name, for example customer-local paths such as `/models/foobar`.
- Key implementation: add model detectors based on Diffusers pipeline class names for Qwen-Image, Qwen-Image-2512, Qwen-Image-Edit, Qwen-Image-Edit-2509/2511, Qwen-Image-Layered, and GLM-Image.
- Key code excerpt:

```python
register_configs(
    sampling_param_cls=QwenImageSamplingParams,
    pipeline_config_cls=QwenImagePipelineConfig,
    hf_model_paths=["Qwen/Qwen-Image"],
    model_detectors=[
        lambda pipeline_class: "qwenimagepipeline" in pipeline_class.lower()
    ],
)
```

```python
model_detectors=[
    lambda pipeline_class: "qwenimageeditpluspipeline" in pipeline_class.lower()
],
```

- Reviewed files: `python/sglang/multimodal_gen/registry.py`
- Validation implications: test local directories whose names do not contain `Qwen-Image`, and verify model info resolves from `_class_name` / pipeline class.

### PR #20429 - Fused LayerNorm + scale/shift/gate select01 kernel

- Link: https://github.com/sgl-project/sglang/pull/20429
- State: open
- Diff stats: `2` files, `+350/-22`
- Diff coverage: Triton scale/shift kernel and Qwen-Image modulate path reviewed.
- Motivation: Qwen-Image `_modulate()` did residual update, layernorm, index-based scale/shift/gate selection, and modulation as multiple kernels. The PR fuses these for CUDA to reduce launch overhead and memory traffic.
- Key implementation: add Triton `_fused_modulate_kernel` that optionally computes residual, runs LayerNorm, selects `(shift, scale, gate)` from two branches using `index`, and writes both modulated output and gate. Qwen-Image gates it behind `_FUSE_LN_SCALE_SHIFT_SELECT01`.
- Key code excerpts:

```python
@triton.jit
def _fused_modulate_kernel(..., HAS_RESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr):
    residual_out = gate_x * x + residual
    mean = tl.sum(x_for_norm, axis=0) / C
    var = tl.sum(xbar * xbar, axis=0) / C
    y = x_hat * (1.0 + scale) + shift
```

```python
if _FUSE_LN_SCALE_SHIFT_SELECT01:
    x, residual_out, gate_result = fused_modulate_kernel(
        x=x,
        residual=residual_x if is_scale_residual else None,
        gate_x=gate_x if is_scale_residual else None,
        ln_weight=ln_weight,
        ln_bias=ln_bias,
        ...
    )
```

- Reviewed files: `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- Validation implications: compare fused vs unfused image outputs with fixed seed and verify fp16 clipping/residual paths. Hidden dimension limits in the Triton kernel should be part of failure handling.

### PR #20432 - Dual-stream forward for Qwen-Image

- Link: https://github.com/sgl-project/sglang/pull/20432
- State: open
- Diff stats: `1` file, `+232/-26`
- Diff coverage: full diff reviewed.
- Motivation: on B200, Qwen-Image text qkv/feedforward work can be overlapped with image qkv/feedforward. PR body reports E2E latency improving from `7.83s` to `7.63s`; on H200 gains were trivial, so the feature is optional.
- Key implementation: introduce `QWEN_IMAGE_DUAL_STREAM_FORWARD`, a global high-priority stream, separate image/text QKV helpers, dual-stream attention prep, and dual-stream post-attention MLP overlap.
- Key code excerpts:

```python
_DUAL_STREAM_FORWARD = os.environ.get("QWEN_IMAGE_DUAL_STREAM_FORWARD", "0") == "1"
```

```python
with self.device_module.stream(high_priority_stream):
    img_query, img_key, img_value = _get_qkv_projections_img(self, hidden_states)
    img_query = img_query.unflatten(-1, (self.num_heads, -1))
    ...
main_stream.wait_stream(high_priority_stream)
```

```python
with self.device_module.stream(high_priority_stream):
    img_mlp_output = self.img_mlp(img_modulated2)[0]
    hidden_states = self.fuse_mul_add(img_mlp_output, img_gate2, hidden_states)
```

- Reviewed files: `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- Validation implications: this should be benchmarked per GPU family. B200 gain does not imply H200 gain; keep the env toggle off by default unless validated for target hardware.

### PR #20447 - TeaCache for GLM-Image, Qwen-Image, Flux and related DiTs

- Link: https://github.com/sgl-project/sglang/pull/20447
- State: open
- Diff stats: `8` files, `+295/-105`
- Diff coverage: sampling params, TeaCache base, Flux/Flux2/GLM-Image/Qwen-Image forward integrations reviewed.
- Motivation: TeaCache can skip expensive denoise steps by reusing residuals when timestep-conditioned hidden-state changes are small. PR body reports large speedups for Qwen-Image-2512 (`156.12s -> 61.29s`) and quality preview outputs.
- Key implementation: add Qwen/GLM/Flux TeaCache params with thresholds and polynomial coefficients, expand CFG-supported prefixes, make TeaCache read `enable_teacache` from forward context, implement default residual caching, and wrap Qwen-Image transformer blocks in `teacache_skip_or_prepare()` / `teacache_finalize()`.
- Key code excerpts:

```python
@dataclass
class QwenImageTeaCacheParams(TeaCacheParams):
    teacache_thresh: float = 0.2
    coefficients: list[float] = field(
        default_factory=lambda: [
            7.33226126e02,
            -4.01131952e02,
            6.75869174e01,
            -3.14987800e00,
            9.61237896e-02,
        ]
    )
```

```python
def teacache_skip_or_prepare(self, hidden_states: torch.Tensor, temb: torch.Tensor):
    if self.should_skip_forward_for_cached_states(temb=temb):
        return True, self.retrieve_cached_states(hidden_states)
    return False, hidden_states.clone() if self.enable_teacache else None
```

```python
skip, hs_or_orig = self.teacache_skip_or_prepare(hidden_states, temb)
if skip:
    hidden_states = hs_or_orig
else:
    ...
    self.teacache_finalize(hidden_states, hs_or_orig)
```

- Reviewed files: `configs/sample/qwenimage.py`, `runtime/cache/teacache.py`, `runtime/models/dits/qwen_image.py`, `glm_image.py`, `flux.py`, `flux_2.py`
- Validation implications: TeaCache is a quality/speed tradeoff. Validate image quality per model and CFG branch; do not claim a speedup without also showing visual or metric agreement.

### PR #20810 - Reland Qwen-Image CUDA graph

- Link: https://github.com/sgl-project/sglang/pull/20810
- State: open
- Diff stats: `4` files, `+681/-47`
- Diff coverage: Qwen-Image graph wrappers, denoising hooks, server args, and new CUDA graph utility reviewed.
- Motivation: relands `#19516` with a safer graph cache design, bounded graph entries, reusable static input buffers, explicit text length buckets, and mutual exclusion with `torch.compile`.
- Key implementation: add `CudaGraphCallableCache`, `SharedStaticInputPool`, padding helpers, and graph replay signature checks. `QwenImageTransformerBlock` routes pre/post attention through `_maybe_graph_*` wrappers and selects a text bucket from `--cuda-graph-txt-lengths` or next power of two. Server args normalize text buckets and reject `--enable-torch-compile` together with `--enable-cuda-graph`.
- Key code excerpts:

```python
class CudaGraphCallableCache:
    def capture_or_replay(self, key, fn, example_inputs, call_inputs=None, input_buffer_key=None):
        ...
        with torch.cuda.graph(graph, pool=self._get_pool_handle()):
            output = fn(*static_inputs)
        ...
        entry.graph.replay()
        return entry.output
```

```python
def _select_text_graph_bucket(self, seq_len: int) -> int | None:
    if self._cuda_graph_text_buckets is None:
        return shape_with_next_power_of_2((seq_len,), dim=0)[0]
    for capture_length in self._cuda_graph_text_buckets:
        if capture_length >= seq_len:
            return capture_length
    return None
```

```python
if self.enable_torch_compile and self.enable_cuda_graph:
    raise ValueError(
        "--enable-torch-compile and --enable-cuda-graph are mutually exclusive for diffusion runtime"
    )
```

- Reviewed files: `qwen_image.py`, `denoising.py`, `server_args.py`, `runtime/utils/cuda_graph.py`
- Validation implications: this is the CUDA graph design to prefer over `#19516`. Validate bucket fallback, replay signature mismatch errors, graph memory, image equality, and interaction with dynamic prompt lengths.

### PR #21988 - Qwen-Image conditional batch multi-output fix

- Link: https://github.com/sgl-project/sglang/pull/21988
- State: open
- Diff stats: `1` file, `+45/-2`
- Diff coverage: full diff reviewed.
- Motivation: Qwen-Image base generation failed with `num_outputs_per_prompt > 1` because latent samples were expanded while text condition batches stayed at original prompt batch size. The reproduced error was `unsupported tensor shape: torch.Size([2, 3072])`.
- Key implementation: add `_expand_cond_tensor_batch()` and `_expand_cond_batch()` in `QwenImagePipelineConfig`, repeat-interleaving prompt and negative prompt embeddings to `batch.batch_size`, and route positive/negative cond preparation through the expanded tensors.
- Key code excerpts:

```python
if target_batch_size % current_batch_size != 0:
    raise ValueError(
        f"QwenImage expects `{cond_name}` batch size ({current_batch_size}) "
        f"to divide target batch size ({target_batch_size})."
    )
repeat_factor = target_batch_size // current_batch_size
return tensor.repeat_interleave(repeat_factor, dim=0).contiguous()
```

```python
def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
    return self._prepare_cond_kwargs(
        batch, self.get_pos_prompt_embeds(batch), rotary_emb, device, dtype
    )
```

- Reviewed files: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- Validation implications: multi-output should be tested at `n=1,2,4,8` for 512 and 1024 resolutions, with VAE decode OOM tracked separately from denoise correctness.

### PR #22362 - Qwen-Image-Layered serve endpoint fix

- Link: https://github.com/sgl-project/sglang/pull/22362
- State: open
- Diff stats: `2` files, `+4/-2`
- Diff coverage: full diff reviewed.
- Motivation: `sglang generate` worked for Qwen-Image-Layered without a prompt, but `/v1/images/edits` required `prompt`. Server output also defaulted to JPEG, which cannot save RGBA images with alpha.
- Key implementation: make image edits `prompt` default to `" "` like `DiffGenerator._resolve_prompts`, and default output image extension to PNG unless the user explicitly asks for another supported format.
- Key code excerpts:

```python
# like DiffGenerator._resolve_prompts, use " " as default
prompt: str = Form(" "),
```

```python
if (background or "auto").lower() == "transparent":
    return "png"
# the default format should be png, same logical with DataType.get_default_extension
return "png"
```

- Reviewed files: `runtime/entrypoints/openai/image_api.py`, `runtime/entrypoints/openai/utils.py`
- Validation implications: test both CLI and OpenAI image edit endpoint for layered/RGBA outputs. PNG default is part of the serving contract.

### PR #22397 - Qwen-Image weight-name mapping for `to_out` and added QKV

- Link: https://github.com/sgl-project/sglang/pull/22397
- State: open
- Diff stats: `1` file, `+20/-0`
- Diff coverage: full diff reviewed.
- Motivation: some Qwen-Image checkpoints store attention output as `transformer_blocks.N.attn.to_out.{weight|bias}` while SGLang exposes `to_out.0.{weight|bias}`. Added Q/K/V projections can also be separate in checkpoint form but fused as `to_added_qkv` in SGLang.
- Key implementation: add anchored mapping rules before generic mappings: flat `to_out` maps to indexed `to_out.0`, and `add_q_proj`, `add_k_proj`, `add_v_proj` merge into `to_added_qkv` with shard ids `0/1/2`.
- Key code excerpt:

```python
r"^(transformer_blocks\.[0-9]+\.attn\.to_out)\.(weight|bias)$": r"\1.0.\2",
r"^(transformer_blocks\.(\d+)\.attn)\.add_q_proj\.(.+)$": (
    r"\1.to_added_qkv.\3",
    0,
    3,
),
```

- Reviewed files: `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`
- Validation implications: loader tests should cover flat vs indexed `to_out` and split vs fused added-QKV checkpoint formats.

### PR #22953 - Avoid illegal memory access in Qwen-Image RoPE

- Link: https://github.com/sgl-project/sglang/pull/22953
- State: merged at `2026-04-23T04:41:27Z`
- Diff stats: `1` file, `+12/-0`
- Diff coverage: full diff fetched with `gh pr diff --patch`, `32` lines, `1` file; current-main source rechecked at `bca3dd958`.
- Motivation: Qwen-Image-Edit-2511 can hit CUDA illegal memory access when too many input images and long prompts make text sequence length exceed the RoPE text cache length. Failing inside the CUDA kernel corrupts the GPU context and hides the real validation issue.
- Key implementation: before building FlashInfer RoPE cos/sin caches, compare `max(txt_seq_lens)` with `txt_freqs.shape[0]` and raise a clear `ValueError` with required length, cache length, overflow, and remediation hints.
- Key code excerpt:

```python
max_txt_seq_len = max(txt_seq_lens) if txt_seq_lens else 0
txt_cache_len = int(txt_freqs.shape[0])
if max_txt_seq_len > txt_cache_len:
    overflow = max_txt_seq_len - txt_cache_len
    raise ValueError(
        "QwenImage RoPE text cache overflow before denoising: "
        f"required_txt_seq_len={max_txt_seq_len}, txt_cache_len={txt_cache_len}, "
        f"overflow={overflow}. "
        "Please reduce the number of input images, shorten the prompt, "
        "or lower the requested resolution."
    )
```

- Reviewed files: `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- Validation implications: large multi-image edit requests should fail fast before entering RoPE kernels. This guard is present in current SGLang main; add a negative test for too many images or overly long prompts.

### PR #23155 - Qwen Image ModelOpt FP8 support

- Link: https://github.com/sgl-project/sglang/pull/23155
- State: open
- Diff stats: `4` files, `+210/-33`
- Diff coverage: diffusion quantization docs, local ModelOpt quant skill, Qwen-Image DiT quant-aware modules, and FP8 converter reviewed.
- Motivation: Qwen-Image and Qwen-Image-Edit need ModelOpt FP8 transformer override support. A naive FP8 conversion caused severe dark/blurred image-quality regression, so a validated BF16 fallback set is required.
- Key implementation: make Qwen-Image attention, MLP, `img_in`, `txt_in`, and `proj_out` use SGLang quant-aware `ReplicatedLinear` with full prefixes. Replace Diffusers `FeedForward` with `QwenImageFeedForward`. Add a Qwen-Image fallback profile in `build_modelopt_fp8_transformer.py`, canonicalize `.img_mod.1` / `.txt_mod.1`, and write explicit BF16 fallback tensors before ignore-preservation skips source tensors. Docs add published FP8 checkpoints and commands.
- Key code excerpts:

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
class QwenImageFeedForward(nn.Module):
    self.net = nn.ModuleList(
        [
            QwenImageGELU(..., prefix=f"{prefix}.net.0"),
            nn.Dropout(0.0),
            ReplicatedLinear(..., prefix=f"{prefix}.net.2"),
        ]
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
if name.endswith(".weight") and is_ignored_by_modelopt(name, ignore_patterns):
    preserved_ignored_weight_count += 1
    continue
```

- Reviewed files: `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`, diffusion ModelOpt quant skill.
- Validation implications: FP8 support requires BF16-vs-FP8 image comparison, benchmark JSON, and profiler artifacts. Validate both Qwen-Image and Qwen-Image-Edit; do not trust startup alone.

## Cookbook and Public Docs Evidence

- sgl-cookbook `#49`, `#55`, `#60`, `#103`: diffusion benchmark/model initialization, docs restructuring, and command generator groundwork.
- sgl-cookbook `#146`: Qwen-Image-Edit AMD MI300X/MI325X/MI355X support.
- sgl-cookbook `#147`: Qwen-Image AMD MI300X/MI325X/MI355X support.
- SGLang Diffusion public docs/blogs cover OpenAI-compatible API, CLI, Python interface, Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-2511, Qwen-Image-2512, Qwen-Image-Layered, GLM-Image, and LoRA API coverage.

## Validation Notes

- Most Qwen-Image optimization PRs are open. #22953 is merged and is now current-main behavior; re-check the remaining open PR state before implementing against them.
- Treat CUDA graph, TeaCache, dual stream, ModelOpt FP8, RoPE/RMSNorm fusion, LN/modulate fusion, and conditional batching as independent toggles.
- Every quality-affecting change needs fixed prompt/seed/resolution/steps, saved BF16 output, saved optimized output, latency, memory, and any profiler artifact used for the claim.
