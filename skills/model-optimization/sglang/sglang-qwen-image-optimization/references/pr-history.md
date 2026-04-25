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

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG Qwen-Image PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

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

## Diff Cards

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


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
