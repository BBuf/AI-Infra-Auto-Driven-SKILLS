# SGLang Qwen-Image 支持与优化 PR 历史

本文覆盖 Qwen-Image、Qwen-Image-Edit、Qwen-Image-Layered、CUDA graph、TeaCache、conditional batch、ModelOpt FP8、AMD diffusion kernel 等路径。下列 PR 均已打开 GitHub diff 阅读，并按 motivation、关键实现和关键代码片段回填。

证据快照：

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- 手工 diff 阅读日期：`2026-04-23`
- 对应 skill：`skills/model-optimization/sglang/sglang-qwen-image-optimization`
- 详细 PR 卡片：`skills/model-optimization/sglang/sglang-qwen-image-optimization/references/pr-history.md`

## 关键代码面

- `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`
- `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`
- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`
- `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`
- `python/sglang/multimodal_gen/runtime/cache/teacache.py`
- `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`
- `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py`
- `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`
- `docs/diffusion/quantization.md`

## PR 卡片

### #18530 - AMD fuse norm & RoPE for Qwen-Image

- 链接：https://github.com/sgl-project/sglang/pull/18530
- 状态：Open，`2` files，`+95/-35`
- Motivation：Qwen-Image cross-attention 在 AMD ROCm 上把 Q/K RMSNorm 和 RoPE 分开执行，PR 通过 AITER 融合这两步，Qwen-Image-Edit latency 从 `84.48s` 降到 `79.72s`。
- 实现思路：新增 `SGLANG_ENABLE_FUSED_ROPE_RMS_2WAY`，在 AITER + HIP + RMSNorm + 有 image rotary embedding 时调用 `aiter.fused_rope_rms_2way` 直接生成 joint query/key。
- 关键代码：

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

- 验证含义：只对 AMD/AITER 路径有效，必须做图像质量和 denoise latency 对比。

### #19066 - Qwen2.5-VL ViT/text encoder 优化

- 链接：https://github.com/sgl-project/sglang/pull/19066
- 状态：Open，`7` files，`+874/-21`
- Motivation：Qwen-Image diffusion 相邻路径使用 Qwen2.5-VL 视觉/text encoder。HF ViT 路径不够 torch.compile-friendly，AMD SDPA 非 contiguous q/k/v 会累积数值误差。
- 实现思路：实现自定义 `Qwen2_5_VisionTransformer`，统一 ViT attention backend，SDPA 切片 `.contiguous()`，缓存 RoPE/window index，ViT MLP fuse `gate_up_proj`，text encoder 也纳入 torch.compile。
- 关键代码：

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

- 验证含义：需要 custom ViT vs HF ViT 数值一致性测试，再跑 Qwen-Image/Edit e2e。

### #19516 - Qwen-Image CUDA graph 初版

- 链接：https://github.com/sgl-project/sglang/pull/19516
- 状态：Open，`3` files，`+315/-36`
- Motivation：Qwen-Image profile 里 CPU overhead 约 `30%`。diffusion 输入形状跟分辨率和 prompt length 绑定，不能像 LLM 那样简单全量预建图。
- 实现思路：拆分 DiT block 的 image/text pre/post attention 函数，用 `torch.cuda.make_graphed_callables` 捕获；text 长度 pad 到 next power-of-two，image 保持真实 shape。
- 关键代码：

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

- 验证含义：这是设计历史；实际应优先看 reland 版本 #20810。

### #19521 - Qwen diffusion model detectors

- 链接：https://github.com/sgl-project/sglang/pull/19521
- 状态：Open，`1` file，`+22/-1`
- Motivation：本地模型目录名不含官方模型名时，diffusion registry 无法识别 Qwen-Image pipeline。
- 实现思路：给 Qwen-Image、Qwen-Image-Edit、Qwen-Image-Layered 等注册基于 Diffusers pipeline class name 的 detector。
- 关键代码：

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

- 验证含义：测试 `/models/foobar` 这类本地路径，而不是只测 HF 名称。

### #20429 - LayerNorm + scale/shift/gate select01 fusion

- 链接：https://github.com/sgl-project/sglang/pull/20429
- 状态：Open，`2` files，`+350/-22`
- Motivation：Qwen-Image `_modulate()` 中 residual、LayerNorm、select01、scale/shift/gate 分多 kernel 执行。
- 实现思路：新增 Triton `_fused_modulate_kernel`，可选计算 residual，执行 LayerNorm，根据 index 选择两套调制参数，输出 modulated tensor 和 gate。
- 关键代码：

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

- 验证含义：固定 seed 比较 fused/unfused 图像；检查 fp16 clip 和 hidden dim 上限。

### #20432 - Qwen-Image dual-stream forward

- 链接：https://github.com/sgl-project/sglang/pull/20432
- 状态：Open，`1` file，`+232/-26`
- Motivation：B200 上 text qkv/feedforward 可被 image qkv/feedforward overlap，E2E latency 从 `7.83s` 到 `7.63s`；H200 收益很小，所以必须用 env 控制。
- 实现思路：新增 `QWEN_IMAGE_DUAL_STREAM_FORWARD`，创建 high-priority stream，image qkv/MLP 走 high-priority stream，text 保持 default stream，最后同步。
- 关键代码：

```python
_DUAL_STREAM_FORWARD = os.environ.get("QWEN_IMAGE_DUAL_STREAM_FORWARD", "0") == "1"
```

```python
with self.device_module.stream(high_priority_stream):
    img_query, img_key, img_value = _get_qkv_projections_img(self, hidden_states)
...
main_stream.wait_stream(high_priority_stream)
```

- 验证含义：按 GPU 型号验证；B200 收益不能外推到 H200。

### #20447 - TeaCache for Qwen-Image / GLM-Image / Flux

- 链接：https://github.com/sgl-project/sglang/pull/20447
- 状态：Open，`8` files，`+295/-105`
- Motivation：TeaCache 通过 timestep-conditioned residual 复用跳过部分 denoise block，PR 里 Qwen-Image-2512 从 `156.12s` 到 `61.29s`。
- 实现思路：给 QwenImageSamplingParams 增加 TeaCacheParams；TeaCache 基类读取 forward context 中的 `enable_teacache`，支持 CFG branch 分离；Qwen-Image forward 用 `teacache_skip_or_prepare` 和 `teacache_finalize` 包裹 transformer blocks。
- 关键代码：

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

- 验证含义：TeaCache 是质量/速度 tradeoff，必须同时保存图像和 latency。

### #20810 - Reland Qwen-Image CUDA graph

- 链接：https://github.com/sgl-project/sglang/pull/20810
- 状态：Open，`4` files，`+681/-47`
- Motivation：重做 #19516，用更安全的 graph cache、静态输入池、text bucket 和 replay signature 检查。
- 实现思路：新增 `CudaGraphCallableCache`、`SharedStaticInputPool`、padding helpers；block 里的 pre/post attention 都走 `_maybe_graph_*`；`--cuda-graph-txt-lengths` 控制 text bucket；和 torch.compile 互斥。
- 关键代码：

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

- 验证含义：优先参考这个 CUDA graph 设计；测试 bucket fallback、graph memory、图像一致性。

### #21988 - conditional batch multi-output 修复

- 链接：https://github.com/sgl-project/sglang/pull/21988
- 状态：Open，`1` file，`+45/-2`
- Motivation：`num_outputs_per_prompt > 1` 时 latent batch 扩大，但 text condition batch 未扩大，导致 denoise shape mismatch。
- 实现思路：在 `QwenImagePipelineConfig` 中将 prompt/negative prompt embeds 通过 `repeat_interleave` 扩到 `batch.batch_size`。
- 关键代码：

```python
repeat_factor = target_batch_size // current_batch_size
return tensor.repeat_interleave(repeat_factor, dim=0).contiguous()
```

- 验证含义：测试 `num_outputs_per_prompt=1/2/4/8`，区分 denoise 修复和 VAE OOM。

### #22362 - Qwen-Image-Layered serve 修复

- 链接：https://github.com/sgl-project/sglang/pull/22362
- 状态：Open，`2` files，`+4/-2`
- Motivation：`sglang serve` 的 `/v1/images/edits` 强制要求 prompt，而 Layered 任务可无文本；RGBA 输出默认 jpg 会失败。
- 实现思路：OpenAI image edit endpoint 的 prompt 默认 `" "`；默认图片输出扩展名改为 PNG。
- 关键代码：

```python
prompt: str = Form(" ")
```

```python
return "png"
```

- 验证含义：CLI 和服务端 layered/RGBA 输出都要测。

### #22397 - Qwen-Image weight-name mapping

- 链接：https://github.com/sgl-project/sglang/pull/22397
- 状态：Open，`1` file，`+20/-0`
- Motivation：有些 checkpoint 保存 `attn.to_out.weight`，而 SGLang 模型是 `attn.to_out.0.weight`；added Q/K/V 也可能是拆开的。
- 实现思路：在 `QwenImageArchConfig.param_names_mapping` 最前面加精确规则，把 flat `to_out` 映射到 indexed `to_out.0`，把 `add_q/k/v_proj` 按 shard 0/1/2 merge 到 `to_added_qkv`。
- 关键代码：

```python
r"^(transformer_blocks\.[0-9]+\.attn\.to_out)\.(weight|bias)$": r"\1.0.\2",
r"^(transformer_blocks\.(\d+)\.attn)\.add_q_proj\.(.+)$": (
    r"\1.to_added_qkv.\3",
    0,
    3,
),
```

- 验证含义：loader 测试要覆盖 flat/indexed `to_out` 与 split/fused added QKV。

### #22953 - 避免 Qwen-Image RoPE CUDA illegal memory access

- 链接：https://github.com/sgl-project/sglang/pull/22953
- 状态：Open，`1` file，`+12/-0`
- Motivation：Qwen-Image-Edit-2511 输入图太多、prompt 过长时 text seq len 超过 RoPE text cache，进入 CUDA kernel 后会 illegal memory access。
- 实现思路：在进入 FlashInfer RoPE 前检查 `max(txt_seq_lens)` 是否超过 `txt_freqs.shape[0]`，提前抛出明确 `ValueError`。
- 关键代码：

```python
if max_txt_seq_len > txt_cache_len:
    overflow = max_txt_seq_len - txt_cache_len
    raise ValueError(
        "QwenImage RoPE text cache overflow before denoising: "
        f"required_txt_seq_len={max_txt_seq_len}, txt_cache_len={txt_cache_len}, "
        f"overflow={overflow}. "
    )
```

- 验证含义：长 prompt / 多图输入应 fail fast，不应污染 CUDA context。

### #23155 - Qwen Image ModelOpt FP8

- 链接：https://github.com/sgl-project/sglang/pull/23155
- 状态：Open，`4` files，`+210/-33`
- Motivation：Qwen-Image / Qwen-Image-Edit 需要 ModelOpt FP8 transformer override；直接 FP8 会出现明显暗、糊质量回退，所以需要 BF16 fallback profile。
- 实现思路：Qwen-Image attention、MLP、`img_in`、`txt_in`、`proj_out` 改为 quant-aware `ReplicatedLinear`；新增 `QwenImageFeedForward`；converter 增加 Qwen fallback patterns，并确保 fallback tensor 先写入再处理 ModelOpt ignored weights。
- 关键代码：

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

- 验证含义：必须做 BF16 vs FP8 图像对比、benchmark JSON 和 profiler；启动成功不等于支持完成。

## 验证矩阵

- BF16 text-to-image 固定 prompt/seed/resolution/steps。
- BF16 edit 固定输入图和 prompt。
- CUDA graph on/off，包含 text bucket fallback。
- TeaCache on/off，保存图像和 latency。
- `num_outputs_per_prompt=1/2/4/8`。
- Qwen-Image-Layered `/v1/images/edits` 无 prompt 和 RGBA 输出。
- ModelOpt FP8 Qwen-Image 与 Qwen-Image-Edit。
- AMD AITER fused RoPE/RMSNorm 与 Triton fused modulation。
