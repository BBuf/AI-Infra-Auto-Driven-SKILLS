# SGLang Qwen-Image 支持与优化 PR 历史

本文覆盖 Qwen-Image、Qwen-Image-Edit、Qwen-Image-Layered、CUDA graph、TeaCache、conditional batch、ModelOpt FP8、AMD diffusion kernel 等路径。下列 PR 均已打开 GitHub diff 阅读，并按 motivation、关键实现和关键代码片段回填。

证据快照：

- SGLang `origin/main`: `bca3dd958` (`2026-04-24`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- 手工 diff 阅读日期：`2026-04-24`
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
- 状态：已合入，`2026-04-23T04:41:27Z`，`1` file，`+12/-0`
- Diff 覆盖：完整 diff `32` 行；已在 SGLang `bca3dd958` 主线源码复查。
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

- 验证含义：长 prompt / 多图输入应 fail fast，不应污染 CUDA context。该保护已经是当前 SGLang main 行为。

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

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `Qwen-Image`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
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

### 逐 PR 代码 diff 阅读记录

### PR #18530 - [Diffusion] [AMD] fuse norm & rope for qwen-image

- 链接：https://github.com/sgl-project/sglang/pull/18530
- 状态/时间：`open`，created 2026-02-10；作者 `qichu-yun`。
- 代码 diff 已读范围：`2` 个文件，`+95/-35`；代码面：model wrapper, multimodal/processor；关键词：attention, cache, cuda, flash, kv, mla, quant。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +94/-35 (129 lines); hunk: import functools; ); 符号: _get_qkv_projections, forward
  - `python/sglang/srt/environ.py` modified +1/-0 (1 lines); hunk: class Envs:; 符号: Envs:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py`；patch 关键词为 attention, cache, cuda, flash, kv, mla。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/srt/environ.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19066 - [diffusion] model: optimize Qwen2.5-VL ViT and text encoder

- 链接：https://github.com/sgl-project/sglang/pull/19066
- 状态/时间：`open`，created 2026-02-20；作者 `Jasen2201`。
- 代码 diff 已读范围：`7` 个文件，`+874/-21`；代码面：model wrapper, multimodal/processor, tests/benchmarks；关键词：config, cuda, attention, flash, spec, vision, cache, kv, quant, test。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py` modified +451/-16 (467 lines); hunk: # Copied and adapted from: https://github.com/hao-ai-lab/FastVideo; from sglang.multimodal_gen.configs.models.encoders.qwen_image import Qwen2_5VLConfig; 符号: Qwen2_5_VLVisionAttention, __init__, forward, Qwen2_5_VLVisionBlock
  - `test/unit/test_qwen25_vit.py` added +238/-0 (238 lines); hunk: +"""; 符号: _calculate_dimensions, _pixel_to_grid, _build_grid_cases, _ensure_tp_initialized
  - `python/sglang/multimodal_gen/runtime/models/encoders/vision.py` modified +140/-1 (141 lines); hunk: # Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/models/vision.py; def get_patch_grid_length(self) -> int:; 符号: get_patch_grid_length, get_vit_attn_backend, _vit_flash_attn_varlen, _vit_sdpa_varlen
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/image_encoding.py` modified +37/-0 (37 lines); hunk: from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution; def __init__(; 符号: __init__, _maybe_compile_text_encoder, load_model
  - `python/sglang/multimodal_gen/envs.py` modified +5/-0 (5 lines); hunk: VERBOSE: bool = False; def _getter():; 符号: _getter
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`, `test/unit/test_qwen25_vit.py`, `python/sglang/multimodal_gen/runtime/models/encoders/vision.py`；patch 关键词为 config, cuda, attention, flash, spec, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py`, `test/unit/test_qwen25_vit.py`, `python/sglang/multimodal_gen/runtime/models/encoders/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19516 - [Diffusion] add cuda graph support for Qwen-Image

- 链接：https://github.com/sgl-project/sglang/pull/19516
- 状态/时间：`open`，created 2026-02-27；作者 `zyksir`。
- 代码 diff 已读范围：`3` 个文件，`+315/-36`；代码面：model wrapper, multimodal/processor；关键词：cuda, attention, scheduler, cache, config, kv, mla, processor, quant。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +288/-35 (323 lines); hunk: ); def forward(; 符号: forward, pad_tensor_into_power_of_2, QwenImageTransformerBlock, __init__
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +19/-1 (20 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, _maybe_enable_cuda_graph_capture, _maybe_enable_torch_compile
  - `python/sglang/multimodal_gen/runtime/server_args.py` modified +8/-0 (8 lines); hunk: class ServerArgs:; def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:; 符号: ServerArgs:, add_cli_args
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/runtime/server_args.py`；patch 关键词为 cuda, attention, scheduler, cache, config, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, `python/sglang/multimodal_gen/runtime/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19521 - [Feature] Adding detectors for Qwen diffusion models

- 链接：https://github.com/sgl-project/sglang/pull/19521
- 状态/时间：`open`，created 2026-02-27；作者 `shenoyvvarun`。
- 代码 diff 已读范围：`1` 个文件，`+22/-1`；代码面：multimodal/processor；关键词：config。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/registry.py` modified +22/-1 (23 lines); hunk: def _register_configs():; 符号: _register_configs
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/registry.py`；patch 关键词为 config。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/registry.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20429 - [Diffusion][Qwen-Image] Kernel fusion on layernorm and fuse_scale_shift_gate_select01

- 链接：https://github.com/sgl-project/sglang/pull/20429
- 状态/时间：`open`，created 2026-03-12；作者 `SYChen123`。
- 代码 diff 已读范围：`2` 个文件，`+350/-22`；代码面：model wrapper, kernel, multimodal/processor；关键词：triton, attention, config, cuda。
- 代码 diff 细节：
  - `python/sglang/jit_kernel/diffusion/triton/scale_shift.py` modified +301/-3 (304 lines); hunk: +from typing import Optional, Tuple; def fuse_scale_shift_gate_select01_kernel_blc_opt(; 符号: fuse_scale_shift_gate_select01_kernel_blc_opt, fuse_scale_shift_gate_select01_kernel_blc_opt, fuse_scale_shift_gate_select01_kernel, _fused_modulate_kernel
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +49/-19 (68 lines); hunk: from sglang.jit_kernel.diffusion.triton.scale_shift import (; logger = init_logger(__name__) # pylint: disable=invalid-name; 符号: _modulate
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`；patch 关键词为 triton, attention, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/jit_kernel/diffusion/triton/scale_shift.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20432 - [Diffusion][Qwen-Image] Dual stream forward for Qwen-Image

- 链接：https://github.com/sgl-project/sglang/pull/20432
- 状态/时间：`open`，created 2026-03-12；作者 `SYChen123`。
- 代码 diff 已读范围：`1` 个文件，`+232/-26`；代码面：model wrapper, multimodal/processor；关键词：attention, cache, cuda, flash, kv, processor。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +232/-26 (258 lines); hunk: # SPDX-License-Identifier: Apache-2.0; ); 符号: _get_or_create_alt_stream, _get_qkv_projections_img, _get_qkv_projections_txt, _get_qkv_projections
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`；patch 关键词为 attention, cache, cuda, flash, kv, processor。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20447 - [Diffusion] Feat: support teacache for glm-image, qwen-image et.al

- 链接：https://github.com/sgl-project/sglang/pull/20447
- 状态/时间：`open`，created 2026-03-12；作者 `RuixiangMa`。
- 代码 diff 已读范围：`8` 个文件，`+295/-105`；代码面：model wrapper, multimodal/processor, scheduler/runtime, docs/config；关键词：cache, config, attention, processor, kv, quant, spec, triton。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/cache/teacache.py` modified +107/-19 (126 lines); hunk: def forward(self, hidden_states, timestep, ...):; def _init_teacache_state(self) -> None:; 符号: forward, _init_teacache_state, _init_teacache_state, _update_teacache_status
  - `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py` modified +38/-29 (67 lines); hunk: import torch; def forward(; 符号: forward, forward
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +34/-24 (58 lines); hunk: import functools; def forward(; 符号: forward, forward
  - `python/sglang/multimodal_gen/runtime/models/dits/flux.py` modified +27/-20 (47 lines); hunk: import torch; def __init__(; 符号: __init__, forward, forward
  - `python/sglang/multimodal_gen/configs/sample/glmimage.py` modified +30/-1 (31 lines); hunk: -from dataclasses import dataclass; class GlmImageSamplingParams(SamplingParams):; 符号: GlmImageTeaCacheParams, GlmImageSamplingParams
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/cache/teacache.py`, `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`；patch 关键词为 cache, config, attention, processor, kv, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/cache/teacache.py`, `python/sglang/multimodal_gen/runtime/models/dits/flux_2.py`, `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20810 - [Diffusion] Reland qwen image cuda graph

- 链接：https://github.com/sgl-project/sglang/pull/20810
- 状态/时间：`open`，created 2026-03-18；作者 `BBuf`。
- 代码 diff 已读范围：`4` 个文件，`+681/-47`；代码面：model wrapper, kernel, multimodal/processor；关键词：cuda, attention, cache, config, quant, scheduler, kv, mla, processor。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +368/-46 (414 lines); hunk: ); def forward(; 符号: forward, QwenImageTransformerBlock, _get_shared_cuda_graph_pool_handle, _get_shared_cuda_graph_input_pool
  - `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py` added +240/-0 (240 lines); hunk: +from __future__ import annotations; 符号: pad_tensor_along_dim, pad_tensor_to_next_power_of_2, replace_shape_dim, shape_with_next_power_of_2
  - `python/sglang/multimodal_gen/runtime/server_args.py` modified +46/-0 (46 lines); hunk: class ServerArgs:; def _adjust_parameters(self):; 符号: ServerArgs:, _adjust_parameters, _validate_parameters, _adjust_save_paths
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` modified +27/-1 (28 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, _maybe_enable_cuda_graph_capture, _maybe_enable_torch_compile
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`, `python/sglang/multimodal_gen/runtime/server_args.py`；patch 关键词为 cuda, attention, cache, config, quant, scheduler。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `python/sglang/multimodal_gen/runtime/utils/cuda_graph.py`, `python/sglang/multimodal_gen/runtime/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21988 - [diffusion] fix: align qwen-image cond batch for multi-output generation

- 链接：https://github.com/sgl-project/sglang/pull/21988
- 状态/时间：`open`，created 2026-04-03；作者 `MikukuOvO`。
- 代码 diff 已读范围：`1` 个文件，`+45/-2`；代码面：multimodal/processor, docs/config；关键词：cache, config。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` modified +45/-2 (47 lines); hunk: def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):; def _prepare_cond_kwargs(self, batch, prompt_embeds, rotary_emb, device, dtype):; 符号: get_freqs_cis, _expand_cond_tensor_batch, _expand_cond_batch, get_pos_prompt_embeds
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`；patch 关键词为 cache, config。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22362 - [Fix] fix error in sglang serve for qwen-image-layer

- 链接：https://github.com/sgl-project/sglang/pull/22362
- 状态/时间：`open`，created 2026-04-08；作者 `XingyongCheng`。
- 代码 diff 已读范围：`2` 个文件，`+3/-2`；代码面：multimodal/processor；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py` modified +2/-1 (3 lines); hunk: def choose_output_image_ext(; 符号: choose_output_image_ext, build_sampling_params
  - `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py` modified +1/-1 (2 lines); hunk: async def edits(; 符号: edits
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`, `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py`；patch 关键词为 n/a。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py`, `python/sglang/multimodal_gen/runtime/entrypoints/openai/image_api.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22397 - [BugFix] qwenimage: fix weight-name mapping for to_out and added QKV projections

- 链接：https://github.com/sgl-project/sglang/pull/22397
- 状态/时间：`open`，created 2026-04-08；作者 `jy-song-hub`。
- 代码 diff 已读范围：`1` 个文件，`+20/-0`；代码面：model wrapper, multimodal/processor, docs/config；关键词：attention, config, kv, lora, quant, spec。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py` modified +20/-0 (20 lines); hunk: class QwenImageArchConfig(DiTArchConfig):; 符号: QwenImageArchConfig
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py`；patch 关键词为 attention, config, kv, lora, quant, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/configs/models/dits/qwenimage.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22953 - [diffusion][bugfix] avoid illegal memory access in qwen image

- 链接：https://github.com/sgl-project/sglang/pull/22953
- 状态/时间：`merged`，created 2026-04-16, merged 2026-04-23；作者 `IPostYellow`。
- 代码 diff 已读范围：`1` 个文件，`+12/-0`；代码面：multimodal/processor, docs/config；关键词：cache, config, flash。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` modified +12/-0 (12 lines); hunk: def get_freqs_cis(img_shapes, txt_seq_lens, rotary_emb, device, dtype):; 符号: get_freqs_cis
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py`；patch 关键词为 cache, config, flash。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/configs/pipeline_configs/qwen_image.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #23155 - [Diffusion] Add Qwen Image ModelOpt FP8 support

- 链接：https://github.com/sgl-project/sglang/pull/23155
- 状态/时间：`open`，created 2026-04-19；作者 `BBuf`。
- 代码 diff 已读范围：`4` 个文件，`+210/-33`；代码面：model wrapper, quantization, multimodal/processor, docs/config；关键词：config, fp8, quant, doc, fp4, kv, attention, benchmark, flash, spec。
- 代码 diff 细节：
  - `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py` modified +124/-20 (144 lines); hunk: import torch; def __init__(; 符号: __init__, __init__, forward, QwenImageGELU
  - `docs/diffusion/quantization.md` modified +36/-6 (42 lines); hunk: backend.; sglang generate \
  - `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md` modified +29/-2 (31 lines); hunk: This repo now contains:; For `FLUX.1-dev`, the validated fallback set currently keeps these modules in BF
  - `python/sglang/multimodal_gen/tools/build_modelopt_fp8_transformer.py` modified +21/-5 (26 lines); hunk: r"^transformer_blocks\.(0\|43\|44\|45\|46\|47)\.(attn1\|attn2\|audio_attn1\|audio_attn2\|audio_to_video_attn\|video_to_audio_attn)\.to_out\.0$",; def _module_name_variant; 符号: _resolve_transformer_dir, _module_name_variants, get_default_keep_bf16_patterns, build_modelopt_fp8_transformer
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md`；patch 关键词为 config, fp8, quant, doc, fp4, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py`, `docs/diffusion/quantization.md`, `python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-modelopt-quant/SKILL.md` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：13；open PR 数：12。
- 仍需跟进的 open PR：[#18530](https://github.com/sgl-project/sglang/pull/18530), [#19066](https://github.com/sgl-project/sglang/pull/19066), [#19516](https://github.com/sgl-project/sglang/pull/19516), [#19521](https://github.com/sgl-project/sglang/pull/19521), [#20429](https://github.com/sgl-project/sglang/pull/20429), [#20432](https://github.com/sgl-project/sglang/pull/20432), [#20447](https://github.com/sgl-project/sglang/pull/20447), [#20810](https://github.com/sgl-project/sglang/pull/20810), [#21988](https://github.com/sgl-project/sglang/pull/21988), [#22362](https://github.com/sgl-project/sglang/pull/22362), [#22397](https://github.com/sgl-project/sglang/pull/22397), [#23155](https://github.com/sgl-project/sglang/pull/23155)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
