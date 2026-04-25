# SGLang GLM VLM/OCR 支持与优化时间线

本文基于 SGLang `origin/main` 快照 `b3e6cf60a`（2026-04-22）和 sgl-cookbook `origin/main` 快照 `816bad5`（2026-04-21）整理，覆盖 GLM-4V、GLM-4.1V、GLM-4.5V、GLM-4.6V、GLM-Glyph、GLM-OCR。

本轮重新整理时逐个读取了相关 PR 的源码 diff。每张卡片都按同一格式记录：motivation/root cause、关键实现思路、关键代码片段、验证信号和风险。

## 结论

GLM VLM/OCR 的主风险不在普通文本 MoE 本身，而在这些边界：

- multimodal processor 是否能注册成功，尤其是 GLM-OCR optional dependency。
- vision encoder 的 TP/DP/PP 切分是否和 12 个 vision heads、dummy heads、pipeline stage 对齐。
- GLM4V MoE、GLM-OCR 是否有独立 loader，不能只检查 `glm4v.py`。
- Transformers 5.x 后字段可能移动到 `text_config`，不能假设老字段总在顶层。
- Conv3D/Linear patch embedding 优化有性能收益，也带来了 loader 和回归链路。
- GLM-OCR 必须用 OCR/MTP/NextN 任务验证，不能只跑图像描述。

## 代码面

- `python/sglang/srt/models/glm4v.py`
- `python/sglang/srt/models/glm4v_moe.py`
- `python/sglang/srt/models/glm_ocr.py`
- `python/sglang/srt/models/glm_ocr_nextn.py`
- `python/sglang/srt/multimodal/processors/glm4v.py`
- `python/sglang/srt/layers/attention/vision.py`
- `python/sglang/srt/layers/rotary_embedding.py`
- `docs_new/docs/basic_usage/glmv.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.5V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-4.6V.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-Glyph.mdx`
- `docs_new/cookbook/autoregressive/GLM/GLM-OCR.mdx`

## 已合入主线 PR

### `#8798`：GLM-4.1V / GLM-4.5V 基础支持

- Motivation：让 SGLang 原生支持 GLM-4.1V Thinking 和 GLM-4.5V，而不是复用一个不完整的 Qwen2.5-VL 路径。这个 PR 补齐模型注册、GLM4V/GLM4V-MoE 模型文件、processor、chat template、MRoPE 和测试。
- 关键实现：新增 `Glm4vForConditionalGeneration` 架构注册；定义 `glm4v` 对话模板；支持多个 vision start token；把 GLM4V 纳入 multimodal RoPE index 逻辑。
- 关键代码片段：

```python
"Glm4vForConditionalGeneration",
```

```python
register_conv_template(
    Conversation(
        name="glm4v",
        image_token="<|begin_of_image|><|image|><|end_of_image|>",
        video_token="<|begin_of_video|><|video|><|end_of_video|>",
    )
)
```

```python
vision_start_token_id = (
    [vision_start_token_id]
    if isinstance(vision_start_token_id, int)
    else vision_start_token_id
)
```

- 验证信号：PR 中覆盖单图、多图、视频输入和 MMMU，GLM4.1V 报告约 `0.701`。
- 风险：GLM4V 和 Qwen VLM 很像，但 token wrapper、MRoPE 和 processor 不是完全一致，不能直接合并假设。

### `#9059`：GLM4.1V/4.5V dummy-head TP

- Motivation：文本侧 `num_key_value_heads=8`，vision attention 有 12 个 heads，TP=8 时无法整除。需要 dummy heads 让 vision attention 也能跑更大的 TP。
- 关键实现：根据 TP size 计算 `num_dummy_heads`，写入 `vision_config`，并在 GLM4V/GLM4V-MoE loader 中 pad q/k/v、proj、norm 权重。
- 关键代码片段：

```python
num_dummy_heads = ((num_heads + tp_size) // tp_size) * tp_size - num_heads
setattr(self.config.vision_config, "num_dummy_heads", num_dummy_heads)
```

```python
if "attn.qkv_proj" in name:
    wq, wk, wv = loaded_weight.chunk(3, dim=0)
    loaded_weight = torch.cat([wq, wk, wv], dim=0)
```

- 验证信号：解决 vision head 数量导致的 TP 上限问题。
- 风险：dummy-head 配置和 loader padding 必须同步，否则要么形状不整除，要么权重错位。

### `#9245`：GLM-4.5V 默认使用 FA3

- Motivation：GLM-4.5V 推荐使用 FA3，但默认 backend 未体现这一点。
- 关键实现：把 `Glm4vMoeForConditionalGeneration` 加入 FA3 默认架构列表。
- 关键代码片段：

```python
"Glm4vMoeForConditionalGeneration",
```

- 验证信号：不改模型数学，只改变默认 backend。
- 风险：平台不支持 FA3 时必须显式指定其它 backend，并在文档/launch generator 里说明。

### `#9554`：修复 GLM45V torch.compile launch

- Motivation：cuda graph + `torch.compile` 启动时 fake tensor 和 `out=` 操作产生 grad 相关错误。
- 关键实现：给 Qwen2.5-VL/GLM45V 共享 forward 路径加 `torch.no_grad()`。
- 关键代码片段：

```python
@torch.no_grad()
def forward(...):
```

- 验证信号：PR 报告 GLM45V cuda graph + torch compile 可以启动，MMMU 不变。
- 风险：GLM4V 的 compile 问题可能出现在共享 VLM 文件，而不是 GLM 文件。

### `#9884`：修复 GLM4V vision block norm 签名

- Motivation：共享 vision block 改成 `norm2(..., residual=attn2d)` 后，GLM4V 自己覆盖的 RMSNorm 不接受 `residual` 参数。
- 关键实现：不再用不兼容的 GLM4V norm 覆盖父类 norm，改用可接受 residual 的路径并传入 `rms_norm_eps`。
- 关键代码片段：

```python
rms_norm_eps=config.rms_norm_eps,
```

- 验证信号：修复继承 Qwen VLM block 后的签名不匹配。
- 风险：VLM block 继承时要查函数签名，不能只看 tensor shape。

### `#10147` / `#10228`：补齐 EAGLE3 字段

- Motivation：EAGLE3/speculative decoding 逻辑期望模型对象上存在 `capture_aux_hidden_states`。
- 关键实现：分别在 GLM4V 和 GLM4V-MoE 上初始化该字段。
- 关键代码片段：

```python
self.capture_aux_hidden_states = False
```

- 验证信号：避免 speculative infrastructure 访问不存在的属性。
- 风险：dense、MoE、OCR 分支都要维持相同 attribute contract。

### `#11166`：`utils.py` 包结构迁移

- Motivation：把 `python/sglang/srt/utils.py` 移到 `utils/` 包内。
- 关键实现：新增 `utils/__init__.py` re-export，同时更新 GLM VLM 导入。
- 关键代码片段：

```python
from .common import *
```

```python
from sglang.srt.utils.hf_transformers_utils import get_processor
```

- 验证信号：结构重排，不改变模型行为。
- 风险：GLM processor 本来就有 optional dependency，import 路径变化也可能影响注册。

### `#11388`：用 `torch.cat` 替换 `F.pad`

- Motivation：`cu_seqlens` 前置 0 的热路径里，`torch.cat` 比 `F.pad` 更轻。
- 关键实现：GLM4V/Qwen VLM 等文件把 prefix zero 改成 `cat`。
- 关键代码片段：

```python
cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
```

- 验证信号：小型热路径优化，不改语义。
- 风险：后续不要在视觉序列构造里又引入更重的 padding。

### `#11922`：ruff 检查增强

- Motivation：pre-commit 参数写法有问题，未稳定检查 F401/F821。
- 关键实现：拆分 ruff args，并自动修复未使用/未定义 import；GLM4V 中补齐缺失 import。
- 关键代码片段：

```yaml
args:
  - --select=F401,F821
  - --fix
```

- 验证信号：import hygiene。
- 风险：看似 lint 的 PR 也会影响 GLM processor 动态导入。

### `#12117`：GLM-4-0414 / GLM-4.1V refactor

- Motivation：把 GLM-4 和 GLM-4.1V 迁移到新接口，去掉旧代码。
- 关键实现：`Glm4vVisionBlock` 改为独立模块；使用 `VisionAttention`；把 multimodal embedding 走 `general_mm_embed_routine`；使用 `MultiModalityDataPaddingPatternMultimodalTokens`；对齐 PP missing-layer。
- 关键代码片段：

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    projection_size=dim,
    use_qkv_parallel=True,
    proj_bias=True,
)
```

```python
x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
x = x_after_add + mlp_out
```

```python
pattern = MultiModalityDataPaddingPatternMultimodalTokens()
return pattern.pad_input_tokens(input_ids, mm_inputs)
```

- 验证信号：后续 DP/PP 和 processor 优化都建立在这个重构后的公共接口上。
- 风险：绕开 `general_mm_embed_routine` 容易破坏 PP proxy tensor 或 image token padding。

### `#13228`：清理 vision attention 相关代码

- Motivation：多个模型里有硬编码/死参数，B200 的 `triton_attn` 等 backend 选择应该集中。
- 关键实现：删除每个模型里的 backend 硬编码，让 `VisionAttention` 统一处理。
- 关键代码片段：

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    projection_size=dim,
    flatten_batch=True,
)
```

- 验证信号：减少 GLM/Qwen 视觉路径分叉。
- 风险：backend 调优应放在公共 attention 或 launch args，而不是单个 GLM block。

### `#14097`：GLM-V vision encoder DP

- Motivation：TP=8 场景下 vision heads 不整除，且 VLM TTFT 受 vision encoder 影响明显，需要 vision encoder data parallel。
- 关键实现：读取 `mm_enable_dp_encoder`；DP 时让 vision merger 使用 TP size/rank `1/0`；图像/视频 feature 通过 `run_dp_sharded_mrope_vision_model(..., rope_type="rope_3d")`。
- 关键代码片段：

```python
self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
```

```python
return run_dp_sharded_mrope_vision_model(
    self.visual, pixel_values, image_grid_thw.tolist(), rope_type="rope_3d"
)
```

```python
self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()
self.tp_rank = 0 if use_data_parallel else get_tensor_model_parallel_rank()
```

- 验证信号：官方 GLM VLM 文档也把 TP=8 与 `--mm-enable-dp-encoder` 绑定说明。
- 风险：DP 改变 feature 生成和聚合方式，必须保留 no-DP baseline。

### `#14720`：GLM-4.6V / GLM-4.1V Pipeline Parallelism

- Motivation：PP 下非最后 rank 没有 `lm_head.weight`，而 GLM4V loader 和 multimodal embedding 路径仍按全模型处理。
- 关键实现：`forward` 接收并传递 `PPProxyTensors`；非最后 PP rank 跳过 `lm_head.*`；当前 stage 不存在的参数直接跳过。
- 关键代码片段：

```python
def forward(..., pp_proxy_tensors: Optional[PPProxyTensors] = None):
    hidden_states = general_mm_embed_routine(..., pp_proxy_tensors=pp_proxy_tensors)
```

```python
if name.startswith("lm_head.") and not self.pp_group.is_last_rank:
    continue
if name not in params_dict:
    continue
```

- 验证信号：新增 GLM4.1V PP accuracy 测试。
- 风险：GLM VLM loader 变更必须考虑 PP stage ownership。

### `#14927`：给 `glm4v_moe` 加 nightly CI

- Motivation：之前 CI 覆盖了 GLM4V dense，但没有覆盖 GLM4V-MoE，GLM-4.5V FP8 容易静默回归。
- 关键实现：把 `zai-org/GLM-4.5V-FP8` 加入 nightly VLM MMMU eval。
- 关键代码片段：

```python
ModelLaunchSettings(
    "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
): ModelEvalMetrics(0.26, 32.0)
```

- 验证信号：为 GLM4V-MoE 增加持续回归保护。
- 风险：后续 loader/quantization/processor 改动不要只跑 dense GLM4V。

### `#14998`：GLM-4.6V MoE Transformers 版本检查

- Motivation：GLM-4.6V MoE 需要 Transformers 5.x，但其它模型不应被无差别要求 TF5。
- 关键实现：通过 model path 或 `vision_config.model_type == "glm4v_moe_vision"` 判断是否 GLM-4.6V MoE，需要 TF5 的模型用旧版本直接报错，其它模型仅 warning。
- 关键代码片段：

```python
is_glm_46vmoe = "glm-4.6v" in self.model_path.lower() or (
    vision_config is not None
    and getattr(vision_config, "model_type", None) == "glm4v_moe_vision"
)
```

- 验证信号：把版本问题提前到 config 阶段。
- 风险：版本判断必须精准，不要把不需要 TF5 的模型误杀。

### `#15205`：Qwen3-VL / GLM-4.1V cos/sin cache

- Motivation：2D vision RoPE 重复计算频率和 `cos()/sin()`，长视频/多图下成为热路径。PR 报告局部从约 `490us` 降到 `186us`。
- 关键实现：`RotaryEmbedding` 提供 `get_cos_sin`；GLM4V visual RoPE 改用 `get_rope`；`rot_pos_emb` 返回缓存的 cos/sin；`VisionAttention` 支持显式 `rotary_pos_emb_cos/sin`。
- 关键代码片段：

```python
def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

```python
rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(grid_thw)
```

- 验证信号：PR 报告 Qwen3-VL MMMU 无下降，VLM cache 场景 TTFT 有改善。
- 风险：GLM 视觉 RoPE 的半维拼接和 grid index 必须保持完全一致。

### `#15434`：NPU 上 `cu_seqlens` 只转 CPU 一次

- Motivation：Ascend/NPU 的 vision attention 每层都把 `cu_seqlens` 转 CPU，造成 dispatch 间隙。
- 关键实现：在 `VisionAttention` 统一解析 CPU seqlens；GLM4V vision forward 中 NPU 场景提前移动一次。
- 关键代码片段：

```python
cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
```

```python
if is_npu():
    cu_seqlens = cu_seqlens.to("cpu")
```

- 验证信号：平台热路径优化，不改变 CUDA。
- 风险：NPU 相关转换不要放在 per-layer 循环里。

### `#17122`：GLM-4V dummy heads / NPU processor bugfix

- Motivation：`VisionAttention` 计算时需要 `num_dummy_heads`，否则 `(num_dummy_heads + num_heads)` 不能按 TP 整除；NPU processor 也需要特殊处理。
- 关键实现：构造 visual 之前调用 `vision_utils.update_vit_attn_dummy_heads_config`；block 接收 `num_dummy_heads`；NPU processor patch 排除 `Glm4vProcessor`。
- 关键代码片段：

```python
vision_utils.update_vit_attn_dummy_heads_config(self.config)
```

```python
num_dummy_heads=vision_config.num_dummy_heads,
```

```python
elif processor.__class__.__name__ not in {"Glm4vProcessor"}:
```

- 验证信号：新增 Ascend GLM-4.5V 测试。
- 风险：dummy-head 不能只在 loader 里处理，模块构造前也要更新 config。

### `#17420`：优化 GLM4V `get_rope_index`

- Motivation：GLM4V 长上下文/多模态输入的 `get_rope_index` 成本高，PR benchmark 显示长 token 场景有显著加速。
- 关键实现：预分配 token type；减少 `.item()`/CPU 往返；按连续 modality 分组；用 device-local arange 和 tensor reduction 生成 MRoPE delta。
- 关键代码片段：

```python
input_token_type = [""] * len(input_tokens)
```

```python
t_index = (
    torch.arange(llm_grid_t, device=position_ids.device)
    .view(-1, 1)
    .expand(llm_grid_t, llm_grid_h * llm_grid_w)
    .reshape(-1)
)
```

```python
max_position_ids = position_ids.amax(dim=0, keepdim=False)
mrope_position_deltas = max_position_ids.amax(-1, keepdim=True) + 1 - attention_mask.shape[-1]
```

- 验证信号：PR 报告 lmms-eval 无下降。
- 风险：必须覆盖图像、多图和视频 token layout。

### `#17582`：GLM-OCR 支持

- Motivation：支持 GLM-OCR，包括 OCR 架构、processor、Transformers 5.x 要求和 NextN/MTP speculative decoding。
- 关键实现：注册 `GlmOcrForConditionalGeneration`；新增 `glm_ocr.py` 与 `glm_ocr_nextn.py`；GLM-OCR vision 使用 `VisionAttention(qk_normalization_by_head_size=True)`；draft 模型映射到 `GlmOcrForConditionalGenerationNextN`；processor 注册 OCR。
- 关键代码片段：

```python
"GlmOcrForConditionalGeneration",
```

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    qk_normalization_by_head_size=True,
    flatten_batch=True,
)
```

```python
if is_draft_model and self.hf_config.architectures[0] in [
    "GlmOcrForConditionalGeneration",
]:
    self.hf_config.architectures[0] = "GlmOcrForConditionalGenerationNextN"
```

- 验证信号：官方 GLM-OCR 文档提供 OCRBench 和 OmniDocBench 方向，部署 generator 支持 EAGLE/MTP。
- 风险：OCR 不能只用 caption smoke，需要 OCR 页面、表格、公式、MTP acceptance 验证。

### `#18885`：`glm_ocr` 不可用时 GLM-4V processor 仍可注册

- Motivation：`#17582` 后，环境里没有 `transformers.models.glm_ocr` 会导致整个 `glm4v.py` processor 模块 import 失败，连 GLM-4.1V/4.5V processor 也被一起丢掉。
- 关键实现：只把 OCR import 包进 `try/except ImportError`，再从 `models` 列表中过滤 `None`。
- 关键代码片段：

```python
try:
    from sglang.srt.models.glm_ocr import GlmOcrForConditionalGeneration
except ImportError:
    GlmOcrForConditionalGeneration = None
```

```python
models = [
    m
    for m in [
        Glm4vForConditionalGeneration,
        Glm4vMoeForConditionalGeneration,
        GlmOcrForConditionalGeneration,
    ]
    if m is not None
]
```

- 验证信号：PR body 指向 nightly 中 GLM-4V “No processor registered” 的失败。
- 风险：optional OCR 依赖不能影响非 OCR 的 GLM VLM。

### `#20033`：GLM4V Conv3D projection 改 Linear

- Motivation：GLM4V patch embedding 的 Conv3D 在输入已经 flatten 的情况下可等价为 Linear，性能更好。
- 关键实现：新增 `linear`；load 后把 Conv3D weight reshape/copy 到 Linear；删除原 `proj`；更新 dtype/device；新增性能测试。
- 关键代码片段：

```python
k = self.in_channels * self.temporal_patch_size * self.patch_size**2
self.linear = nn.Linear(in_features=k, out_features=self.hidden_size, bias=True)
```

```python
def copy_conv3d_weight_to_linear(self):
    with torch.no_grad():
        self.linear.weight.copy_(self.proj.weight.view(self.hidden_size, -1))
        self.linear.bias.copy_(self.proj.bias)
    del self.proj
```

```python
self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- 验证信号：新增 Conv3D/Linear close test 和 CUDA benchmark；PR 报告 lmms_eval 不掉分。
- 风险：GLM4V-MoE/OCR 有独立 loader，不能只在 dense loader 里处理。

### `#20282`：统一 Conv2dLayer/Conv3dLayer

- Motivation：PyTorch 2.9.1 + 旧 CuDNN 有 Conv3D bug，并且 patch embedding 中 kernel=stride 的 Conv 可用 unfold+linear 加速。
- 关键实现：新增 `sglang/srt/layers/conv.py`；检查 conv 是否可线性化；迁移 GLM4V、Qwen VLM 和多个视觉模型；删除全局 server arg compatibility check。
- 关键代码片段：

```python
def _check_enable_linear(kernel_size, stride, padding, dilation, groups) -> bool:
    return (
        kernel_size == stride
        and all(p == 0 for p in padding)
        and all(d == 1 for d in dilation)
        and groups == 1
    )
```

```python
x = x.unfold(2, K1, K1).unfold(3, K2, K2)
x = F.linear(x, self.weight.reshape(self.out_channels, -1), self.bias)
```

- 验证信号：新增 `test/unit/test_conv_layer.py`。
- 风险：这是 `#20033` 后更通用的修复面；历史上 `#20463`/`#20740` 的回归链路要一起看。

### `#20463` 与 `#20740`：MoE/OCR loader 回归修复与回退

- Motivation：`#20033` 只在 `glm4v.py` loader 末尾 copy Conv3D weight 到 Linear，`glm4v_moe.py` 和 `glm_ocr.py` 自己有 loader，导致 Linear 权重可能保持随机，视觉输出和图片无关。
- `#20463` 关键实现：在 MoE/OCR loader 末尾加 `copy_conv3d_weight_to_linear()`，并用 `is_nextn` 避免 draft-only load 没有 `visual`。
- `#20740` 关键实现：按 maintainer 要求回退上述直接调用，当前主线不保留这段调用。
- 关键代码片段：

```python
if not is_nextn:
    self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

```python
# #20740 removed the direct MoE/OCR loader copy call.
```

- 验证信号：`#20463` body 写明用 GLM-4.6V-FP8 on B200 TP=4 复现/修复；`#20740` 定义当前主线状态。
- 风险：文档必须同时写两个 PR，不能只说“修复了”。真正长期方向是共享 Conv layer 与 loader-aware 测试。

### `#21134`：GLM-V/OCR Transformers 5.x 字段检测和 MTP omission

- Motivation：Transformers 5.x 后 GLM 字段可能在 `text_config`；MTP safetensors 读取漏掉该字段会让 acceptance length 异常；GLM-OCR connector 维度应来自 `text_config.intermediate_size`。
- 关键实现：`maybe_add_mtp_safetensors` 从 `hf_config.text_config` 读 `num_nextn_predict_layers`；GLM4V-MoE loader 先规范化 `language_model.` / `model.visual.`；OCR visual model 接收 `text_config` 并用 `intermediate_size`。
- 关键代码片段：

```python
num_nextn_layers = getattr(
    getattr(hf_config, "text_config", hf_config),
    "num_nextn_predict_layers",
    getattr(hf_config, "num_nextn_predict_layers", 0),
)
```

```python
if "language_model." in name:
    name = name.replace("language_model.", "")
if "model.visual." in name:
    name = name.replace("model.visual.", "visual.")
```

```python
context_dim=text_config.intermediate_size,
```

- 验证信号：修复 MTP acceptance 和 GLM-OCR 配置字段漂移。
- 风险：后续 GLM-OCR 应把 `text_config` 当成优先来源。

## Open PR 雷达

### `#9349`：GLM-4.5V FP8 fused-MoE tuning

- Motivation：为 GLM-4.5V FP8 添加 MoE kernel generation 支持。
- 关键实现：tuning 脚本识别 `Glm4vMoeForConditionalGeneration`，从 `config.text_config` 读取专家数、top-k、moe intermediate size，并新增 L40S FP8 配置。
- 关键代码片段：

```python
cfg_source = config.text_config if is_glm4v_moe else config
E = cfg_source.n_routed_experts
topk = cfg_source.num_experts_per_tok
```

- 状态/风险：open，暂无 benchmark；只能作为后续 FP8 tuning radar。

### `#14662`：GLM4.6V ktransformers

- Motivation：给 GLM4.6V 暴露 expert-location 元数据。
- 关键实现：`Glm4vMoeForConditionalGeneration.get_model_config_for_expert_location` 从 `text_config` 返回 layer 和 expert 数。
- 关键代码片段：

```python
return ModelConfigForExpertLocation(
    num_layers=config.text_config.num_hidden_layers,
    num_logical_experts=config.text_config.n_routed_experts,
    num_groups=None,
)
```

- 状态/风险：open；这是 expert placement 元数据，不是视觉正确性修复。

### `#19728`：ROCm GLM-4.5V-FP8 startup

- Motivation：MI300X 上 `SGLANG_USE_AITER=0` 启动 GLM-4.5V-FP8 时，FP8 MoE padding 和 HIP FP8 fallback padding 造成启动/graph capture 失败。
- 关键实现：如果 runtime hidden size 已等于 `w1.shape[2]`，禁用 padding adjustment；HIP fallback 拷贝到 padded buffer 时只写真实行并填充尾行。
- 关键代码片段：

```python
elif hidden_states.shape[1] == w1.shape[2]:
    padded_size = 0
```

```python
dst[: src.shape[0]].copy_(src)
if dst.shape[0] > src.shape[0]:
    dst[src.shape[0] :].fill_(pad_value)
```

- 验证信号：PR 报告 MI300X targeted tests 和端到端启动通过。
- 状态/风险：open；合入后需要同步 AMD GLM-4.5V 文档和 FP8 fallback 测试。

### `#22961`：NPU GLM-4.5V

- Motivation：NPU 上 `split_qkv_rmsnorm_rope` 已支持 `NORMS=False`，GLM-4.5V 需要根据 `use_qk_norm` 传参。
- 关键实现：有 QK norm 时传 norm 权重和 eps；没有时传 `None`，避免错误调用。
- 关键代码片段：

```python
if self.use_qk_norm:
    eps = self.q_norm.variance_epsilon
    q_weight = self.q_norm.weight
    k_weight = self.k_norm.weight
else:
    eps = None
    q_weight = None
    k_weight = None
```

- 验证信号：PR 报告 MMMU accuracy `0.2802`、invalid `0.000`、latency `89.380s`、output throughput `33.565 token/s`。
- 状态/风险：open；虽然改的是 `glm4_moe.py` 文本 attention，但目标是 GLM-4.5V 平台支持。

## sgl-cookbook / 文档证据

- SGLang `docs/basic_usage/glmv.mdx`：覆盖 FP8/BF16 launch、`--keep-mm-feature-on-device`、`--mm-attention-backend`、`--mm-max-concurrent-calls`、`--mm-enable-dp-encoder`、`SGLANG_USE_CUDA_IPC_TRANSPORT=1`、`SGLANG_VLM_CACHE_SIZE_MB=0`、GLM thinking budget/custom logit processor。
- GLM-4.5V cookbook：硬件覆盖 NVIDIA B200/H100/H200 和 AMD MI300X/MI325X/MI355X；TP=8 时建议 `--mm-enable-dp-encoder`。
- GLM-4.6V cookbook：强调 128K context、native multimodal function calling、document understanding、frontend replication、video input。
- GLM-OCR cookbook：给出 EAGLE/MTP launch、OCRBench、OmniDocBench 线索。
- LMSYS blog `GLM-4.5 Meets SGLang`：提供 GLM parser、tool-call parser、MTP/EAGLE、FP8 variants 和 MoE 架构背景。
- sgl-cookbook `#95`：GLM-4.5V AMD MI300X/MI325X/MI355X。
- sgl-cookbook `#131`：GLM-4.5V/4.6V MI325X。
- sgl-cookbook `#136`：GLM-OCR cookbook。

## 下一步优化建议

1. GLM-OCR 加 OCRBench/OmniDocBench 小样例和 MTP acceptance 检查，不要只做 caption smoke。
2. GLM VLM 的 loader 变更必须覆盖 dense、MoE、OCR、NextN 四个路径。
3. vision encoder DP/PP 改动必须保留 no-DP baseline 和 PP stage loader 检查。
4. AMD/NPU 平台 PR 必须保留启动、graph capture、模型接口请求和任务精度四类证据。
5. 文档中凡引用 PR，都按 motivation、实现思路、关键代码、验证、当前状态/风险写完整卡片。

<!-- MODEL_PR_DIFF_AUDIT:START zh -->

## 逐 PR diff 审计卡（2026-04-25 重做）

本节按 `sgl-project/sglang` 的 Pull Request API 和文件级 patch 重新审计 `GLM VLM / OCR`。验收口径：每个 PR 都要有状态、代码面、文件级 diff 摘要、支持/优化点判断和风险验证点；没有公开相关 PR 时必须写清检索结论，不能编造。

### 时间线总览

| 创建日期 | PR | 状态 | 标题 | 代码面 | 主要 diff 文件 |
| --- | ---: | --- | --- | --- | --- |
| 2025-08-05 | [#8798](https://github.com/sgl-project/sglang/pull/8798) | merged | Support glm4.1v and glm4.5v | model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-08-11 | [#9059](https://github.com/sgl-project/sglang/pull/9059) | merged | [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8 | model wrapper, attention/backend, MoE/router, multimodal/processor, tests/benchmarks | `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py` |
| 2025-08-15 | [#9245](https://github.com/sgl-project/sglang/pull/9245) | merged | Set the default attention backend for GLM-4.5v to fa3 | misc | `python/sglang/srt/utils.py` |
| 2025-08-19 | [#9349](https://github.com/sgl-project/sglang/pull/9349) | open | Add support for GLM 4.5V FP8 | MoE/router, quantization, kernel, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` |
| 2025-08-24 | [#9554](https://github.com/sgl-project/sglang/pull/9554) | merged | Fix GLM45v launch server cuda torch compile bug | model wrapper | `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-09-01 | [#9884](https://github.com/sgl-project/sglang/pull/9884) | merged | [Bug Fix] Fix Glm4vVisionBlock norm | model wrapper | `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/glm4v.py` |
| 2025-09-08 | [#10147](https://github.com/sgl-project/sglang/pull/10147) | merged | Fix: (glm4v) Add missing field | model wrapper | `python/sglang/srt/models/glm4v.py` |
| 2025-09-09 | [#10228](https://github.com/sgl-project/sglang/pull/10228) | merged | Add self.capture_aux_hidden_states For GLM-4.5V | model wrapper, MoE/router | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-10-02 | [#11166](https://github.com/sgl-project/sglang/pull/11166) | merged | Tiny move files to utils folder | model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config | `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/managers/scheduler.py` |
| 2025-10-09 | [#11388](https://github.com/sgl-project/sglang/pull/11388) | merged | Replace pad with cat for better performance | model wrapper | `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py` |
| 2025-10-21 | [#11922](https://github.com/sgl-project/sglang/pull/11922) | merged | [lint] improve ruff check | model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config | `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/utils/common.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` |
| 2025-10-25 | [#12117](https://github.com/sgl-project/sglang/pull/12117) | merged | GLM-4-0414 and GLM-4.1V Code Refactor | model wrapper, MoE/router | `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2025-11-13 | [#13228](https://github.com/sgl-project/sglang/pull/13228) | merged | Cleanup vision attention related codes | model wrapper, MoE/router | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` |
| 2025-11-28 | [#14097](https://github.com/sgl-project/sglang/pull/14097) | merged | support GLM-V vision model dp | model wrapper, MoE/router, tests/benchmarks | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py` |
| 2025-12-08 | [#14662](https://github.com/sgl-project/sglang/pull/14662) | open | [Glm46v] support ktransformers | model wrapper, MoE/router | `python/sglang/srt/models/glm4v_moe.py` |
| 2025-12-09 | [#14720](https://github.com/sgl-project/sglang/pull/14720) | merged | [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V | model wrapper, tests/benchmarks | `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/test/test_utils.py` |
| 2025-12-11 | [#14927](https://github.com/sgl-project/sglang/pull/14927) | merged | [CI]add nightly CI for glm4v_moe arch model | tests/benchmarks | `test/nightly/test_vlms_mmmu_eval.py` |
| 2025-12-12 | [#14998](https://github.com/sgl-project/sglang/pull/14998) | merged | add transformers version validation for glm-4.6v moe models | docs/config | `python/sglang/srt/configs/model_config.py` |
| 2025-12-15 | [#15205](https://github.com/sgl-project/sglang/pull/15205) | merged | [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V | model wrapper, attention/backend, multimodal/processor | `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2025-12-19 | [#15434](https://github.com/sgl-project/sglang/pull/15434) | merged | Convert cu_seqlens to CPU for npu_flash_attention_unpad operator | model wrapper, attention/backend, MoE/router, multimodal/processor | `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py` |
| 2026-01-15 | [#17122](https://github.com/sgl-project/sglang/pull/17122) | merged | [bugfix]GLM-4V model | model wrapper, multimodal/processor, tests/benchmarks | `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py` |
| 2026-01-20 | [#17420](https://github.com/sgl-project/sglang/pull/17420) | merged | [VLM] Optimize get_rope_index for GLM4v | tests/benchmarks | `benchmark/bench_rope/benchmark_rope_index.py`, `python/sglang/srt/layers/rotary_embedding.py` |
| 2026-01-22 | [#17582](https://github.com/sgl-project/sglang/pull/17582) | merged | [GLM-OCR] Support GLM-OCR Model | model wrapper, attention/backend, multimodal/processor, docs/config | `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/layers/attention/vision.py` |
| 2026-02-16 | [#18885](https://github.com/sgl-project/sglang/pull/18885) | merged | Fix GLM-4V processor registration when glm_ocr is unavailable | multimodal/processor | `python/sglang/srt/multimodal/processors/glm4v.py` |
| 2026-03-03 | [#19728](https://github.com/sgl-project/sglang/pull/19728) | open | Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback | MoE/router, quantization, kernel, tests/benchmarks | `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py` |
| 2026-03-06 | [#20033](https://github.com/sgl-project/sglang/pull/20033) | merged | [VLM] Replace conv3d proj with linear for GLM4V | model wrapper, tests/benchmarks | `test/registered/vlm/test_patch_embed_perf.py`, `python/sglang/srt/models/glm4v.py` |
| 2026-03-10 | [#20282](https://github.com/sgl-project/sglang/pull/20282) | merged | Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug | model wrapper, tests/benchmarks | `test/unit/test_conv_layer.py`, `python/sglang/srt/layers/conv.py`, `python/sglang/srt/server_args.py` |
| 2026-03-12 | [#20463](https://github.com/sgl-project/sglang/pull/20463) | merged | [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr | model wrapper, MoE/router | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-17 | [#20740](https://github.com/sgl-project/sglang/pull/20740) | merged | Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr" | model wrapper, MoE/router | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-03-22 | [#21134](https://github.com/sgl-project/sglang/pull/21134) | merged | [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix | model wrapper, MoE/router | `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm_ocr.py` |
| 2026-04-16 | [#22961](https://github.com/sgl-project/sglang/pull/22961) | open | [NPU] Support GLM-4.5V | model wrapper, MoE/router | `python/sglang/srt/models/glm4_moe.py` |

### 逐 PR 代码 diff 阅读记录

### PR #8798 - Support glm4.1v and glm4.5v

- 链接：https://github.com/sgl-project/sglang/pull/8798
- 状态/时间：`merged`，created 2025-08-05, merged 2025-08-09；作者 `byjiang1996`。
- 代码 diff 已读范围：`21` 个文件，`+1584/-19`；代码面：model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config；关键词：vision, attention, config, processor, test, cache, cuda, kv, lora, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunk: +import logging; 符号: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__
  - `python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunk: +import logging; 符号: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunk: # Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py; def __init__(; 符号: __init__, forward, get_rope_index, get_rope_index_glm4v
  - `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunk: +import re; 符号: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async
  - `test/srt/test_jinja_template_utils.py` modified +80/-0 (80 lines); hunk: def test_detect_empty_template(self):; 符号: test_detect_empty_template, test_detect_msg_content_pattern, with, test_detect_m_content_pattern
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 vision, attention, config, processor, test, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9059 - [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8

- 链接：https://github.com/sgl-project/sglang/pull/9059
- 状态/时间：`merged`，created 2025-08-11, merged 2025-08-18；作者 `byjiang1996`。
- 代码 diff 已读范围：`9` 个文件，`+150/-102`；代码面：model wrapper, attention/backend, MoE/router, multimodal/processor, tests/benchmarks；关键词：attention, config, vision, kv, quant, benchmark, moe, triton, expert, processor。
- 代码 diff 细节：
  - `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunk: +"""Utility functions for vision attention layers."""; 符号: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunk: from sglang.srt.hf_transformers_utils import get_processor; def __init__(; 符号: __init__, __init__, get_video_feature, _update_hf_config
  - `python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunk: from transformers.activations import ACT2FN; def __init__(; 符号: __init__, __init__, _update_vision_config, pixel_shuffle
  - `python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunk: from torch import nn; def __init__(; 符号: __init__, __init__, _update_hf_config, pixel_shuffle
  - `benchmark/mmmu/bench_hf.py` modified +6/-2 (8 lines); hunk: def eval_mmmu(args):; 符号: eval_mmmu
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`；patch 关键词为 attention, config, vision, kv, quant, benchmark。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9245 - Set the default attention backend for GLM-4.5v to fa3

- 链接：https://github.com/sgl-project/sglang/pull/9245
- 状态/时间：`merged`，created 2025-08-15, merged 2025-08-17；作者 `zifeitong`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：misc；关键词：config, moe。
- 代码 diff 细节：
  - `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunk: def is_fa3_default_architecture(hf_config):; 符号: is_fa3_default_architecture
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/utils.py`；patch 关键词为 config, moe。影响判断：改动落在杂项路径，要从文件列表反推实际影响面。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9349 - Add support for GLM 4.5V FP8

- 链接：https://github.com/sgl-project/sglang/pull/9349
- 状态/时间：`open`，created 2025-08-19；作者 `pakjoeng`。
- 代码 diff 已读范围：`2` 个文件，`+153/-4`；代码面：MoE/router, quantization, kernel, tests/benchmarks, docs/config；关键词：config, moe, triton, benchmark, expert, fp8, topk。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunk: +{
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunk: def main(args: argparse.Namespace):; 符号: main
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`；patch 关键词为 config, moe, triton, benchmark, expert, fp8。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9554 - Fix GLM45v launch server cuda torch compile bug

- 链接：https://github.com/sgl-project/sglang/pull/9554
- 状态/时间：`merged`，created 2025-08-24, merged 2025-08-25；作者 `byjiang1996`。
- 代码 diff 已读范围：`1` 个文件，`+1/-0`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunk: def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:; 符号: get_video_feature, get_input_embeddings, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_5_vl.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_5_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #9884 - [Bug Fix] Fix Glm4vVisionBlock norm

- 链接：https://github.com/sgl-project/sglang/pull/9884
- 状态/时间：`merged`，created 2025-09-01, merged 2025-09-05；作者 `sdpkjc`。
- 代码 diff 已读范围：`2` 个文件，`+4/-4`；代码面：model wrapper；关键词：config, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-2 (5 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/glm4v.py`；patch 关键词为 config, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/glm4v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10147 - Fix: (glm4v) Add missing field

- 链接：https://github.com/sgl-project/sglang/pull/10147
- 状态/时间：`merged`，created 2025-09-08, merged 2025-09-08；作者 `JustinTong0323`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：model wrapper；关键词：config, eagle。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunk: def __init__(; 符号: __init__, get_image_feature
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`；patch 关键词为 config, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #10228 - Add self.capture_aux_hidden_states For GLM-4.5V

- 链接：https://github.com/sgl-project/sglang/pull/10228
- 状态/时间：`merged`，created 2025-09-09, merged 2025-09-14；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：model wrapper, MoE/router；关键词：config, eagle, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunk: def __init__(; 符号: __init__, determine_num_fused_shared_experts
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v_moe.py`；patch 关键词为 config, eagle, expert, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11166 - Tiny move files to utils folder

- 链接：https://github.com/sgl-project/sglang/pull/11166
- 状态/时间：`merged`，created 2025-10-02, merged 2025-10-03；作者 `fzyzcjy`。
- 代码 diff 已读范围：`66` 个文件，`+91/-79`；代码面：model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, attention, processor, test, benchmark, cache, cuda, expert, lora, moe。
- 代码 diff 细节：
  - `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunk: def setUp(self):; def setUp(self):; 符号: setUp, setUp, setUp, setUp
  - `python/sglang/srt/managers/tp_worker.py` modified +6/-6 (12 lines); hunk: from sglang.srt.configs.model_config import ModelConfig; PPProxyTensors,
  - `python/sglang/srt/managers/scheduler.py` modified +5/-5 (10 lines); hunk: ); set_random_seed,
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunk: from sglang.srt.aio_rwlock import RWLock; get_zmq_socket,
  - `python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunk: from transformers import PretrainedConfig
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/managers/scheduler.py`；patch 关键词为 config, attention, processor, test, benchmark, cache。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/managers/scheduler.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11388 - Replace pad with cat for better performance

- 链接：https://github.com/sgl-project/sglang/pull/11388
- 状态/时间：`merged`，created 2025-10-09, merged 2025-10-10；作者 `yuan-luo`。
- 代码 diff 已读范围：`5` 个文件，`+5/-5`；代码面：model wrapper；关键词：n/a。
- 代码 diff 细节：
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunk: def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:; 符号: forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-1 (2 lines); hunk: def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`；patch 关键词为 n/a。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #11922 - [lint] improve ruff check

- 链接：https://github.com/sgl-project/sglang/pull/11922
- 状态/时间：`merged`，created 2025-10-21, merged 2025-10-22；作者 `hnyls2002`。
- 代码 diff 已读范围：`19` 个文件，`+73/-31`；代码面：model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config；关键词：config, quant, attention, benchmark, cache, kv, triton, doc, expert, flash。
- 代码 diff 细节：
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunk: PrecisionConfig,; def triton_kernel_fused_experts(; 符号: triton_kernel_fused_experts, triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias, triton_kernel_fused_experts_with_bias
  - `python/sglang/srt/utils/common.py` modified +10/-2 (12 lines); hunk: import threading; from multiprocessing.reduction import ForkingPickler; 符号: monkey_patch_vllm_gguf_config, get_quant_method_with_embedding_replaced, direct_register_custom_op
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunk: from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunk: ); class PrefillMetadata:; 符号: PrefillMetadata:, FlashInferMhaChunkKVRunner:, __init__
  - `.pre-commit-config.yaml` modified +3/-1 (4 lines); hunk: repos:
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/utils/common.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`；patch 关键词为 config, quant, attention, benchmark, cache, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；scheduler/runtime/cache 路径发生变化，要核对连续批处理、spec/PD/DP、cache 生命周期和异常分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/utils/common.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #12117 - GLM-4-0414 and GLM-4.1V Code Refactor

- 链接：https://github.com/sgl-project/sglang/pull/12117
- 状态/时间：`merged`，created 2025-10-25, merged 2025-10-27；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`4` 个文件，`+679/-173`；代码面：model wrapper, MoE/router；关键词：config, quant, attention, cache, cuda, eagle, kv, processor, triton, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunk: # Modeling from:; def __init__(; 符号: Glm4MLP, __init__, forward, Glm4Attention
  - `python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunk: +# Copyright 2023-2024 SGLang Team; from sglang.srt.layers.pooler import Pooler, PoolingType; 符号: __init__, forward, Glm4vVisionBlock, Glm4vVisionBlock
  - `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunk: def _triton_mrope_forward(; def _triton_mrope_forward(; 符号: _triton_mrope_forward, _triton_mrope_forward, triton_mrope, triton_mrope
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunk: def __init__(; 符号: __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 config, quant, attention, cache, cuda, eagle。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #13228 - Cleanup vision attention related codes

- 链接：https://github.com/sgl-project/sglang/pull/13228
- 状态/时间：`merged`，created 2025-11-13, merged 2025-11-16；作者 `JustinTong0323`。
- 代码 diff 已读范围：`15` 个文件，`+4/-142`；代码面：model wrapper, MoE/router；关键词：config, kv, quant, attention, flash, vision, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
  - `python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-13 (13 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`；patch 关键词为 config, kv, quant, attention, flash, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14097 - support GLM-V vision model dp

- 链接：https://github.com/sgl-project/sglang/pull/14097
- 状态/时间：`merged`，created 2025-11-28, merged 2025-12-05；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`4` 个文件，`+91/-52`；代码面：model wrapper, MoE/router, tests/benchmarks；关键词：attention, config, kv, moe, processor, quant, test, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunk: from einops import rearrange; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; 符号: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunk: def __init__(; 符号: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunk: def __init__(; 符号: __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunk: SimpleNamespace(model="Qwen/Qwen2.5-VL-72B-Instruct", mmmu_accuracy=0.55),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 attention, config, kv, moe, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14662 - [Glm46v] support ktransformers

- 链接：https://github.com/sgl-project/sglang/pull/14662
- 状态/时间：`open`，created 2025-12-08；作者 `mrhaoxx`。
- 代码 diff 已读范围：`1` 个文件，`+8/-0`；代码面：model wrapper, MoE/router；关键词：config, expert, moe, quant, triton。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunk: from sglang.srt.layers.moe import get_moe_a2a_backend; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights, get_model_config_for_expert_location
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v_moe.py`；patch 关键词为 config, expert, moe, quant, triton。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14720 - [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V

- 链接：https://github.com/sgl-project/sglang/pull/14720
- 状态/时间：`merged`，created 2025-12-09, merged 2025-12-10；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+66/-2`；代码面：model wrapper, tests/benchmarks；关键词：test, mla, cuda, fp4, moe, spec, vision。
- 代码 diff 细节：
  - `test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunk: from sglang.test.test_utils import (; def test_chunked_prefill_with_small_bs(self):; 符号: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass
  - `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunk: general_mm_embed_routine,; def forward(; 符号: forward, forward, load_weights, load_weights
  - `python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunk: DEFAULT_MODEL_NAME_FOR_TEST_MLA = "lmsys/sglang-ci-dsv3-test"
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunk: TestFile("test_gpt_oss_4gpu.py", 300),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/test/test_utils.py`；patch 关键词为 test, mla, cuda, fp4, moe, spec。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/test/test_utils.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14927 - [CI]add nightly CI for glm4v_moe arch model

- 链接：https://github.com/sgl-project/sglang/pull/14927
- 状态/时间：`merged`，created 2025-12-11, merged 2025-12-12；作者 `zminglei`。
- 代码 diff 已读范围：`1` 个文件，`+3/-0`；代码面：tests/benchmarks；关键词：fp8, test。
- 代码 diff 细节：
  - `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunk: ): ModelEvalMetrics(0.310, 16.7),
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/nightly/test_vlms_mmmu_eval.py`；patch 关键词为 fp8, test。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/nightly/test_vlms_mmmu_eval.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #14998 - add transformers version validation for glm-4.6v moe models

- 链接：https://github.com/sgl-project/sglang/pull/14998
- 状态/时间：`merged`，created 2025-12-12, merged 2025-12-13；作者 `yhyang201`。
- 代码 diff 已读范围：`1` 个文件，`+37/-0`；代码面：docs/config；关键词：attention, config, moe, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunk: def __init__(; def _verify_dual_chunk_attention_config(self) -> None:; 符号: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/configs/model_config.py`；patch 关键词为 attention, config, moe, quant, vision。影响判断：文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/configs/model_config.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- 链接：https://github.com/sgl-project/sglang/pull/15205
- 状态/时间：`merged`，created 2025-12-15, merged 2025-12-18；作者 `yuan-luo`。
- 代码 diff 已读范围：`4` 个文件，`+100/-80`；代码面：model wrapper, attention/backend, multimodal/processor；关键词：cache, vision, config, processor, quant, attention。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunk: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; 符号: forward, forward, forward, Glm4vVisionRotaryEmbedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunk: import torch.nn as nn; from sglang.srt.layers.logits_processor import LogitsProcessor; 符号: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunk: def forward(; def forward(; 符号: forward, forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunk: def get_cos_sin_with_position(self, positions):; 符号: get_cos_sin_with_position, get_cos_sin, forward_native
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`；patch 关键词为 cache, vision, config, processor, quant, attention。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #15434 - Convert cu_seqlens to CPU for npu_flash_attention_unpad operator

- 链接：https://github.com/sgl-project/sglang/pull/15434
- 状态/时间：`merged`，created 2025-12-19, merged 2026-01-04；作者 `xiaobaicxy`。
- 代码 diff 已读范围：`9` 个文件，`+36/-13`；代码面：model wrapper, attention/backend, MoE/router, multimodal/processor；关键词：attention, flash, vision, processor, config, cuda, quant, expert, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunk: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; def forward(; 符号: forward
  - `python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunk: from sglang.srt.models.glm4 import Glm4Model; def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:; 符号: forward
  - `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunk: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; 符号: Projector, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunk: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; def forward(; 符号: forward
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1 (5 lines); hunk: from sglang.srt.distributed import parallel_state; def forward(; 符号: forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`；patch 关键词为 attention, flash, vision, processor, config, cuda。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17122 - [bugfix]GLM-4V model

- 链接：https://github.com/sgl-project/sglang/pull/17122
- 状态/时间：`merged`，created 2026-01-15, merged 2026-04-01；作者 `KnightLTC`。
- 代码 diff 已读范围：`3` 个文件，`+38/-3`；代码面：model wrapper, multimodal/processor, tests/benchmarks；关键词：attention, cuda, benchmark, cache, config, kv, processor, quant, test, vision。
- 代码 diff 细节：
  - `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunk: +import unittest; 符号: TestGLM4Models, test_vlm_mmmu_benchmark
  - `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunk: def process_mm_data(; 符号: process_mm_data
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`；patch 关键词为 attention, cuda, benchmark, cache, config, kv。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17420 - [VLM] Optimize get_rope_index for GLM4v

- 链接：https://github.com/sgl-project/sglang/pull/17420
- 状态/时间：`merged`，created 2026-01-20, merged 2026-02-01；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+526/-86`；代码面：tests/benchmarks；关键词：attention, config, vision, benchmark, cuda, moe, test。
- 代码 diff 细节：
  - `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunk: +# This script benchmarks MRotaryEmbedding.get_rope_index_glm4v (GLM4V mrope index builder).; 符号: DummyVisionConfig:, DummyHFConfig:, calculate_stats, _sync
  - `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunk: def get_rope_index(; def get_rope_index(; 符号: get_rope_index, get_rope_index, get_rope_index, get_rope_index_glm4v
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `benchmark/bench_rope/benchmark_rope_index.py`, `python/sglang/srt/layers/rotary_embedding.py`；patch 关键词为 attention, config, vision, benchmark, cuda, moe。影响判断：测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `benchmark/bench_rope/benchmark_rope_index.py`, `python/sglang/srt/layers/rotary_embedding.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #17582 - [GLM-OCR] Support GLM-OCR Model

- 链接：https://github.com/sgl-project/sglang/pull/17582
- 状态/时间：`merged`，created 2026-01-22, merged 2026-01-27；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`9` 个文件，`+679/-29`；代码面：model wrapper, attention/backend, multimodal/processor, docs/config；关键词：config, attention, spec, kv, processor, quant, vision, moe, cache, cuda。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__
  - `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunk: +# Copyright 2023-2024 SGLang Team; 符号: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN
  - `python/sglang/srt/layers/attention/vision.py` modified +49/-19 (68 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, _init_qk_norm
  - `python/sglang/srt/models/glm4.py` modified +18/-6 (24 lines); hunk: def __init__(; def __init__(; 符号: __init__, __init__, __init__, __init__
  - `python/sglang/srt/configs/model_config.py` modified +7/-1 (8 lines); hunk: def _config_draft_model(self):; def _verify_transformers_version(self):; 符号: _config_draft_model, _verify_transformers_version, is_generation_model
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/layers/attention/vision.py`；patch 关键词为 config, attention, spec, kv, processor, quant。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；attention、KV cache 或 backend 选择发生变化，要重点核对 prefill/decode、page size、RoPE/MLA/MQA 分支；多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接；文档或配置面发生变化，要核对 serve flags、默认值和 cookbook 命令是否与代码一致。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/layers/attention/vision.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #18885 - Fix GLM-4V processor registration when glm_ocr is unavailable

- 链接：https://github.com/sgl-project/sglang/pull/18885
- 状态/时间：`merged`，created 2026-02-16, merged 2026-02-16；作者 `alisonshao`。
- 代码 diff 已读范围：`1` 个文件，`+12/-4`；代码面：multimodal/processor；关键词：config, moe, processor, spec。
- 代码 diff 细节：
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunk: from sglang.srt.layers.rotary_embedding import MRotaryEmbedding; 符号: Glm4vImageProcessor, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/multimodal/processors/glm4v.py`；patch 关键词为 config, moe, processor, spec。影响判断：多模态 processor 或 media token 路径发生变化，要核对 image/video/audio metadata、position ids 和 batch 拼接。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/multimodal/processors/glm4v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- 链接：https://github.com/sgl-project/sglang/pull/19728
- 状态/时间：`open`，created 2026-03-03；作者 `andyluo7`。
- 代码 diff 已读范围：`4` 个文件，`+104/-4`；代码面：MoE/router, quantization, kernel, tests/benchmarks；关键词：fp8, quant, expert, moe, test, triton, cache, config, cuda, mla。
- 代码 diff 细节：
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunk: import unittest; def test_various_configurations(self):; 符号: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunk: def per_token_group_quant_mla_deep_gemm_masked_fp8(; def _native_dynamic_per_token_quant_fp8(output, input, scale):; 符号: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_token_quant_fp8
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunk: import pytest; def test_scaled_fp8_quant_with_padding(dtype) -> None:; 符号: test_scaled_fp8_quant_with_padding
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunk: def fused_experts_impl(; 符号: fused_experts_impl
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`；patch 关键词为 fp8, quant, expert, moe, test, triton。影响判断：MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支；量化加载或量化 kernel 发生变化，要核对 scale、zero-point、checkpoint 命名和 fallback 行为；CUDA/Triton/C++ kernel 或 binding 发生变化，要核对 shape guard、dtype、设备后端和 benchmark；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20033 - [VLM] Replace conv3d proj with linear for GLM4V

- 链接：https://github.com/sgl-project/sglang/pull/20033
- 状态/时间：`merged`，created 2026-03-06, merged 2026-03-08；作者 `yuan-luo`。
- 代码 diff 已读范围：`2` 个文件，`+192/-9`；代码面：model wrapper, tests/benchmarks；关键词：benchmark, config, cuda, test, vision。
- 代码 diff 细节：
  - `test/registered/vlm/test_patch_embed_perf.py` added +166/-0 (166 lines); hunk: +import os; 符号: ReferenceConv3dPatchEmbed, __init__, forward, _build_modules
  - `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunk: def __init__(; def __init__(; 符号: __init__, forward, copy_conv3d_weight_to_linear, forward
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/registered/vlm/test_patch_embed_perf.py`, `python/sglang/srt/models/glm4v.py`；patch 关键词为 benchmark, config, cuda, test, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/registered/vlm/test_patch_embed_perf.py`, `python/sglang/srt/models/glm4v.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- 链接：https://github.com/sgl-project/sglang/pull/20282
- 状态/时间：`merged`，created 2026-03-10, merged 2026-03-15；作者 `yhyang201`。
- 代码 diff 已读范围：`18` 个文件，`+704/-90`；代码面：model wrapper, tests/benchmarks；关键词：config, attention, vision, quant, cuda, lora, moe, spec, test。
- 代码 diff 细节：
  - `test/unit/test_conv_layer.py` added +363/-0 (363 lines); hunk: +import unittest; 符号: _copy_weights, TestConv2dLayer, test_basic_patch_embedding, test_enable_linear
  - `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunk: +"""; 符号: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad
  - `python/sglang/srt/server_args.py` modified +0/-48 (48 lines); hunk: def check_server_args(self):; def check_server_args(self):; 符号: check_server_args, check_server_args, check_torch_2_9_1_cudnn_compatibility, check_lora_server_args
  - `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunk: from sglang.srt.layers.activation import SiluAndMul; def __init__(; 符号: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger
  - `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunk: from sglang.srt.layers.activation import SiluAndMul; class VisionTransformer(nn.Module):; 符号: VisionTransformer, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `test/unit/test_conv_layer.py`, `python/sglang/srt/layers/conv.py`, `python/sglang/srt/server_args.py`；patch 关键词为 config, attention, vision, quant, cuda, lora。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；测试或 benchmark 被更新，要把这些用例作为回归入口而不是只看模型能否加载。
- 风险与验证：回归时优先跑能覆盖 `test/unit/test_conv_layer.py`, `python/sglang/srt/layers/conv.py`, `python/sglang/srt/server_args.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20463 - [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr

- 链接：https://github.com/sgl-project/sglang/pull/20463
- 状态/时间：`merged`，created 2026-03-12, merged 2026-03-14；作者 `JustinTong0323`。
- 代码 diff 已读范围：`2` 个文件，`+6/-0`；代码面：model wrapper, MoE/router；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #20740 - Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"

- 链接：https://github.com/sgl-project/sglang/pull/20740
- 状态/时间：`merged`，created 2026-03-17, merged 2026-03-18；作者 `mickqian`。
- 代码 diff 已读范围：`2` 个文件，`+0/-6`；代码面：model wrapper, MoE/router；关键词：moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; 符号: load_weights
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`；patch 关键词为 moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #21134 - [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix

- 链接：https://github.com/sgl-project/sglang/pull/21134
- 状态/时间：`merged`，created 2026-03-22, merged 2026-03-23；作者 `zRzRzRzRzRzRzR`。
- 代码 diff 已读范围：`3` 个文件，`+16/-9`；代码面：model wrapper, MoE/router；关键词：config, moe, expert, quant, vision。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunk: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; 符号: load_weights, load_weights
  - `python/sglang/srt/model_loader/weight_utils.py` modified +5/-1 (6 lines); hunk: def maybe_add_mtp_safetensors(; 符号: maybe_add_mtp_safetensors
  - `python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunk: from einops import rearrange; class GlmOcrVisionModel(Glm4vVisionModel):; 符号: GlmOcrVisionModel, __init__, __init__, __init__
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm_ocr.py`；patch 关键词为 config, moe, expert, quant, vision。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm_ocr.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。

### PR #22961 - [NPU] Support GLM-4.5V

- 链接：https://github.com/sgl-project/sglang/pull/22961
- 状态/时间：`open`，created 2026-04-16；作者 `zhsurpass`。
- 代码 diff 已读范围：`1` 个文件，`+17/-5`；代码面：model wrapper, MoE/router；关键词：kv, moe。
- 代码 diff 细节：
  - `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunk: def forward_prepare(; 符号: forward_prepare
- 支持/优化点判断：该 PR 的实际 diff 主要落在 `python/sglang/srt/models/glm4_moe.py`；patch 关键词为 kv, moe。影响判断：模型 wrapper/forward/weight-load 路径发生变化，要核对 architecture mapping、hidden-state 形状和权重名映射；MoE/router/top-k/expert 分支发生变化，要核对 shared/routed expert、EP/TP/DP 组合和空 token 分支。
- 风险与验证：回归时优先跑能覆盖 `python/sglang/srt/models/glm4_moe.py` 的模型加载/推理路径，再叠加上面的代码面专项检查；如果改动包含测试、benchmark 或 serve flag，需要把它们纳入验证。


### 补漏和优化点排查

- 已覆盖 PR 数：31；open PR 数：4。
- 仍需跟进的 open PR：[#9349](https://github.com/sgl-project/sglang/pull/9349), [#14662](https://github.com/sgl-project/sglang/pull/14662), [#19728](https://github.com/sgl-project/sglang/pull/19728), [#22961](https://github.com/sgl-project/sglang/pull/22961)
- 后续新增 PR 必须补齐时间线和逐 PR diff 卡片，不能只写一句标题。

<!-- MODEL_PR_DIFF_AUDIT:END zh -->
