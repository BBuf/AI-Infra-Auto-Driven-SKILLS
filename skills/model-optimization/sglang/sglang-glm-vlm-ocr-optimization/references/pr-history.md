# GLM VLM/OCR PR History

Evidence sweep:

- SGLang `origin/main`: `b3e6cf60a` (`2026-04-22`)
- sgl-cookbook `origin/main`: `816bad5` (`2026-04-21`)
- Source diff method: each PR below was inspected with PR metadata, changed files, and patch/diff excerpts. The cards intentionally record motivation, implementation idea, and the small code fragments that explain the behavior.
- Searched paths: `glm4v.py`, `glm4v_moe.py`, `glm_ocr.py`, `glm_ocr_nextn.py`, GLM multimodal processors, GLM VLM/OCR docs/snippets.
- Searched PR terms: `GLM-4V`, `GLM-4.1V`, `GLM-4.5V`, `GLM-4.6V`, `GLM-OCR`, `GLM-Glyph`, `glm4v`, `glm_ocr`.

## Runtime Surfaces

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
- `docs_new/src/snippets/autoregressive/glm-45v-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-46v-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-glyph-deployment.jsx`
- `docs_new/src/snippets/autoregressive/glm-ocr-deployment.jsx`

## Merged/Current-Main PR Cards

### [#8798](https://github.com/sgl-project/sglang/pull/8798) - Support GLM-4.1V and GLM-4.5V

- Status: merged `2025-08-09`, merge commit `f29aba8`, about `+1584/-19`, 21 files.
- Motivation: close GLM-4V support gaps for GLM-4.1V Thinking and GLM-4.5V. The diff references the prior HF implementation path and adds SGLang-side model, processor, template, MRoPE, and tests instead of relying on generic Qwen2.5-VL fallback.
- Key implementation: registers GLM4V generation architecture, adds GLM4V and GLM4V-MoE model files, creates `glm4v` chat template, wires `Glm4vImageProcessor`, supports multiple vision start token ids, and routes GLM4V through the multimodal RoPE index path.
- Key code fragment:

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
vision_start_indices = torch.nonzero(torch.isin(input_ids, ids)).squeeze(1)
```

- Validation signal: PR body reports GLM-4.1V/4.5V single-image, multi-image, and video tests, plus MMMU validation around GLM4.1V `0.701`.
- Risk to preserve: GLM4V shares logic with Qwen VLM but has its own token wrappers and MRoPE behavior; never collapse it into a plain Qwen2.5-VL assumption without checking token placement.

### [#9059](https://github.com/sgl-project/sglang/pull/9059) - GLM4.1V/4.5V dummy-head TP support

- Status: merged `2025-08-18`, merge commit `c2fbf60`, about `+150/-102`, 9 files.
- Motivation: GLM text attention can shard over `num_key_value_heads=8`, but vision attention has 12 heads. Without dummy heads, TP is capped by divisibility and TP=8 fails.
- Key implementation: compute `num_dummy_heads` from TP size, store it on `vision_config`, and pad q/k/v plus projection/norm weights during GLM4V and GLM4V-MoE weight loading.
- Key code fragment:

```python
num_dummy_heads = ((num_heads + tp_size) // tp_size) * tp_size - num_heads
setattr(self.config.vision_config, "head_dim", head_dim)
setattr(self.config.vision_config, "num_dummy_heads", num_dummy_heads)
```

```python
if "attn.qkv_proj" in name:
    wq, wk, wv = loaded_weight.chunk(3, dim=0)
    loaded_weight = torch.cat([wq, wk, wv], dim=0)
```

- Validation signal: raises practical GLM4V TP ceiling from 4 to 8 when the vision encoder is correctly padded.
- Risk to preserve: any future refactor of vision attention or loader must keep dummy-head padding and sharding metadata in sync.

### [#9245](https://github.com/sgl-project/sglang/pull/9245) - Default GLM-4.5V to FA3

- Status: merged `2025-08-17`, merge commit `84b30d9`, `+1`, 1 file.
- Motivation: the GLM-4.5V model page recommended SGLang with FA3, and leaving backend selection generic made users miss the expected default performance lane.
- Key implementation: add GLM4V-MoE architecture to the default-FA3 architecture list.
- Key code fragment:

```python
"Glm4vMoeForConditionalGeneration",
```

- Validation signal: no model math changed; the value is in launch defaults.
- Risk to preserve: if a platform cannot run FA3, docs and launch generators must tell users which explicit `--mm-attention-backend` to choose.

### [#9554](https://github.com/sgl-project/sglang/pull/9554) - Fix GLM45V torch.compile launch bug

- Status: merged `2025-08-25`, merge commit `24a8cee`, `+1`, 1 file.
- Motivation: GLM45V inherited the Qwen2.5-VL forward path, and cuda graph plus `torch.compile` failed because a fake tensor with grad interacted with an `out=` operation.
- Key implementation: wrap VLM forward in no-grad mode so the compile/cuda-graph launch path sees inference-only tensors.
- Key code fragment:

```python
@torch.no_grad()
def forward(...):
```

- Validation signal: PR body reports GLM45V cuda graph and torch compile launch success with unchanged MMMU.
- Risk to preserve: compile launch issues can appear far from GLM files because GLM4V reuses common VLM primitives.

### [#9884](https://github.com/sgl-project/sglang/pull/9884) - Fix Glm4vVisionBlock norm

- Status: merged `2025-09-05`, merge commit `0f6ac5`, `+4/-4`, 2 files.
- Motivation: after a shared vision-block refactor, `Qwen2_5_VisionBlock` called `norm2(..., residual=attn2d)`. GLM4V overrode the norms with a `Glm4vRMSNorm.forward(x)` signature that did not accept `residual`, causing runtime failures.
- Key implementation: stop overriding GLM4V `norm1`/`norm2` with the incompatible wrapper and pass the GLM RMSNorm epsilon into the parent path.
- Key code fragment:

```python
rms_norm_eps=config.rms_norm_eps,
```

- Validation signal: this is a compatibility fix between common Qwen-VL block code and GLM-specific norm semantics.
- Risk to preserve: when inheriting Qwen VLM blocks, check call signatures, not just tensor shapes.

### [#10147](https://github.com/sgl-project/sglang/pull/10147) - Add missing GLM4V EAGLE3 field

- Status: merged `2025-09-08`, merge commit `811680`, `+3`, 1 file.
- Motivation: CI/EAGLE3 paths expected `capture_aux_hidden_states` to exist on the model object.
- Key implementation: initialize the flag explicitly for GLM4V.
- Key code fragment:

```python
self.capture_aux_hidden_states = False
```

- Validation signal: no forward math change; the PR prevents speculative-decoding infrastructure from assuming a missing attribute.
- Risk to preserve: all GLM variants that share speculative plumbing need the same attribute contract.

### [#10228](https://github.com/sgl-project/sglang/pull/10228) - Add `capture_aux_hidden_states` for GLM-4.5V

- Status: merged `2025-09-14`, merge commit `b8347b`, `+3`, 1 file.
- Motivation: the MoE visual variant hit the same EAGLE3 attribute contract as GLM4V.
- Key implementation: initialize `capture_aux_hidden_states` in `glm4v_moe.py`.
- Key code fragment:

```python
self.capture_aux_hidden_states = False
```

- Validation signal: keeps GLM4V-MoE compatible with speculative decoding and auxiliary-hidden-state logic.
- Risk to preserve: adding a field to the dense visual model is not enough; mirror it into MoE and OCR variants when they subclass or fork the loader.

### [#11166](https://github.com/sgl-project/sglang/pull/11166) - Move utility files into `utils/`

- Status: merged `2025-10-03`, merge commit `fdc4e1`, about `+91/-79`, 66 files.
- Motivation: structural cleanup, moving `srt/utils.py` into a package while preserving import compatibility.
- Key implementation: create `sglang/srt/utils/__init__.py` that re-exports common utilities and update GLM VLM imports to use the new package layout.
- Key code fragment:

```python
from .common import *
```

```python
from sglang.srt.utils.hf_transformers_utils import get_processor
```

- Validation signal: not a GLM algorithm change, but it touched GLM4V and GLM4V-MoE import surfaces.
- Risk to preserve: import-only PRs can still break processor discovery if lazy imports hide optional dependencies.

### [#11388](https://github.com/sgl-project/sglang/pull/11388) - Replace `pad` with `cat`

- Status: merged `2025-10-10`, merge commit `b5044f`, `+5/-5`, 5 VLM files.
- Motivation: building `cu_seqlens` with `F.pad` was slower than concatenating a leading zero and occurred in hot multimodal paths.
- Key implementation: replace padding with a small `torch.cat` on GLM4V, Qwen VLM, and other vision models.
- Key code fragment:

```python
cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
```

- Validation signal: micro-optimization with no semantic change.
- Risk to preserve: this is small but hot; avoid reintroducing `F.pad` in repeated vision sequence construction.

### [#11922](https://github.com/sgl-project/sglang/pull/11922) - Improve ruff checks

- Status: merged `2025-10-22`, merge commit `9d6120`, about `+73/-31`, 19 files.
- Motivation: the pre-commit arguments were malformed and did not reliably catch unused/missing imports.
- Key implementation: set ruff `F401,F821` as separate args and auto-fix; GLM4V received a missing import cleanup.
- Key code fragment:

```yaml
args:
  - --select=F401,F821
  - --fix
```

- Validation signal: import hygiene matters for GLM processor registration because optional dependencies can already make imports fragile.
- Risk to preserve: do not dismiss lint-only PRs when auditing model support; missing imports directly affect runtime discovery.

### [#12117](https://github.com/sgl-project/sglang/pull/12117) - GLM-4-0414 and GLM-4.1V refactor

- Status: merged `2025-10-27`, merge commit `a88b006`, about `+679/-173`, 4 core files.
- Motivation: migrate GLM-4 text and GLM-4.1V code to newer SGLang interfaces and discard older custom code paths.
- Key implementation: rewrite `Glm4vVisionBlock` as a standalone module, use `VisionAttention`, route image embedding through `general_mm_embed_routine`, use `MultiModalityDataPaddingPatternMultimodalTokens`, and align PP missing-layer behavior.
- Key code fragment:

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

- Validation signal: the refactor is a baseline for later DP/PP and processor work.
- Risk to preserve: the new generic multimodal embedding routine is now part of GLM correctness; bypassing it can break PP proxy tensors or image-token padding.

### [#13228](https://github.com/sgl-project/sglang/pull/13228) - Cleanup vision attention code

- Status: merged `2025-11-16`, merge commit `b1c688f`, about `+4/-142`, 15 files.
- Motivation: vision attention had dead or hard-coded backend parameters and needed cleaner backend selection, including automatic `triton_attn` behavior on B200.
- Key implementation: remove per-model hard-coding and let `VisionAttention` resolve the backend centrally.
- Key code fragment:

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    projection_size=dim,
    use_qkv_parallel=True,
    proj_bias=True,
    flatten_batch=True,
)
```

- Validation signal: code deletion reduces divergent GLM/Qwen vision paths.
- Risk to preserve: backend-specific tuning should live in `VisionAttention` or launch args, not in one GLM block.

### [#14097](https://github.com/sgl-project/sglang/pull/14097) - Support GLM-V vision model DP

- Status: merged `2025-12-05`, merge commit `8fce9e7`, about `+91/-52`, 4 files.
- Motivation: GLM-V needed vision-encoder data parallelism so large TP launches could avoid vision-head divisibility and reduce TTFT.
- Key implementation: read `mm_enable_dp_encoder`, make the vision MLP/merger use TP size/rank `1/0` under DP, and run image/video feature extraction through the sharded MRoPE helper with `rope_type="rope_3d"`.
- Key code fragment:

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

- Validation signal: GLM V docs now recommend `--mm-enable-dp-encoder` for TP=8 when 12 vision heads cannot divide cleanly.
- Risk to preserve: always compare DP and no-DP output because DP changes feature distribution before LLM embedding.

### [#14720](https://github.com/sgl-project/sglang/pull/14720) - GLM-4.6V and GLM-4.1V PP

- Status: merged `2025-12-10`, merge commit `03836d`, about `+66/-2`, GLM4V plus PP test files.
- Motivation: GLM PP failed because non-last pipeline ranks do not own `lm_head.weight`, while loader and multimodal embedding still assumed full-rank ownership.
- Key implementation: pass `PPProxyTensors` into `general_mm_embed_routine`, skip `lm_head.*` on non-last PP ranks, and ignore parameters absent from the current PP stage.
- Key code fragment:

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

- Validation signal: adds PP accuracy coverage for GLM4.1V with MMMU threshold.
- Risk to preserve: loader changes for GLM visual models must be stage-aware.

### [#14927](https://github.com/sgl-project/sglang/pull/14927) - Nightly CI for `glm4v_moe`

- Status: merged `2025-12-12`, merge commit `b62e7e`, `+3`, 1 test file.
- Motivation: CI covered GLM4V dense (`GLM-4.1V-9B-Thinking`) but did not cover `glm4v_moe`, so MoE regressions could land silently.
- Key implementation: add `zai-org/GLM-4.5V-FP8` to nightly VLM MMMU evaluation with TP=2.
- Key code fragment:

```python
ModelLaunchSettings(
    "zai-org/GLM-4.5V-FP8", extra_args=["--tp=2"]
): ModelEvalMetrics(0.26, 32.0)
```

- Validation signal: regression guard for GLM4V-MoE rather than another dense VLM test.
- Risk to preserve: keep MoE VLM in CI after loader, quantization, or processor edits.

### [#14998](https://github.com/sgl-project/sglang/pull/14998) - Transformers version validation for GLM-4.6V MoE

- Status: merged `2025-12-13`, merge commit `a81cc1`, `+37`, 1 file.
- Motivation: GLM-4.6V MoE requires Transformers 5.x, but unnecessarily forcing TF5 for other models can create RoPE/config incompatibilities.
- Key implementation: detect GLM-4.6V by path or by `vision_config.model_type == "glm4v_moe_vision"`, raise on old Transformers for the models that need it, and warn on newer Transformers for models that do not.
- Key code fragment:

```python
is_glm_46vmoe = "glm-4.6v" in self.model_path.lower() or (
    vision_config is not None
    and getattr(vision_config, "model_type", None) == "glm4v_moe_vision"
)
```

```python
if tf_version < required_version:
    if needs_tf_v5:
        raise ValueError(...)
elif not needs_tf_v5:
    logger.warning(...)
```

- Validation signal: makes incompatibility fail early and descriptively.
- Risk to preserve: version checks must be scoped to the exact architecture/config requiring TF5.

### [#15205](https://github.com/sgl-project/sglang/pull/15205) - Cos/sin cache for Qwen3-VL and GLM-4.1V

- Status: merged `2025-12-18`, merge commit `8fa3dc3`, about `+100/-80`, 4 files.
- Motivation: GLM-4.1V and Qwen3-VL recomputed 2D vision RoPE frequencies and `cos()/sin()` in a hot path. The PR body reports a local latency cut from about `490us` to `186us` and end-to-end TTFT improvement.
- Key implementation: expose `RotaryEmbedding.get_cos_sin`, switch GLM4V visual RoPE to `get_rope`, return cached cos/sin plus position ids from `rot_pos_emb`, and let `VisionAttention` apply either `position_embeddings` or explicit cached cos/sin tensors.
- Key code fragment:

```python
def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_sin = self.cos_sin_cache[:seqlen]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return cos, sin
```

```python
self.rotary_pos_emb = get_rope(
    head_size=head_dim,
    rotary_dim=head_dim // 2,
    max_position=8192,
    base=10000.0,
    is_neox_style=True,
)
```

```python
rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(grid_thw)
```

- Validation signal: PR body reports Qwen3-VL MMMU no drop and GLM/Qwen video-cache examples with faster repeated requests.
- Risk to preserve: cached cos/sin must match the pre-existing half-dim duplication and GLM image grid indexing exactly.

### [#15434](https://github.com/sgl-project/sglang/pull/15434) - Convert `cu_seqlens` to CPU once for NPU flash attention

- Status: merged `2026-01-04`, merge commit `25fa2ac`, about `+36/-13`, 9 files.
- Motivation: Ascend/NPU vision attention converted `cu_seqlens` inside each layer, causing repeated device transfers and dispatch bubbles.
- Key implementation: make `VisionAttention` resolve sequence lengths on CPU and move GLM4V vision `cu_seqlens` to CPU once when running on NPU.
- Key code fragment:

```python
cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
```

```python
if is_npu():
    cu_seqlens = cu_seqlens.to("cpu")
```

- Validation signal: platform-path optimization with unchanged CUDA behavior.
- Risk to preserve: keep NPU-specific transfers outside repeated per-layer loops.

### [#17122](https://github.com/sgl-project/sglang/pull/17122) - GLM-4V bugfix

- Status: merged `2026-04-01`, merge commit `2488233`, about `+38/-3`, 3 files.
- Motivation: `VisionAttention` needs `num_dummy_heads` for shape calculation; if missing or stale, the TP divisibility check can fail. The PR also fixed processor handling on NPU.
- Key implementation: call `vision_utils.update_vit_attn_dummy_heads_config(self.config)` before building GLM4V visual modules, pass `num_dummy_heads` into blocks, and exclude `Glm4vProcessor` from the generic NPU processor patch branch.
- Key code fragment:

```python
vision_utils.update_vit_attn_dummy_heads_config(self.config)
```

```python
num_dummy_heads=vision_config.num_dummy_heads,
```

```python
elif processor.__class__.__name__ not in {"Glm4vProcessor"}:
```

- Validation signal: adds GLM-4.5V Ascend/NPU test coverage.
- Risk to preserve: dummy-head config must be recomputed before module construction, not only during weight loading.

### [#17420](https://github.com/sgl-project/sglang/pull/17420) - Optimize `get_rope_index` for GLM4V

- Status: merged `2026-02-01`, merge commit `4ea4f2`, about `+526/-86`, 2 files.
- Motivation: GLM4V `get_rope_index` became expensive for long multimodal inputs. PR benchmarks report speedups from about 12 percent to multiple times faster on longer token lengths.
- Key implementation: preallocate token-type tracking, move `attention_mask` once, use device-local ranges, group consecutive modalities, avoid repeated CPU `.item()` paths, and compute final MRoPE deltas with tensor reductions.
- Key code fragment:

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
idx_mask = curr_mask == 1
position_ids[..., i, idx_mask] = llm_positions.to(position_ids.device)
```

```python
max_position_ids = position_ids.amax(dim=0, keepdim=False)
mrope_position_deltas = max_position_ids.amax(-1, keepdim=True) + 1 - attention_mask.shape[-1]
```

- Validation signal: PR body notes lmms-eval no drop.
- Risk to preserve: MRoPE speed changes must be checked with image, multi-image, and video token layouts.

### [#17582](https://github.com/sgl-project/sglang/pull/17582) - GLM-OCR support

- Status: merged `2026-01-27`, merge commit `7106f6c`, about `+679/-29`, docs/config/model/processor files.
- Motivation: add first-class GLM-OCR support, including the OCR model architecture, processor registration, Transformers 5.x requirements, and speculative NextN support.
- Key implementation: register `GlmOcrForConditionalGeneration`, add `glm_ocr.py` and `glm_ocr_nextn.py`, build `GlmOcrVisionModel` on top of GLM4V vision pieces, use `VisionAttention` with head-size QK normalization, map draft architecture to `GlmOcrForConditionalGenerationNextN`, and teach the GLM4V processor to register OCR.
- Key code fragment:

```python
"GlmOcrForConditionalGeneration",
```

```python
self.attn = VisionAttention(
    embed_dim=dim,
    num_heads=num_heads,
    qkv_bias=attn_qkv_bias,
    qk_normalization_by_head_size=True,
    flatten_batch=True,
)
```

```python
class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    self.visual = GlmOcrVisionModel(...)
    self.model = Glm4Model(...)
```

```python
if is_draft_model and self.hf_config.architectures[0] in [
    "GlmOcrForConditionalGeneration",
]:
    self.hf_config.architectures[0] = "GlmOcrForConditionalGenerationNextN"
```

```python
models = [
    Glm4vForConditionalGeneration,
    Glm4vMoeForConditionalGeneration,
    GlmOcrForConditionalGeneration,
]
```

- Validation signal: official GLM-OCR docs later add OCRBench and OmniDocBench references plus EAGLE/MTP launch flags.
- Risk to preserve: OCR must be validated with OCR prompts and MTP/NextN, not only with image captioning.

### [#18885](https://github.com/sgl-project/sglang/pull/18885) - GLM-4V processor registration when GLM-OCR is unavailable

- Status: merged `2026-02-16`, merge commit `206accd`, `+12/-4`, 1 file.
- Motivation: `#17582` added an unconditional `GlmOcrForConditionalGeneration` import. In environments without `transformers.models.glm_ocr`, that import made the whole GLM4V processor module fail, dropping GLM-4.1V and GLM-4.5V registrations too.
- Key implementation: wrap only the OCR import in `try/except ImportError` and filter `None` from the processor model list.
- Key code fragment:

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

- Validation signal: PR body cites nightly failures where GLM4V models had no registered processor or were not evaluated.
- Risk to preserve: optional OCR dependency must not break non-OCR GLM VLM serving.

### [#20033](https://github.com/sgl-project/sglang/pull/20033) - Replace Conv3D projection with Linear for GLM4V

- Status: merged `2026-03-08`, merge commit `97a2a9b`, about `+192/-9`, 2 files.
- Motivation: the GLM4V patch embedding uses a Conv3D over flattened patch tensors, which can be represented as a single linear projection and benchmarked faster.
- Key implementation: add a linear layer with `C*T*P*P` input features, copy Conv3D weights into the linear layer after weight loading, delete the original projection, update `dtype/device` properties, and add a benchmark/regression test comparing Conv3D and Linear.
- Key code fragment:

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
return self.linear(x)
```

```python
self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Validation signal: added CPU close test and CUDA benchmark test for patch embedding; PR body reports no lmms-eval drop.
- Risk to preserve: every loader override that owns a GLM visual module must either call the copy or use the later shared `Conv3dLayer` abstraction.

### [#20282](https://github.com/sgl-project/sglang/pull/20282) - Add `Conv2dLayer`/`Conv3dLayer`

- Status: merged `2026-03-15`, merge commit `1c456a0`, about `+704/-90`, 18 files.
- Motivation: PyTorch 2.9.1 with CuDNN older than 9.15 had a Conv3D bug, and patch-embedding Conv2D/Conv3D can be much faster as unfold plus linear when kernel equals stride.
- Key implementation: introduce `sglang/srt/layers/conv.py`, detect when convolution is equivalent to unfold plus `F.linear`, migrate GLM4V/Qwen VLM and many 2D patch embedders, and remove the global server-arg compatibility check.
- Key code fragment:

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

- Validation signal: added `test/unit/test_conv_layer.py` for Conv2D/Conv3D behavior.
- Risk to preserve: this supersedes the one-off GLM4V Conv3D-to-Linear trick; if both exist, check loader order and whether `proj` or `linear` still exists.

### [#20463](https://github.com/sgl-project/sglang/pull/20463) - GLM-4.6V vision regression in MoE/OCR loaders

- Status: merged `2026-03-14`, merge commit `c330b68`, `+6`, 2 files; later reverted by `#20740`.
- Motivation: `#20033` added `copy_conv3d_weight_to_linear()` to `glm4v.py`, but `glm4v_moe.py` and `glm_ocr.py` have their own `load_weights()` overrides. Their linear projection could remain random, producing visual embeddings unrelated to the image.
- Key implementation: call the copy method at the end of MoE and OCR loaders, guarded by `is_nextn` so draft-only loads without `visual` do not crash.
- Key code fragment:

```python
if not is_nextn:
    self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Validation signal: PR body reports GLM-4.6V-FP8 on B200 TP=4 correctly describing image content after the fix and root cause confirmed by bisect.
- Risk to preserve: because this was later reverted, treat it as a historical warning. The durable fix path is the shared Conv layer design and loader-aware validation, not blindly reapplying this call.

### [#20740](https://github.com/sgl-project/sglang/pull/20740) - Revert `#20463`

- Status: merged `2026-03-18`, merge commit `f15b333`, `-6`, 2 files.
- Motivation: revert the direct MoE/OCR copy call from `#20463` per maintainer request after the Conv-layer migration changed the intended fix surface.
- Key implementation: remove `self.visual.patch_embed.copy_conv3d_weight_to_linear()` from `glm4v_moe.py` and `glm_ocr.py`.
- Key code fragment:

```python
# removed from MoE/OCR loaders:
# if not is_nextn:
#     self.visual.patch_embed.copy_conv3d_weight_to_linear()
```

- Validation signal: establishes current-main state: the loader copy call is not present in MoE/OCR.
- Risk to preserve: history must mention both PRs. Otherwise readers will think `#20463` is still the final solution.

### [#21134](https://github.com/sgl-project/sglang/pull/21134) - GLM-V/OCR Transformers 5.x and MTP omission fix

- Status: merged `2026-03-23`, merge commit `fcaad42`, about `+16/-9`, 3 files.
- Motivation: Transformers 5.x moved some GLM fields into `text_config`; MTP loading logic missed that field, and GLM-OCR used a numerically coincidental but semantically wrong connector dimension.
- Key implementation: read `num_nextn_predict_layers` from `hf_config.text_config` when present, normalize `language_model.` and `model.visual.` weight names before MTP/decoder mapping, and pass OCR text config into `GlmOcrVisionModel` so the merger uses `text_config.intermediate_size`.
- Key code fragment:

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
self.merger = GlmOcrVisionPatchMerger(
    d_model=vision_config.out_hidden_size,
    context_dim=text_config.intermediate_size,
)
```

- Validation signal: fixes both acceptance-rate-sensitive MTP omission and GLM-OCR config-field drift.
- Risk to preserve: future GLM-OCR changes must treat `text_config` as authoritative when available.

## Open/Unmerged PR Radar

### [#9349](https://github.com/sgl-project/sglang/pull/9349) - GLM-4.5V FP8 fused-MoE tuning

- Status: open, about `+153/-4`, 2 files.
- Motivation: add MoE kernel-generation support for GLM-4.5V FP8, specifically the GLM4V-MoE shape.
- Key implementation: teach the fused-MoE tuning script to read expert count, top-k, and intermediate size from `config.text_config` when architecture is `Glm4vMoeForConditionalGeneration`; add an L40S FP8 config file for `E=128,N=352`.
- Key code fragment:

```python
elif config.architectures[0] in [
    "Glm4MoeForCausalLM",
    "Glm4vMoeForConditionalGeneration",
]:
    cfg_source = config.text_config if is_glm4v_moe else config
```

```json
"BLOCK_SIZE_M": 16,
"BLOCK_SIZE_N": 64,
"BLOCK_SIZE_K": 128
```

- Validation signal: no benchmark was included in the PR body.
- Risk to track: useful for GLM-4.5V FP8 tuning, but do not treat it as merged behavior until CI and benchmark evidence exist.

### [#14662](https://github.com/sgl-project/sglang/pull/14662) - GLM4.6V ktransformers support

- Status: open, `+8`, 1 file.
- Motivation: expose GLM4.6V expert-layout metadata to ktransformers/expert-location integration.
- Key implementation: add `get_model_config_for_expert_location` on `Glm4vMoeForConditionalGeneration` and return logical expert metadata from `config.text_config`.
- Key code fragment:

```python
return ModelConfigForExpertLocation(
    num_layers=config.text_config.num_hidden_layers,
    num_logical_experts=config.text_config.n_routed_experts,
    num_groups=None,
)
```

- Validation signal: no accuracy or performance evidence in the PR body.
- Risk to track: this is routing/placement metadata, not a vision feature fix; validate MoE routing and expert placement separately.

### [#19728](https://github.com/sgl-project/sglang/pull/19728) - ROCm GLM-4.5V-FP8 startup

- Status: open, about `+104/-4`, 4 files.
- Motivation: GLM-4.5V-FP8 failed to start on MI300X with ROCm when AITER was disabled. Root causes were unpadded MoE weights being rejected under global padding and native HIP FP8 fallback copying unpadded data into padded output/scale buffers.
- Key implementation: disable effective MoE hidden-size padding when runtime hidden shape already matches `w1.shape[2]`; add a HIP helper to copy 2D tensors into padded destinations and fill padded rows; add ROCm-specific regression tests.
- Key code fragment:

```python
elif hidden_states.shape[1] == w1.shape[2]:
    padded_size = 0
```

```python
def _copy_with_optional_row_padding(dst, src, pad_value=0.0):
    dst[: src.shape[0]].copy_(src)
    if dst.shape[0] > src.shape[0]:
        dst[src.shape[0] :].fill_(pad_value)
```

- Validation signal: PR body reports targeted ROCm tests passed and an MI300X end-to-end launch with `GET /model_info` and chat completion success.
- Risk to track: open PR; if accepted, update AMD GLM-4.5V docs and FP8 fallback tests.

### [#22961](https://github.com/sgl-project/sglang/pull/22961) - NPU GLM-4.5V

- Status: open, about `+17/-5`, 1 file.
- Motivation: support GLM-4.5V on Ascend/NPU where the fused `split_qkv_rmsnorm_rope` kernel can run with or without QK norm.
- Key implementation: in `glm4_moe.py`, pass norm weights/bias/epsilon only when `use_qk_norm` is true; otherwise pass `None` values to the NPU split-QKV/RMSNorm/RoPE kernel.
- Key code fragment:

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

```python
q, k, v = split_qkv_rmsnorm_rope(
    qkv,
    self.rotary_emb.position_sin,
    self.rotary_emb.position_cos,
    self.q_size,
    self.kv_size,
    self.head_dim,
    eps=eps,
    q_weight=q_weight,
    k_weight=k_weight,
)
```

- Validation signal: PR body reports MMMU accuracy `0.2802`, invalid `0.000`, latency `89.380s`, output throughput `33.565 token/s`.
- Risk to track: touches GLM text MoE attention, but the motivation is GLM-4.5V platform enablement; verify both VLM and text paths.

## Cookbook and Public-Doc Evidence

- SGLang docs `docs/basic_usage/glmv.mdx`: FP8/BF16 launch, `--keep-mm-feature-on-device`, `--mm-attention-backend`, `--mm-max-concurrent-calls`, `--mm-enable-dp-encoder`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `SGLANG_VLM_CACHE_SIZE_MB=0`, and GLM thinking budget through a custom logit processor.
- SGLang GLM-4.5V cookbook: hardware support includes NVIDIA B200/H100/H200 and AMD MI300X/MI325X/MI355X; TP=8 requires `--mm-enable-dp-encoder` because the vision attention has 12 heads.
- SGLang GLM-4.6V cookbook: 128K context, native multimodal function calling, visual document understanding, frontend replication, and video inputs.
- SGLang GLM-OCR cookbook: EAGLE/MTP launch recipe and OCRBench/OmniDocBench validation cues.
- SGLang GLM-Glyph cookbook: belongs near GLM multimodal docs because its deployment generator and GLM parser flags are maintained with the same GLM cookbook surface.
- LMSYS blog "GLM-4.5 Meets SGLang" (`2025-07-31`): establishes the GLM parser, tool-call parser, MTP/EAGLE, FP8 variants, and MoE architecture background that later VLM/OCR docs reuse.
- sgl-cookbook [#95](https://github.com/sgl-project/sgl-cookbook/pull/95): GLM-4.5V AMD MI300X/MI325X/MI355X support.
- sgl-cookbook [#131](https://github.com/sgl-project/sgl-cookbook/pull/131): MI325X support for GLM-4.5V and GLM-4.6V.
- sgl-cookbook [#136](https://github.com/sgl-project/sgl-cookbook/pull/136): GLM-OCR cookbook.

## Validation Notes

- GLM VLM/OCR paths are sensitive to Transformers field drift; prefer `getattr(config, "text_config", config)` style defensive reads.
- OCR validation needs OCR-specific prompts and OCRBench-style checks, not only image captioning.
- Vision encoder DP/PP changes need a no-DP baseline and PP-stage loader inspection.
- FP8/ROCm/NPU changes must include platform-startup tests because failures often occur during graph capture or kernel dispatch rather than normal Python execution.
- `#20463` and `#20740` must be read together: the former explains a real regression mechanism, while the latter defines current-main behavior.

<!-- MODEL_PR_DIFF_AUDIT:START reference -->

# SGLANG GLM VLM / OCR PR Diff Audit Reference

This reference is rebuilt from the same audited PR metadata used by `model-pr-optimization-history`. It is intentionally concise but keeps a file-level diff digest for every indexed PR.

## Timeline

| Created | PR | State | Title | Code surface | Main diff files |
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

## Diff Cards

### PR #8798 - Support glm4.1v and glm4.5v

- Link: https://github.com/sgl-project/sglang/pull/8798
- Status/date: `merged`, created 2025-08-05, merged 2025-08-09; author `byjiang1996`.
- Diff scope read: `21` files, `+1584/-19`; areas: model wrapper, MoE/router, multimodal/processor, tests/benchmarks, docs/config; keywords: vision, attention, config, processor, test, cache, cuda, kv, lora, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` added +589/-0 (589 lines); hunks: +import logging; symbols: Glm4vRMSNorm, forward, Glm4vVisionMLP, __init__
  - `python/sglang/srt/models/glm4v_moe.py` added +400/-0 (400 lines); hunks: +import logging; symbols: Glm4vMoeForConditionalGeneration, __init__, determine_num_fused_shared_experts, load_weights
  - `python/sglang/srt/layers/rotary_embedding.py` modified +230/-1 (231 lines); hunks: # Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py; def __init__(; symbols: __init__, forward, get_rope_index, get_rope_index_glm4v
  - `python/sglang/srt/multimodal/processors/glm4v.py` added +132/-0 (132 lines); hunks: +import re; symbols: Glm4vImageProcessor, __init__, preprocess_video, process_mm_data_async
  - `test/srt/test_jinja_template_utils.py` modified +80/-0 (80 lines); hunks: def test_detect_empty_template(self):; symbols: test_detect_empty_template, test_detect_msg_content_pattern, with, test_detect_m_content_pattern
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: vision, attention, config, processor, test, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9059 - [GLM4.1V and GLM4.5V] Add vision transformer num_dummy_head support: max tp=4 -> max tp=8

- Link: https://github.com/sgl-project/sglang/pull/9059
- Status/date: `merged`, created 2025-08-11, merged 2025-08-18; author `byjiang1996`.
- Diff scope read: `9` files, `+150/-102`; areas: model wrapper, attention/backend, MoE/router, multimodal/processor, tests/benchmarks; keywords: attention, config, vision, kv, quant, benchmark, moe, triton, expert, processor.
- Code diff details:
  - `python/sglang/srt/layers/attention/vision_utils.py` added +65/-0 (65 lines); hunks: +"""Utility functions for vision attention layers."""; symbols: update_vit_attn_dummy_heads_config, pad_vit_attn_dummy_heads
  - `python/sglang/srt/models/glm4v.py` modified +52/-1 (53 lines); hunks: from sglang.srt.hf_transformers_utils import get_processor; def __init__(; symbols: __init__, __init__, get_video_feature, _update_hf_config
  - `python/sglang/srt/models/internvl.py` modified +4/-49 (53 lines); hunks: from transformers.activations import ACT2FN; def __init__(; symbols: __init__, __init__, _update_vision_config, pixel_shuffle
  - `python/sglang/srt/models/interns1.py` modified +5/-46 (51 lines); hunks: from torch import nn; def __init__(; symbols: __init__, __init__, _update_hf_config, pixel_shuffle
  - `benchmark/mmmu/bench_hf.py` modified +6/-2 (8 lines); hunks: def eval_mmmu(args):; symbols: eval_mmmu
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`; keywords observed in patches: attention, config, vision, kv, quant, benchmark. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/attention/vision_utils.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/internvl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9245 - Set the default attention backend for GLM-4.5v to fa3

- Link: https://github.com/sgl-project/sglang/pull/9245
- Status/date: `merged`, created 2025-08-15, merged 2025-08-17; author `zifeitong`.
- Diff scope read: `1` files, `+1/-0`; areas: misc; keywords: config, moe.
- Code diff details:
  - `python/sglang/srt/utils.py` modified +1/-0 (1 lines); hunks: def is_fa3_default_architecture(hf_config):; symbols: is_fa3_default_architecture
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/utils.py`; keywords observed in patches: config, moe. Impact reading: the patch is in miscellaneous paths; infer the actual impact from the touched files.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9349 - Add support for GLM 4.5V FP8

- Link: https://github.com/sgl-project/sglang/pull/9349
- Status/date: `open`, created 2025-08-19; author `pakjoeng`.
- Diff scope read: `2` files, `+153/-4`; areas: MoE/router, quantization, kernel, tests/benchmarks, docs/config; keywords: config, moe, triton, benchmark, expert, fp8, topk.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json` added +146/-0 (146 lines); hunks: +{
  - `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py` modified +7/-4 (11 lines); hunks: def main(args: argparse.Namespace):; symbols: main
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`; keywords observed in patches: config, moe, triton, benchmark, expert, fp8. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_4_0/E=128,N=352,device_name=NVIDIA_L40S,dtype=fp8_w8a8.json`, `benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9554 - Fix GLM45v launch server cuda torch compile bug

- Link: https://github.com/sgl-project/sglang/pull/9554
- Status/date: `merged`, created 2025-08-24, merged 2025-08-25; author `byjiang1996`.
- Diff scope read: `1` files, `+1/-0`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-0 (1 lines); hunks: def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:; symbols: get_video_feature, get_input_embeddings, forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_5_vl.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_5_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #9884 - [Bug Fix] Fix Glm4vVisionBlock norm

- Link: https://github.com/sgl-project/sglang/pull/9884
- Status/date: `merged`, created 2025-09-01, merged 2025-09-05; author `sdpkjc`.
- Diff scope read: `2` files, `+4/-4`; areas: model wrapper; keywords: config, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +3/-2 (5 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4v.py` modified +1/-2 (3 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/glm4v.py`; keywords observed in patches: config, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/glm4v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10147 - Fix: (glm4v) Add missing field

- Link: https://github.com/sgl-project/sglang/pull/10147
- Status/date: `merged`, created 2025-09-08, merged 2025-09-08; author `JustinTong0323`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper; keywords: config, eagle.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +3/-0 (3 lines); hunks: def __init__(; symbols: __init__, get_image_feature
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`; keywords observed in patches: config, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #10228 - Add self.capture_aux_hidden_states For GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/10228
- Status/date: `merged`, created 2025-09-09, merged 2025-09-14; author `zRzRzRzRzRzRzR`.
- Diff scope read: `1` files, `+3/-0`; areas: model wrapper, MoE/router; keywords: config, eagle, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: def __init__(; symbols: __init__, determine_num_fused_shared_experts
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v_moe.py`; keywords observed in patches: config, eagle, expert, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11166 - Tiny move files to utils folder

- Link: https://github.com/sgl-project/sglang/pull/11166
- Status/date: `merged`, created 2025-10-02, merged 2025-10-03; author `fzyzcjy`.
- Diff scope read: `66` files, `+91/-79`; areas: model wrapper, MoE/router, kernel, multimodal/processor, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, attention, processor, test, benchmark, cache, cuda, expert, lora, moe.
- Code diff details:
  - `test/srt/test_tokenizer_manager.py` modified +12/-4 (16 lines); hunks: def setUp(self):; def setUp(self):; symbols: setUp, setUp, setUp, setUp
  - `python/sglang/srt/managers/tp_worker.py` modified +6/-6 (12 lines); hunks: from sglang.srt.configs.model_config import ModelConfig; PPProxyTensors,
  - `python/sglang/srt/managers/scheduler.py` modified +5/-5 (10 lines); hunks: ); set_random_seed,
  - `python/sglang/srt/managers/tokenizer_manager.py` modified +5/-5 (10 lines); hunks: from sglang.srt.aio_rwlock import RWLock; get_zmq_socket,
  - `python/sglang/srt/configs/model_config.py` modified +4/-4 (8 lines); hunks: from transformers import PretrainedConfig
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/managers/scheduler.py`; keywords observed in patches: config, attention, processor, test, benchmark, cache. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `test/srt/test_tokenizer_manager.py`, `python/sglang/srt/managers/tp_worker.py`, `python/sglang/srt/managers/scheduler.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11388 - Replace pad with cat for better performance

- Link: https://github.com/sgl-project/sglang/pull/11388
- Status/date: `merged`, created 2025-10-09, merged 2025-10-10; author `yuan-luo`.
- Diff scope read: `5` files, `+5/-5`; areas: model wrapper; keywords: n/a.
- Code diff details:
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +1/-1 (2 lines); hunks: def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:; symbols: forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/models/qwen2_vl.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-1 (2 lines); hunks: def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`; keywords observed in patches: n/a. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/dots_vlm_vit.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #11922 - [lint] improve ruff check

- Link: https://github.com/sgl-project/sglang/pull/11922
- Status/date: `merged`, created 2025-10-21, merged 2025-10-22; author `hnyls2002`.
- Diff scope read: `19` files, `+73/-31`; areas: model wrapper, attention/backend, MoE/router, quantization, kernel, scheduler/runtime, tests/benchmarks, docs/config; keywords: config, quant, attention, benchmark, cache, kv, triton, doc, expert, flash.
- Code diff details:
  - `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py` modified +20/-19 (39 lines); hunks: PrecisionConfig,; def triton_kernel_fused_experts(; symbols: triton_kernel_fused_experts, triton_kernel_fused_experts, triton_kernel_fused_experts_with_bias, triton_kernel_fused_experts_with_bias
  - `python/sglang/srt/utils/common.py` modified +10/-2 (12 lines); hunks: import threading; from multiprocessing.reduction import ForkingPickler; symbols: monkey_patch_vllm_gguf_config, get_quant_method_with_embedding_replaced, direct_register_custom_op
  - `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` modified +7/-0 (7 lines); hunks: from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
  - `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` modified +4/-1 (5 lines); hunks: ); class PrefillMetadata:; symbols: PrefillMetadata:, FlashInferMhaChunkKVRunner:, __init__
  - `.pre-commit-config.yaml` modified +3/-1 (4 lines); hunks: repos:
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/utils/common.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; keywords observed in patches: config, quant, attention, benchmark, cache, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; scheduler/runtime/cache code changed; verify continuous batching, spec/PD/DP, cache lifetime, and exceptional branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py`, `python/sglang/srt/utils/common.py`, `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #12117 - GLM-4-0414 and GLM-4.1V Code Refactor

- Link: https://github.com/sgl-project/sglang/pull/12117
- Status/date: `merged`, created 2025-10-25, merged 2025-10-27; author `zRzRzRzRzRzRzR`.
- Diff scope read: `4` files, `+679/-173`; areas: model wrapper, MoE/router; keywords: config, quant, attention, cache, cuda, eagle, kv, processor, triton, vision.
- Code diff details:
  - `python/sglang/srt/models/glm4.py` modified +391/-77 (468 lines); hunks: # Modeling from:; def __init__(; symbols: Glm4MLP, __init__, forward, Glm4Attention
  - `python/sglang/srt/models/glm4v.py` modified +196/-55 (251 lines); hunks: +# Copyright 2023-2024 SGLang Team; from sglang.srt.layers.pooler import Pooler, PoolingType; symbols: __init__, forward, Glm4vVisionBlock, Glm4vVisionBlock
  - `python/sglang/srt/layers/rotary_embedding.py` modified +92/-40 (132 lines); hunks: def _triton_mrope_forward(; def _triton_mrope_forward(; symbols: _triton_mrope_forward, _triton_mrope_forward, triton_mrope, triton_mrope
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-1 (1 lines); hunks: def __init__(; symbols: __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: config, quant, attention, cache, cuda, eagle. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #13228 - Cleanup vision attention related codes

- Link: https://github.com/sgl-project/sglang/pull/13228
- Status/date: `merged`, created 2025-11-13, merged 2025-11-16; author `JustinTong0323`.
- Diff scope read: `15` files, `+4/-142`; areas: model wrapper, MoE/router; keywords: config, kv, quant, attention, flash, vision, triton.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +1/-26 (27 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +1/-26 (27 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/models/qwen3_vl.py` modified +1/-23 (24 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
  - `python/sglang/srt/models/clip.py` modified +0/-13 (13 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
  - `python/sglang/srt/models/qwen2_vl.py` modified +0/-13 (13 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; keywords observed in patches: config, kv, quant, attention, flash, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen2_5_vl.py`, `python/sglang/srt/models/qwen3_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14097 - support GLM-V vision model dp

- Link: https://github.com/sgl-project/sglang/pull/14097
- Status/date: `merged`, created 2025-11-28, merged 2025-12-05; author `zRzRzRzRzRzRzR`.
- Diff scope read: `4` files, `+91/-52`; areas: model wrapper, MoE/router, tests/benchmarks; keywords: attention, config, kv, moe, processor, quant, test, vision.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +84/-50 (134 lines); hunks: from einops import rearrange; from sglang.srt.model_executor.forward_batch_info import ForwardBatch; symbols: __init__, __init__, __init__, forward
  - `python/sglang/srt/models/glm4.py` modified +3/-1 (4 lines); hunks: def __init__(; symbols: __init__
  - `python/sglang/srt/models/glm4_moe.py` modified +3/-1 (4 lines); hunks: def __init__(; symbols: __init__
  - `test/nightly/test_encoder_dp.py` modified +1/-0 (1 lines); hunks: SimpleNamespace(model="Qwen/Qwen2.5-VL-72B-Instruct", mmmu_accuracy=0.55),
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: attention, config, kv, moe, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/glm4.py`, `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14662 - [Glm46v] support ktransformers

- Link: https://github.com/sgl-project/sglang/pull/14662
- Status/date: `open`, created 2025-12-08; author `mrhaoxx`.
- Diff scope read: `1` files, `+8/-0`; areas: model wrapper, MoE/router; keywords: config, expert, moe, quant, triton.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +8/-0 (8 lines); hunks: from sglang.srt.layers.moe import get_moe_a2a_backend; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights, get_model_config_for_expert_location
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v_moe.py`; keywords observed in patches: config, expert, moe, quant, triton. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14720 - [GLM-4.6V] Support Pipeline Parallelism for GLM-4.6V & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/14720
- Status/date: `merged`, created 2025-12-09, merged 2025-12-10; author `yuan-luo`.
- Diff scope read: `4` files, `+66/-2`; areas: model wrapper, tests/benchmarks; keywords: test, mla, cuda, fp4, moe, spec, vision.
- Code diff details:
  - `test/srt/test_pp_single_node.py` modified +38/-0 (38 lines); hunks: from sglang.test.test_utils import (; def test_chunked_prefill_with_small_bs(self):; symbols: test_chunked_prefill_with_small_bs, TestGLM41VPPAccuracy, setUpClass, tearDownClass
  - `python/sglang/srt/models/glm4v.py` modified +24/-1 (25 lines); hunks: general_mm_embed_routine,; def forward(; symbols: forward, forward, load_weights, load_weights
  - `python/sglang/test/test_utils.py` modified +3/-0 (3 lines); hunks: DEFAULT_MODEL_NAME_FOR_TEST_MLA = "lmsys/sglang-ci-dsv3-test"
  - `test/srt/run_suite.py` modified +1/-1 (2 lines); hunks: TestFile("test_gpt_oss_4gpu.py", 300),
- Optimization/support interpretation: The concrete diff surface is `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/test/test_utils.py`; keywords observed in patches: test, mla, cuda, fp4, moe, spec. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/srt/test_pp_single_node.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/test/test_utils.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14927 - [CI]add nightly CI for glm4v_moe arch model

- Link: https://github.com/sgl-project/sglang/pull/14927
- Status/date: `merged`, created 2025-12-11, merged 2025-12-12; author `zminglei`.
- Diff scope read: `1` files, `+3/-0`; areas: tests/benchmarks; keywords: fp8, test.
- Code diff details:
  - `test/nightly/test_vlms_mmmu_eval.py` modified +3/-0 (3 lines); hunks: ): ModelEvalMetrics(0.310, 16.7),
- Optimization/support interpretation: The concrete diff surface is `test/nightly/test_vlms_mmmu_eval.py`; keywords observed in patches: fp8, test. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/nightly/test_vlms_mmmu_eval.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #14998 - add transformers version validation for glm-4.6v moe models

- Link: https://github.com/sgl-project/sglang/pull/14998
- Status/date: `merged`, created 2025-12-12, merged 2025-12-13; author `yhyang201`.
- Diff scope read: `1` files, `+37/-0`; areas: docs/config; keywords: attention, config, moe, quant, vision.
- Code diff details:
  - `python/sglang/srt/configs/model_config.py` modified +37/-0 (37 lines); hunks: def __init__(; def _verify_dual_chunk_attention_config(self) -> None:; symbols: __init__, _verify_dual_chunk_attention_config, _verify_transformers_version, _get_hf_eos_token_id
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/configs/model_config.py`; keywords observed in patches: attention, config, moe, quant, vision. Impact reading: docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/configs/model_config.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15205 - [VLM] Support cos sin cache for Qwen3-VL & GLM-4.1V

- Link: https://github.com/sgl-project/sglang/pull/15205
- Status/date: `merged`, created 2025-12-15, merged 2025-12-18; author `yuan-luo`.
- Diff scope read: `4` files, `+100/-80`; areas: model wrapper, attention/backend, multimodal/processor; keywords: cache, vision, config, processor, quant, attention.
- Code diff details:
  - `python/sglang/srt/models/glm4v.py` modified +34/-50 (84 lines); hunks: from sglang.srt.layers.logits_processor import LogitsProcessor; def forward(; symbols: forward, forward, forward, Glm4vVisionRotaryEmbedding
  - `python/sglang/srt/models/qwen3_vl.py` modified +41/-20 (61 lines); hunks: import torch.nn as nn; from sglang.srt.layers.logits_processor import LogitsProcessor; symbols: forward, __init__, dtype, device
  - `python/sglang/srt/layers/attention/vision.py` modified +20/-10 (30 lines); hunks: def forward(; def forward(; symbols: forward, forward
  - `python/sglang/srt/layers/rotary_embedding.py` modified +5/-0 (5 lines); hunks: def get_cos_sin_with_position(self, positions):; symbols: get_cos_sin_with_position, get_cos_sin, forward_native
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`; keywords observed in patches: cache, vision, config, processor, quant, attention. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/layers/attention/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #15434 - Convert cu_seqlens to CPU for npu_flash_attention_unpad operator

- Link: https://github.com/sgl-project/sglang/pull/15434
- Status/date: `merged`, created 2025-12-19, merged 2026-01-04; author `xiaobaicxy`.
- Diff scope read: `9` files, `+36/-13`; areas: model wrapper, attention/backend, MoE/router, multimodal/processor; keywords: attention, flash, vision, processor, config, cuda, quant, expert, moe.
- Code diff details:
  - `python/sglang/srt/models/qwen3_vl.py` modified +6/-3 (9 lines); hunks: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; def forward(; symbols: forward
  - `python/sglang/srt/models/glm4v.py` modified +5/-1 (6 lines); hunks: from sglang.srt.models.glm4 import Glm4Model; def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:; symbols: forward
  - `python/sglang/srt/models/paddleocr_vl.py` modified +4/-2 (6 lines); hunks: from sglang.srt.model_executor.forward_batch_info import ForwardBatch; def forward(; symbols: Projector, forward
  - `python/sglang/srt/models/qwen2_5_vl.py` modified +4/-2 (6 lines); hunks: from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model; def forward(; symbols: forward
  - `python/sglang/srt/models/dots_vlm_vit.py` modified +4/-1 (5 lines); hunks: from sglang.srt.distributed import parallel_state; def forward(; symbols: forward
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`; keywords observed in patches: attention, flash, vision, processor, config, cuda. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/qwen3_vl.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/models/paddleocr_vl.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17122 - [bugfix]GLM-4V model

- Link: https://github.com/sgl-project/sglang/pull/17122
- Status/date: `merged`, created 2026-01-15, merged 2026-04-01; author `KnightLTC`.
- Diff scope read: `3` files, `+38/-3`; areas: model wrapper, multimodal/processor, tests/benchmarks; keywords: attention, cuda, benchmark, cache, config, kv, processor, quant, test, vision.
- Code diff details:
  - `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py` added +33/-0 (33 lines); hunks: +import unittest; symbols: TestGLM4Models, test_vlm_mmmu_benchmark
  - `python/sglang/srt/models/glm4v.py` modified +2/-2 (4 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__
  - `python/sglang/srt/multimodal/processors/base_processor.py` modified +3/-1 (4 lines); hunks: def process_mm_data(; symbols: process_mm_data
- Optimization/support interpretation: The concrete diff surface is `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; keywords observed in patches: attention, cuda, benchmark, cache, config, kv. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/ascend/vlm_models/test_ascend_glm_4_5v.py`, `python/sglang/srt/models/glm4v.py`, `python/sglang/srt/multimodal/processors/base_processor.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17420 - [VLM] Optimize get_rope_index for GLM4v

- Link: https://github.com/sgl-project/sglang/pull/17420
- Status/date: `merged`, created 2026-01-20, merged 2026-02-01; author `yuan-luo`.
- Diff scope read: `2` files, `+526/-86`; areas: tests/benchmarks; keywords: attention, config, vision, benchmark, cuda, moe, test.
- Code diff details:
  - `benchmark/bench_rope/benchmark_rope_index.py` added +425/-0 (425 lines); hunks: +# This script benchmarks MRotaryEmbedding.get_rope_index_glm4v (GLM4V mrope index builder).; symbols: DummyVisionConfig:, DummyHFConfig:, calculate_stats, _sync
  - `python/sglang/srt/layers/rotary_embedding.py` modified +101/-86 (187 lines); hunks: def get_rope_index(; def get_rope_index(; symbols: get_rope_index, get_rope_index, get_rope_index, get_rope_index_glm4v
- Optimization/support interpretation: The concrete diff surface is `benchmark/bench_rope/benchmark_rope_index.py`, `python/sglang/srt/layers/rotary_embedding.py`; keywords observed in patches: attention, config, vision, benchmark, cuda, moe. Impact reading: tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `benchmark/bench_rope/benchmark_rope_index.py`, `python/sglang/srt/layers/rotary_embedding.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #17582 - [GLM-OCR] Support GLM-OCR Model

- Link: https://github.com/sgl-project/sglang/pull/17582
- Status/date: `merged`, created 2026-01-22, merged 2026-01-27; author `zRzRzRzRzRzRzR`.
- Diff scope read: `9` files, `+679/-29`; areas: model wrapper, attention/backend, multimodal/processor, docs/config; keywords: config, attention, spec, kv, processor, quant, vision, moe, cache, cuda.
- Code diff details:
  - `python/sglang/srt/models/glm_ocr.py` added +435/-0 (435 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: GlmOcrRMSNorm, GlmOcrVisionMLP, GlmOcrVisionBlock, __init__
  - `python/sglang/srt/models/glm_ocr_nextn.py` added +162/-0 (162 lines); hunks: +# Copyright 2023-2024 SGLang Team; symbols: GlmOcrModelNextN, __init__, forward, GlmOcrForConditionalGenerationNextN
  - `python/sglang/srt/layers/attention/vision.py` modified +49/-19 (68 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, _init_qk_norm
  - `python/sglang/srt/models/glm4.py` modified +18/-6 (24 lines); hunks: def __init__(; def __init__(; symbols: __init__, __init__, __init__, __init__
  - `python/sglang/srt/configs/model_config.py` modified +7/-1 (8 lines); hunks: def _config_draft_model(self):; def _verify_transformers_version(self):; symbols: _config_draft_model, _verify_transformers_version, is_generation_model
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/layers/attention/vision.py`; keywords observed in patches: config, attention, spec, kv, processor, quant. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; attention, KV cache, or backend selection changed; verify prefill/decode, page size, RoPE/MLA/MQA branches; multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching; docs or config changed; verify serve flags, defaults, and cookbook commands against runtime code.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm_ocr.py`, `python/sglang/srt/models/glm_ocr_nextn.py`, `python/sglang/srt/layers/attention/vision.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #18885 - Fix GLM-4V processor registration when glm_ocr is unavailable

- Link: https://github.com/sgl-project/sglang/pull/18885
- Status/date: `merged`, created 2026-02-16, merged 2026-02-16; author `alisonshao`.
- Diff scope read: `1` files, `+12/-4`; areas: multimodal/processor; keywords: config, moe, processor, spec.
- Code diff details:
  - `python/sglang/srt/multimodal/processors/glm4v.py` modified +12/-4 (16 lines); hunks: from sglang.srt.layers.rotary_embedding import MRotaryEmbedding; symbols: Glm4vImageProcessor, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/multimodal/processors/glm4v.py`; keywords observed in patches: config, moe, processor, spec. Impact reading: multimodal processor or media-token code changed; verify image/video/audio metadata, position ids, and batching.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/multimodal/processors/glm4v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #19728 - Fix ROCm GLM-4.5V-FP8 startup with unpadded MoE weights and padded FP8 fallback

- Link: https://github.com/sgl-project/sglang/pull/19728
- Status/date: `open`, created 2026-03-03; author `andyluo7`.
- Diff scope read: `4` files, `+104/-4`; areas: MoE/router, quantization, kernel, tests/benchmarks; keywords: fp8, quant, expert, moe, test, triton, cache, config, cuda, mla.
- Code diff details:
  - `test/registered/moe/test_fused_moe.py` modified +66/-0 (66 lines); hunks: import unittest; def test_various_configurations(self):; symbols: test_various_configurations, test_fp8_unpadded_weights_with_global_moe_padding
  - `python/sglang/srt/layers/quantization/fp8_kernel.py` modified +21/-4 (25 lines); hunks: def per_token_group_quant_mla_deep_gemm_masked_fp8(; def _native_dynamic_per_token_quant_fp8(output, input, scale):; symbols: per_token_group_quant_mla_deep_gemm_masked_fp8, _copy_with_optional_row_padding, _native_dynamic_per_token_quant_fp8, _native_dynamic_per_token_quant_fp8
  - `python/sglang/test/test_custom_ops.py` modified +11/-0 (11 lines); hunks: import pytest; def test_scaled_fp8_quant_with_padding(dtype) -> None:; symbols: test_scaled_fp8_quant_with_padding
  - `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` modified +6/-0 (6 lines); hunks: def fused_experts_impl(; symbols: fused_experts_impl
- Optimization/support interpretation: The concrete diff surface is `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`; keywords observed in patches: fp8, quant, expert, moe, test, triton. Impact reading: MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches; quantized loading or quantized kernels changed; verify scales, zero-points, checkpoint names, and fallback behavior; CUDA/Triton/C++ kernels or bindings changed; verify shape guards, dtype, device backend, and benchmark coverage; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/moe/test_fused_moe.py`, `python/sglang/srt/layers/quantization/fp8_kernel.py`, `python/sglang/test/test_custom_ops.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20033 - [VLM] Replace conv3d proj with linear for GLM4V

- Link: https://github.com/sgl-project/sglang/pull/20033
- Status/date: `merged`, created 2026-03-06, merged 2026-03-08; author `yuan-luo`.
- Diff scope read: `2` files, `+192/-9`; areas: model wrapper, tests/benchmarks; keywords: benchmark, config, cuda, test, vision.
- Code diff details:
  - `test/registered/vlm/test_patch_embed_perf.py` added +166/-0 (166 lines); hunks: +import os; symbols: ReferenceConv3dPatchEmbed, __init__, forward, _build_modules
  - `python/sglang/srt/models/glm4v.py` modified +26/-9 (35 lines); hunks: def __init__(; def __init__(; symbols: __init__, forward, copy_conv3d_weight_to_linear, forward
- Optimization/support interpretation: The concrete diff surface is `test/registered/vlm/test_patch_embed_perf.py`, `python/sglang/srt/models/glm4v.py`; keywords observed in patches: benchmark, config, cuda, test, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/registered/vlm/test_patch_embed_perf.py`, `python/sglang/srt/models/glm4v.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20282 - Add Conv2dLayer/Conv3dLayer to fix PyTorch 2.9.1 CuDNN Conv3d bug

- Link: https://github.com/sgl-project/sglang/pull/20282
- Status/date: `merged`, created 2026-03-10, merged 2026-03-15; author `yhyang201`.
- Diff scope read: `18` files, `+704/-90`; areas: model wrapper, tests/benchmarks; keywords: config, attention, vision, quant, cuda, lora, moe, spec, test.
- Code diff details:
  - `test/unit/test_conv_layer.py` added +363/-0 (363 lines); hunks: +import unittest; symbols: _copy_weights, TestConv2dLayer, test_basic_patch_embedding, test_enable_linear
  - `python/sglang/srt/layers/conv.py` added +300/-0 (300 lines); hunks: +"""; symbols: _tuplify, _check_enable_linear, _reverse_repeat_tuple, _compute_same_padding_for_pad
  - `python/sglang/srt/server_args.py` modified +0/-48 (48 lines); hunks: def check_server_args(self):; def check_server_args(self):; symbols: check_server_args, check_server_args, check_torch_2_9_1_cudnn_compatibility, check_lora_server_args
  - `python/sglang/srt/models/glm4v.py` modified +12/-27 (39 lines); hunks: from sglang.srt.layers.activation import SiluAndMul; def __init__(; symbols: __init__, copy_conv3d_weight_to_linear, forward, Glm4vPatchMerger
  - `python/sglang/srt/models/pixtral.py` modified +3/-2 (5 lines); hunks: from sglang.srt.layers.activation import SiluAndMul; class VisionTransformer(nn.Module):; symbols: VisionTransformer, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `test/unit/test_conv_layer.py`, `python/sglang/srt/layers/conv.py`, `python/sglang/srt/server_args.py`; keywords observed in patches: config, attention, vision, quant, cuda, lora. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; tests or benchmarks changed; use those cases as regression entry points instead of only checking model load.
- Risk and verification: Re-run the model path that exercises `test/unit/test_conv_layer.py`, `python/sglang/srt/layers/conv.py`, `python/sglang/srt/server_args.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20463 - [Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr

- Link: https://github.com/sgl-project/sglang/pull/20463
- Status/date: `merged`, created 2026-03-12, merged 2026-03-14; author `JustinTong0323`.
- Diff scope read: `2` files, `+6/-0`; areas: model wrapper, MoE/router; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +3/-0 (3 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +3/-0 (3 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #20740 - Revert "[Bugfix] Fix GLM-4.6V vision regression in glm4v_moe and glm_ocr"

- Link: https://github.com/sgl-project/sglang/pull/20740
- Status/date: `merged`, created 2026-03-17, merged 2026-03-18; author `mickqian`.
- Diff scope read: `2` files, `+0/-6`; areas: model wrapper, MoE/router; keywords: moe.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +0/-3 (3 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights
  - `python/sglang/srt/models/glm_ocr.py` modified +0/-3 (3 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; symbols: load_weights
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; keywords observed in patches: moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/models/glm_ocr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #21134 - [Bug Fix] GLM-V / GLM-OCR: field detection for transformers 5.x and MTP omission fix

- Link: https://github.com/sgl-project/sglang/pull/21134
- Status/date: `merged`, created 2026-03-22, merged 2026-03-23; author `zRzRzRzRzRzRzR`.
- Diff scope read: `3` files, `+16/-9`; areas: model wrapper, MoE/router; keywords: config, moe, expert, quant, vision.
- Code diff details:
  - `python/sglang/srt/models/glm4v_moe.py` modified +7/-7 (14 lines); hunks: def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=Fal; def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=F; symbols: load_weights, load_weights
  - `python/sglang/srt/model_loader/weight_utils.py` modified +5/-1 (6 lines); hunks: def maybe_add_mtp_safetensors(; symbols: maybe_add_mtp_safetensors
  - `python/sglang/srt/models/glm_ocr.py` modified +4/-1 (5 lines); hunks: from einops import rearrange; class GlmOcrVisionModel(Glm4vVisionModel):; symbols: GlmOcrVisionModel, __init__, __init__, __init__
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm_ocr.py`; keywords observed in patches: config, moe, expert, quant, vision. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4v_moe.py`, `python/sglang/srt/model_loader/weight_utils.py`, `python/sglang/srt/models/glm_ocr.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.

### PR #22961 - [NPU] Support GLM-4.5V

- Link: https://github.com/sgl-project/sglang/pull/22961
- Status/date: `open`, created 2026-04-16; author `zhsurpass`.
- Diff scope read: `1` files, `+17/-5`; areas: model wrapper, MoE/router; keywords: kv, moe.
- Code diff details:
  - `python/sglang/srt/models/glm4_moe.py` modified +17/-5 (22 lines); hunks: def forward_prepare(; symbols: forward_prepare
- Optimization/support interpretation: The concrete diff surface is `python/sglang/srt/models/glm4_moe.py`; keywords observed in patches: kv, moe. Impact reading: model wrapper, forward, or weight-loading code changed; verify architecture mapping, hidden-state shape, and weight-name mapping; MoE/router/top-k/expert logic changed; verify shared/routed experts plus EP/TP/DP and empty-token branches.
- Risk and verification: Re-run the model path that exercises `python/sglang/srt/models/glm4_moe.py`; then add the area-specific checks above, especially any changed tests/benchmarks and serving flags.


<!-- MODEL_PR_DIFF_AUDIT:END reference -->
