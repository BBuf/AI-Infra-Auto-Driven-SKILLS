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
